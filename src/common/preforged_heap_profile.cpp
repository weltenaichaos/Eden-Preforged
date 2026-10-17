// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "common/preforged_heap_profile.h"

#ifdef __ANDROID__

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include <cxxabi.h>
#include <dlfcn.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>
#include <unwind.h>

#include <fmt/format.h>

#include "common/logging.h"

// The app's own operator new and delete are replaced below. While sampling, about one
// allocation per MEAN_INTERVAL bytes is recorded with its call stack, and followed until it is
// freed, so the live samples show which code holds the heap.

namespace Common::Preforged::HeapProfile {
namespace {

constexpr double MEAN_INTERVAL = 1024.0 * 1024.0;
constexpr size_t MAX_FRAMES = 24;
// Frames of the sampler itself that come before the code that allocated.
constexpr size_t MAX_OWN_FRAMES = 6;
constexpr size_t PRINTED_FRAMES = 14;

constexpr size_t STACK_CAPACITY = size_t{1} << 14;
// Samples whose call stack no longer fits in the table are counted under this index.
constexpr uint32_t OTHER_STACK = STACK_CAPACITY;

constexpr size_t LIVE_BITS = 16;
constexpr size_t LIVE_CAPACITY = size_t{1} << LIVE_BITS;
constexpr size_t LIVE_MASK = LIVE_CAPACITY - 1;

// Counts live samples per hash of their address, so most frees can tell without a lock that
// they were not sampled. Small enough to stay in cache.
constexpr size_t FILTER_BITS = 18;
constexpr size_t FILTER_SIZE = size_t{1} << FILTER_BITS;

struct StackEntry {
    uint64_t hash; // 0 while unused
    uint32_t depth;
    std::array<char, 16> thread;
    std::array<uintptr_t, MAX_FRAMES> pcs;
};

struct LiveEntry {
    uintptr_t pointer; // 0 while unused
    uint64_t weight;
    uint32_t stack;
};

struct ThreadState {
    int64_t bytes_left;
    uint64_t random;
    bool started;
    bool busy;
};

std::atomic_bool sampling{false};
std::atomic_bool following{false};
std::atomic_bool hooked{false};

std::mutex mutex;
StackEntry stacks[STACK_CAPACITY + 1];
size_t stacks_used = 0;
LiveEntry live[LIVE_CAPACITY];
size_t live_count = 0;
uint64_t live_weight = 0;
uint64_t samples_taken = 0;
uint64_t samples_dropped = 0;
std::atomic<uint8_t> filter[FILTER_SIZE];

thread_local ThreadState thread_state{};

size_t FilterSlot(uintptr_t pointer) {
    return static_cast<size_t>(((static_cast<uint64_t>(pointer) >> 4) * 0x9E3779B97F4A7C15ULL) >>
                               (64 - FILTER_BITS));
}

size_t LiveSlot(uintptr_t pointer) {
    return static_cast<size_t>(((static_cast<uint64_t>(pointer) >> 4) * 0xC2B2AE3D27D4EB4FULL) >>
                               (64 - LIVE_BITS));
}

uint64_t NextRandom(ThreadState& state) {
    uint64_t x = state.random;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state.random = x;
    return x;
}

// Bytes until the next sample, exponentially distributed so every byte has the same chance.
int64_t NextInterval(ThreadState& state) {
    const double unit = static_cast<double>((NextRandom(state) >> 11) + 1) * 0x1.0p-53;
    return static_cast<int64_t>(-std::log(unit) * MEAN_INTERVAL) + 1;
}

struct Unwound {
    std::array<uintptr_t, MAX_FRAMES + MAX_OWN_FRAMES> pcs;
    size_t count;
};

_Unwind_Reason_Code CollectFrame(_Unwind_Context* context, void* arg) {
    auto* const unwound = static_cast<Unwound*>(arg);
    const uintptr_t pc = _Unwind_GetIP(context);
    if (pc == 0) {
        return _URC_END_OF_STACK;
    }
    unwound->pcs[unwound->count++] = pc;
    return unwound->count < unwound->pcs.size() ? _URC_NO_REASON : _URC_END_OF_STACK;
}

// Call with the mutex held.
uint32_t FindOrAddStack(uint64_t hash, const uintptr_t* pcs, size_t depth, const char* thread) {
    constexpr size_t mask = STACK_CAPACITY - 1;
    for (size_t slot = hash & mask, probe = 0; probe < STACK_CAPACITY;
         slot = (slot + 1) & mask, ++probe) {
        StackEntry& entry = stacks[slot];
        if (entry.hash == 0) {
            if (stacks_used >= STACK_CAPACITY * 3 / 4) {
                return OTHER_STACK;
            }
            entry.hash = hash;
            entry.depth = static_cast<uint32_t>(depth);
            std::copy_n(pcs, depth, entry.pcs.begin());
            std::memcpy(entry.thread.data(), thread, entry.thread.size());
            entry.thread.back() = '\0';
            ++stacks_used;
            return static_cast<uint32_t>(slot);
        }
        if (entry.hash == hash && entry.depth == depth &&
            std::equal(pcs, pcs + depth, entry.pcs.begin())) {
            return static_cast<uint32_t>(slot);
        }
    }
    return OTHER_STACK;
}

// Call with the mutex held.
void AddLive(uintptr_t pointer, uint64_t weight, uint32_t stack) {
    for (size_t slot = LiveSlot(pointer);; slot = (slot + 1) & LIVE_MASK) {
        LiveEntry& entry = live[slot];
        if (entry.pointer == pointer) {
            // Freed by a path that bypassed operator delete, then handed out again.
            live_weight -= entry.weight;
            entry.weight = weight;
            entry.stack = stack;
            live_weight += weight;
            return;
        }
        if (entry.pointer != 0) {
            continue;
        }
        if (live_count >= LIVE_CAPACITY * 3 / 4) {
            ++samples_dropped;
            return;
        }
        entry = LiveEntry{pointer, weight, stack};
        ++live_count;
        live_weight += weight;
        std::atomic<uint8_t>& count = filter[FilterSlot(pointer)];
        const uint8_t value = count.load(std::memory_order_relaxed);
        if (value != 0xFF) {
            count.store(static_cast<uint8_t>(value + 1), std::memory_order_relaxed);
        }
        return;
    }
}

// Call with the mutex held.
void RemoveLive(uintptr_t pointer) {
    size_t slot = LiveSlot(pointer);
    for (;; slot = (slot + 1) & LIVE_MASK) {
        if (live[slot].pointer == 0) {
            return;
        }
        if (live[slot].pointer == pointer) {
            break;
        }
    }
    live_weight -= live[slot].weight;
    --live_count;
    std::atomic<uint8_t>& count = filter[FilterSlot(pointer)];
    const uint8_t value = count.load(std::memory_order_relaxed);
    if (value != 0xFF && value != 0) {
        count.store(static_cast<uint8_t>(value - 1), std::memory_order_relaxed);
    }
    // Shift later entries of the same probe run back into the hole.
    size_t hole = slot;
    for (size_t next = (hole + 1) & LIVE_MASK; live[next].pointer != 0;
         next = (next + 1) & LIVE_MASK) {
        const size_t home = LiveSlot(live[next].pointer);
        if (((next - home) & LIVE_MASK) >= ((next - hole) & LIVE_MASK)) {
            live[hole] = live[next];
            hole = next;
        }
    }
    live[hole] = LiveEntry{};
}

[[gnu::noinline]] void SampleAllocation(void* pointer, size_t size, uintptr_t caller) {
    ThreadState& state = thread_state;
    if (state.busy) {
        return;
    }
    if (!state.started) {
        timespec now{};
        clock_gettime(CLOCK_MONOTONIC, &now);
        state.random = (static_cast<uint64_t>(gettid()) << 32) ^
                       static_cast<uint64_t>(now.tv_nsec) ^ 0x2545F4914F6CDD1DULL;
        if (state.random == 0) {
            state.random = 1;
        }
        state.bytes_left = NextInterval(state);
        state.started = true;
    }
    state.bytes_left -= static_cast<int64_t>(size);
    if (state.bytes_left > 0) {
        return;
    }
    state.busy = true;
    state.bytes_left = NextInterval(state);

    // An allocation of `size` bytes is sampled with this chance, so it stands for 1 / chance
    // allocations like it.
    const double bytes = static_cast<double>(size);
    const double chance = -std::expm1(-bytes / MEAN_INTERVAL);
    const uint64_t weight = chance > 0.0 ? static_cast<uint64_t>(bytes / chance)
                                         : static_cast<uint64_t>(MEAN_INTERVAL);

    Unwound unwound{};
    _Unwind_Backtrace(CollectFrame, &unwound);
    size_t first = 0;
    for (size_t i = 0; i < std::min(unwound.count, MAX_OWN_FRAMES); ++i) {
        if (unwound.pcs[i] == caller) {
            first = i;
            break;
        }
    }
    const size_t depth = std::min(unwound.count - first, MAX_FRAMES);
    const uintptr_t* const pcs = unwound.pcs.data() + first;
    uint64_t hash = 0xCBF29CE484222325ULL ^ depth;
    for (size_t i = 0; i < depth; ++i) {
        hash = (hash ^ pcs[i]) * 0x100000001B3ULL;
    }
    hash |= 1;
    char thread[16]{};
    prctl(PR_GET_NAME, thread);
    {
        std::scoped_lock lock{mutex};
        const uint32_t stack = FindOrAddStack(hash, pcs, depth, thread);
        AddLive(reinterpret_cast<uintptr_t>(pointer), weight, stack);
        ++samples_taken;
    }
    state.busy = false;
}

void FollowFree(void* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (filter[FilterSlot(address)].load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::scoped_lock lock{mutex};
    RemoveLive(address);
}

[[gnu::always_inline]] inline void* Allocate(std::size_t size, uintptr_t caller) {
    if (size == 0) {
        size = 1;
    }
    void* pointer;
    while ((pointer = std::malloc(size)) == nullptr) {
        const std::new_handler handler = std::get_new_handler();
        if (handler == nullptr) {
#if defined(__cpp_exceptions)
            throw std::bad_alloc();
#else
            std::abort();
#endif
        }
        handler();
    }
    if (sampling.load(std::memory_order_relaxed)) [[unlikely]] {
        hooked.store(true, std::memory_order_relaxed);
        SampleAllocation(pointer, size, caller);
    }
    return pointer;
}

[[gnu::always_inline]] inline void Free(void* pointer) {
    if (pointer != nullptr && following.load(std::memory_order_relaxed)) [[unlikely]] {
        FollowFree(pointer);
    }
    std::free(pointer);
}

std::string DescribeFrame(uintptr_t pc) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(pc), &info) == 0 || info.dli_fname == nullptr) {
        return fmt::format("{:#x}", pc);
    }
    const char* module = std::strrchr(info.dli_fname, '/');
    module = module ? module + 1 : info.dli_fname;
    std::string text =
        fmt::format("{}+{:#x}", module, pc - reinterpret_cast<uintptr_t>(info.dli_fbase));
    if (info.dli_sname != nullptr) {
        int status = 0;
        char* const demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
        std::string name = status == 0 && demangled ? demangled : info.dli_sname;
        std::free(demangled);
        if (name.size() > 200) {
            name.resize(197);
            name += "...";
        }
        text += fmt::format(" {}+{:#x}", name, pc - reinterpret_cast<uintptr_t>(info.dli_saddr));
    }
    return text;
}

} // Anonymous namespace

void Start() {
    following.store(true, std::memory_order_relaxed);
    sampling.store(true, std::memory_order_relaxed);
    // Make sure the app goes through the operator new in this file. A direct call, since the
    // compiler may drop a new expression whose result is unused.
    void* const probe = ::operator new(64);
    ::operator delete(probe);
    LOG_INFO(Common,
             "[Preforged] tracking heap allocations: one sample per {} KiB allocated on average, "
             "operator new {}",
             static_cast<size_t>(MEAN_INTERVAL) >> 10,
             hooked.load() ? "hooked" : "not hooked, nothing will be recorded");
}

void Stop() {
    sampling.store(false, std::memory_order_relaxed);
}

bool Sampling() {
    return sampling.load(std::memory_order_relaxed);
}

void Log(const char* when, size_t top) {
    if (!following.load(std::memory_order_relaxed)) {
        return;
    }
    // Allocated before taking the lock, since allocating may take a sample.
    std::vector<uint64_t> bytes(STACK_CAPACITY + 1);
    std::vector<uint32_t> counts(STACK_CAPACITY + 1);
    size_t live_samples;
    uint64_t total;
    uint64_t taken;
    uint64_t dropped;
    size_t stack_count;
    {
        std::scoped_lock lock{mutex};
        for (const LiveEntry& entry : live) {
            if (entry.pointer != 0) {
                bytes[entry.stack] += entry.weight;
                ++counts[entry.stack];
            }
        }
        live_samples = live_count;
        total = live_weight;
        taken = samples_taken;
        dropped = samples_dropped;
        stack_count = stacks_used;
    }

    static std::mutex report_mutex;
    static std::vector<uint64_t> previous_bytes;
    std::scoped_lock report_lock{report_mutex};
    if (previous_bytes.empty()) {
        previous_bytes.resize(STACK_CAPACITY + 1);
    }
    const auto growth = [&](uint32_t index) {
        return static_cast<int64_t>(bytes[index]) - static_cast<int64_t>(previous_bytes[index]);
    };

    std::vector<uint32_t> by_live;
    for (uint32_t index = 0; index <= STACK_CAPACITY; ++index) {
        if (bytes[index] != 0 || previous_bytes[index] != 0) {
            by_live.push_back(index);
        }
    }
    std::vector<uint32_t> by_growth = by_live;
    const size_t live_top = std::min(top, by_live.size());
    std::partial_sort(by_live.begin(), by_live.begin() + live_top, by_live.end(),
                      [&](uint32_t a, uint32_t b) { return bytes[a] > bytes[b]; });
    by_live.resize(live_top);
    std::sort(by_growth.begin(), by_growth.end(),
              [&](uint32_t a, uint32_t b) { return growth(a) > growth(b); });

    LOG_INFO(Common,
             "[Preforged] heap holders {}: about {} MiB of C++ heap in {} live samples, {} call "
             "stacks, {} samples taken, {} dropped",
             when, total >> 20, live_samples, stack_count, taken, dropped);

    // Looking up a symbol walks the whole symbol table, so names are kept between reports.
    static std::unordered_map<uintptr_t, std::string> frame_names;
    const auto frame_name = [&](uintptr_t pc) -> const std::string& {
        auto [it, inserted] = frame_names.try_emplace(pc);
        if (inserted) {
            it->second = DescribeFrame(pc);
        }
        return it->second;
    };
    const auto print = [&](const char* label, size_t rank, uint32_t index) {
        const StackEntry& entry = stacks[index];
        LOG_INFO(Common,
                 "[Preforged]   {} #{}: {} MiB live in {} samples, {:+} MiB since the last report, "
                 "first seen on thread '{}'",
                 label, rank, bytes[index] >> 20, counts[index], growth(index) / (1 << 20),
                 index == OTHER_STACK ? "(stacks that did not fit)" : entry.thread.data());
        if (index == OTHER_STACK) {
            return;
        }
        for (size_t frame = 0; frame < std::min<size_t>(entry.depth, PRINTED_FRAMES); ++frame) {
            LOG_INFO(Common, "[Preforged]       {}", frame_name(entry.pcs[frame]));
        }
    };
    for (size_t rank = 0; rank < by_live.size(); ++rank) {
        print("most held", rank + 1, by_live[rank]);
    }
    // Stacks that grew the most and are not listed above yet.
    size_t printed = 0;
    for (const uint32_t index : by_growth) {
        if (printed == 3 || growth(index) < (int64_t{16} << 20)) {
            break;
        }
        if (std::find(by_live.begin(), by_live.end(), index) != by_live.end()) {
            continue;
        }
        print("grew most", ++printed, index);
    }
    previous_bytes = bytes;
}

} // namespace Common::Preforged::HeapProfile

// Replacements for the global allocation functions; the other forms of new and delete in libc++
// call these.

[[gnu::noinline]] void* operator new(std::size_t size) {
    return Common::Preforged::HeapProfile::Allocate(
        size, reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}

[[gnu::noinline]] void* operator new[](std::size_t size) {
    return Common::Preforged::HeapProfile::Allocate(
        size, reinterpret_cast<uintptr_t>(__builtin_return_address(0)));
}

void operator delete(void* pointer) noexcept {
    Common::Preforged::HeapProfile::Free(pointer);
}

void operator delete[](void* pointer) noexcept {
    Common::Preforged::HeapProfile::Free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    Common::Preforged::HeapProfile::Free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    Common::Preforged::HeapProfile::Free(pointer);
}

#else

#include "common/logging.h"

namespace Common::Preforged::HeapProfile {

void Start() {
    LOG_INFO(Common, "[Preforged] tracking heap allocations is only available on Android");
}

void Stop() {}

bool Sampling() {
    return false;
}

void Log(const char*, size_t) {}

} // namespace Common::Preforged::HeapProfile

#endif
