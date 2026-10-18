// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include "common/cityhash.h"
#include "common/container/unordered_set.h"
#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/logging.h"
#include "common/preforged_debug.h"
#include "common/preforged_frame_stats.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "core/arm/dynarmic/arm_dynarmic_64.h"
#include "core/arm/dynarmic/preforged_cpu_code.h"
#include "core/core.h"
#include "core/hardware_properties.h"
#include "core/hle/kernel/k_process.h"
#include "core/memory.h"

namespace Core::Preforged::CpuCode {
namespace {

using Clock = std::chrono::steady_clock;

// The list file is a header, then one u64 record per compiled block, appended while the game
// runs. A record is the block's location (Dynarmic's A64 LocationDescriptor value) with its PC
// replaced by the offset from the start of the game's code, since the code moves by up to
// 16 MiB on each boot, and with the CPU core in bits 48 to 55.
constexpr u32 FileMagic = 0x42435045; // "EPCB"
constexpr u32 FileVersion = 1;
struct FileHeader {
    u32 magic;
    u32 version;
    u64 code_hash; ///< Hash of the game's executable code, which changes with updates and mods.
    u64 code_size;
};
static_assert(sizeof(FileHeader) == 24);

constexpr u64 PcMask = (u64{1} << 56) - 1;
constexpr u64 SingleStepBit = u64{1} << 57;
constexpr int CoreShift = 48;
constexpr u64 OffsetMask = (u64{1} << CoreShift) - 1;
constexpr u64 CoreMask = u64{0xff} << CoreShift;

// The JIT throws away its whole code cache (128 MiB per core) when it fills up, so the compile
// stops while this much is free, for code the list does not have yet.
constexpr std::size_t KeepFree = std::size_t{32} << 20;
constexpr std::size_t CodeCacheSize = std::size_t{128} << 20;
constexpr std::size_t ChunkSize = 512;
// The app is often closed without shutting the game down, so new blocks go to storage often.
constexpr auto FlushInterval = std::chrono::seconds{10};

struct CoreJob {
    ArmDynarmic64* jit = nullptr;
    std::vector<u64> locations;
    Dynarmic::A64::Jit::CompileAheadResult result;
    double milliseconds = 0;
};

struct State {
    // Recording, guarded by `mutex`.
    std::mutex mutex;
    std::atomic<bool> recording{false};
    const Kernel::KProcess* process = nullptr;
    u64 code_start = 0;
    u64 code_end = 0;
    ::Common::unordered_set<u64> known;
    std::vector<u64> pending;
    std::size_t loaded = 0;
    std::size_t recorded = 0;

    // The list on storage, guarded by `file_mutex`.
    std::mutex file_mutex;
    Common::FS::IOFile file;
    std::size_t written = 0;
    std::jthread flusher;

    // Compiling, guarded by `compile_mutex`.
    std::mutex compile_mutex;
    std::array<CoreJob, Hardware::NUM_CPU_CORES> jobs;
    std::vector<std::jthread> compilers;
    std::atomic<std::size_t> threads{0};
    std::atomic<std::size_t> finished{0};
    std::atomic<std::size_t> done{0};
    std::size_t total = 0;
    std::atomic<bool> cancel{false};
    Clock::time_point started;
};

State& GetState() {
    static State state;
    return state;
}

double MillisecondsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// Hashes the executable pages of the game's code region, as loaded, before the game runs.
u64 HashCode(Kernel::KProcess& process, u64 start, u64 end, std::size_t& executable_bytes) {
    auto& page_table = process.GetPageTable();
    auto& memory = process.GetMemory();
    std::vector<u8> buffer(std::size_t{1} << 20);
    u64 hash = 0;
    u64 address = start;
    while (address < end) {
        Kernel::KMemoryInfo info{};
        Kernel::Svc::PageInfo page_info{};
        if (page_table.QueryInfo(&info, &page_info, address).IsError()) {
            break;
        }
        const u64 block_end = std::min<u64>(static_cast<u64>(info.m_address) + info.m_size, end);
        if (block_end <= address) {
            break;
        }
        if (True(info.m_permission & Kernel::KMemoryPermission::UserExecute)) {
            for (u64 at = address; at < block_end;) {
                const std::size_t size = std::min<u64>(buffer.size(), block_end - at);
                memory.ReadBlock(at, buffer.data(), size);
                hash = Common::CityHash64WithSeeds(reinterpret_cast<const char*>(buffer.data()),
                                                   size, hash, at - start);
                executable_bytes += size;
                at += size;
            }
        }
        address = block_end;
    }
    return hash;
}

/// Writes the blocks recorded since the last call.
void Flush(State& s) {
    std::vector<u64> batch;
    {
        std::scoped_lock lock{s.mutex};
        batch.swap(s.pending);
    }
    if (batch.empty()) {
        return;
    }
    std::scoped_lock lock{s.file_mutex};
    if (!s.file.IsOpen()) {
        return;
    }
    if (s.file.WriteSpan(std::span<const u64>{batch}) != batch.size()) {
        LOG_WARNING(Core_ARM, "[Preforged] could not write the CPU code list");
    }
    void(s.file.Flush());
    s.written += batch.size();
}

void FlushLoop(std::stop_token token) {
    Common::SetCurrentThreadName("CpuCodeList");
    State& s = GetState();
    std::mutex wait_mutex;
    std::condition_variable_any wake;
    while (!token.stop_requested()) {
        {
            std::unique_lock lock{wait_mutex};
            wake.wait_for(lock, token, FlushInterval, [] { return false; });
        }
        Flush(s);
    }
}

/// Reads the list. Returns no records when it is missing, unreadable or made for other code.
/// Clears `intact` when the file ends in part of a record, from the app being closed mid-write.
std::vector<u64> ReadList(const std::filesystem::path& path, const FileHeader& expected,
                          bool& intact) {
    intact = true;
    std::vector<u64> records;
    Common::FS::IOFile in{path, Common::FS::FileAccessMode::Read, Common::FS::FileType::BinaryFile};
    if (!in.IsOpen()) {
        LOG_INFO(Core_ARM, "[Preforged] no CPU code list yet, this session records one");
        return records;
    }
    FileHeader header{};
    if (!in.ReadObject(header) || header.magic != FileMagic || header.version != FileVersion) {
        LOG_INFO(Core_ARM, "[Preforged] CPU code list unreadable, starting a new one");
        return records;
    }
    if (header.code_hash != expected.code_hash || header.code_size != expected.code_size) {
        LOG_INFO(Core_ARM,
                 "[Preforged] the game's code changed since the CPU code list was made (an "
                 "update or a mod), starting a new one");
        return records;
    }
    const u64 size = in.GetSize();
    const u64 body = size > sizeof(header) ? size - sizeof(header) : 0;
    intact = body % sizeof(u64) == 0;
    records.resize(body / sizeof(u64));
    records.resize(in.ReadSpan(std::span<u64>{records}));
    return records;
}

} // Anonymous namespace

void Start(System& system) {
    Stop();
    if (!::Common::Preforged::Options().precompile_cpu_code) {
        LOG_INFO(Core_ARM, "[Preforged] CPU code pre-compile is off");
        return;
    }
    Kernel::KProcess* const process = system.ApplicationProcess();
    if (process == nullptr || !process->Is64Bit()) {
        LOG_INFO(Core_ARM, "[Preforged] CPU code pre-compile only works for 64-bit games");
        return;
    }
    if (dynamic_cast<ArmDynarmic64*>(process->GetArmInterface(0)) == nullptr) {
        LOG_INFO(Core_ARM, "[Preforged] the game runs natively (NCE), no CPU code to compile");
        return;
    }

    State& s = GetState();
    const auto start_time = Clock::now();
    const u64 code_start = GetInteger(process->GetEntryPoint());
    const u64 code_size = process->GetCodeSize();
    std::size_t executable_bytes = 0;
    const FileHeader header{
        .magic = FileMagic,
        .version = FileVersion,
        .code_hash = HashCode(*process, code_start, code_start + code_size, executable_bytes),
        .code_size = code_size,
    };
    const double hash_ms = MillisecondsSince(start_time);

    const auto shader_dir = Common::FS::GetEdenPath(Common::FS::EdenPath::ShaderDir);
    const auto base_dir =
        shader_dir / fmt::format("{:016x}", system.GetApplicationProcessProgramID());
    if (!Common::FS::CreateDir(shader_dir) || !Common::FS::CreateDir(base_dir)) {
        LOG_WARNING(Core_ARM, "[Preforged] could not create {}", base_dir.string());
        return;
    }
    const auto path = base_dir / "cpu_code.bin";
    bool intact = true;
    const std::vector<u64> records = ReadList(path, header, intact);

    std::scoped_lock compile_lock{s.compile_mutex};
    std::array<std::size_t, Hardware::NUM_CPU_CORES> per_core{};
    {
        std::scoped_lock lock{s.mutex};
        s.known.clear();
        s.known.reserve(records.size());
        s.pending.clear();
        s.process = process;
        s.code_start = code_start;
        s.code_end = code_start + code_size;
        for (const u64 record : records) {
            const std::size_t core = (record & CoreMask) >> CoreShift;
            if (core >= Hardware::NUM_CPU_CORES || (record & OffsetMask) >= code_size ||
                !s.known.insert(record).second) {
                continue;
            }
            s.jobs[core].locations.push_back((record & ~PcMask) |
                                             (code_start + (record & OffsetMask)));
        }
        s.loaded = s.known.size();
        s.recorded = 0;
        s.recording.store(true, std::memory_order_release);
    }
    {
        std::scoped_lock lock{s.file_mutex};
        s.written = 0;
        if (records.empty() || !intact) {
            // Starts the file over, keeping the whole records of a list that ends in a part.
            s.file.Open(path, Common::FS::FileAccessMode::Write);
            if (!s.file.IsOpen() || !s.file.WriteObject(header) ||
                s.file.WriteSpan(std::span<const u64>{records}) != records.size()) {
                LOG_WARNING(Core_ARM, "[Preforged] could not write {}", path.string());
            }
            void(s.file.Flush());
        } else {
            s.file.Open(path, Common::FS::FileAccessMode::Append);
        }
    }
    s.flusher = std::jthread([](std::stop_token token) { FlushLoop(token); });

    s.total = 0;
    s.done = 0;
    s.finished = 0;
    s.cancel = false;
    for (std::size_t core = 0; core < Hardware::NUM_CPU_CORES; ++core) {
        CoreJob& job = s.jobs[core];
        job.result = {};
        job.milliseconds = 0;
        per_core[core] = job.locations.size();
        if (job.locations.empty()) {
            continue;
        }
        job.jit = dynamic_cast<ArmDynarmic64*>(process->GetArmInterface(core));
        if (job.jit == nullptr || !job.jit->BeginCompileAhead()) {
            LOG_WARNING(Core_ARM, "[Preforged] CPU core {} already runs, skipping its {} blocks",
                        core, job.locations.size());
            job.jit = nullptr;
            job.locations = {};
            continue;
        }
        s.total += job.locations.size();
    }

    LOG_INFO(Core_ARM,
             "[Preforged] CPU code list: {} blocks from earlier sessions (cores {}, {}, {}, {}); "
             "game code at {:#x}, {} MiB, {} MiB executable, hashed in {:.0f} ms",
             s.loaded, per_core[0], per_core[1], per_core[2], per_core[3], code_start,
             code_size >> 20, executable_bytes >> 20, hash_ms);

    s.started = Clock::now();
    s.threads = 0;
    for (std::size_t core = 0; core < Hardware::NUM_CPU_CORES; ++core) {
        CoreJob& job = s.jobs[core];
        if (job.jit == nullptr) {
            continue;
        }
        ++s.threads;
        s.compilers.emplace_back([&s, &job, core] {
            const std::string name = fmt::format("CpuPrecompile{}", core);
            Common::SetCurrentThreadName(name.c_str());
            const auto begin = Clock::now();
            SCOPE_EXIT {
                job.milliseconds = MillisecondsSince(begin);
                job.jit->EndCompileAhead();
                s.finished.fetch_add(1, std::memory_order_release);
            };
            const std::span<const u64> all{job.locations};
            for (std::size_t at = 0; at < all.size(); at += ChunkSize) {
                if (s.cancel.load(std::memory_order_relaxed)) {
                    break;
                }
                const auto chunk = all.subspan(at, std::min(ChunkSize, all.size() - at));
                const auto result = job.jit->CompileAhead(chunk, KeepFree);
                job.result.compiled += result.compiled;
                job.result.already_compiled += result.already_compiled;
                job.result.code_cache_free = result.code_cache_free;
                s.done.fetch_add(result.compiled + result.already_compiled,
                                 std::memory_order_relaxed);
                if (result.stopped_early) {
                    job.result.stopped_early = true;
                    break;
                }
            }
        });
    }
}

void Wait(const std::function<void(std::size_t done, std::size_t total)>& report) {
    State& s = GetState();
    const auto wait_start = Clock::now();
    bool waited = false;
    while (s.finished.load(std::memory_order_acquire) < s.threads.load()) {
        waited = true;
        if (report) {
            report(s.done.load(std::memory_order_relaxed), s.total);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }
    std::scoped_lock compile_lock{s.compile_mutex};
    if (s.compilers.empty()) {
        return;
    }
    s.compilers.clear(); // Joins them.
    const double total_ms = MillisecondsSince(s.started);
    std::size_t compiled = 0;
    for (std::size_t core = 0; core < Hardware::NUM_CPU_CORES; ++core) {
        CoreJob& job = s.jobs[core];
        if (job.jit != nullptr) {
            const auto& r = job.result;
            compiled += r.compiled;
            LOG_INFO(Core_ARM,
                     "[Preforged]   CPU core {}: compiled {} of {} blocks in {:.0f} ms ({:.1f} us "
                     "each), {} were compiled already, {} MiB of its code cache used{}",
                     core, r.compiled, job.locations.size(), job.milliseconds,
                     r.compiled ? job.milliseconds * 1000.0 / r.compiled : 0.0,
                     r.already_compiled, (CodeCacheSize - r.code_cache_free) >> 20,
                     r.stopped_early ? ", stopped to keep 32 MiB free" : "");
        }
        job.jit = nullptr;
        job.locations = {};
    }
    LOG_INFO(Core_ARM,
             "[Preforged] CPU code pre-compiled: {} blocks in {:.1f} s, loading waited {:.1f} s "
             "for it after the shaders{}",
             compiled, total_ms / 1000.0, waited ? MillisecondsSince(wait_start) / 1000.0 : 0.0,
             s.cancel.load() ? " (cancelled)" : "");
    // Keeps the compile out of the first frame's slow frame report.
    static_cast<void>(::Common::Preforged::FrameStats::Take());
}

void NoteBlock(const Kernel::KProcess* process, std::size_t core, u64 location) {
    State& s = GetState();
    if (!s.recording.load(std::memory_order_relaxed)) {
        return;
    }
    const u64 pc = location & PcMask;
    std::scoped_lock lock{s.mutex};
    if (!s.recording.load(std::memory_order_relaxed) || process != s.process ||
        pc < s.code_start || pc >= s.code_end || (location & SingleStepBit) != 0) {
        return;
    }
    const u64 record = (location & ~PcMask) | (u64{core} << CoreShift) | (pc - s.code_start);
    if (s.known.insert(record).second) {
        s.pending.push_back(record);
        ++s.recorded;
    }
}

void Stop() {
    State& s = GetState();
    s.cancel = true;
    {
        std::scoped_lock compile_lock{s.compile_mutex};
        s.compilers.clear(); // Joins them.
        for (CoreJob& job : s.jobs) {
            job.jit = nullptr;
            job.locations = {};
        }
        s.threads = 0;
        s.finished = 0;
    }
    if (s.flusher.joinable()) {
        s.flusher.request_stop();
        s.flusher.join();
    }
    std::size_t recorded = 0;
    std::size_t known = 0;
    {
        std::scoped_lock lock{s.mutex};
        if (!s.recording.exchange(false)) {
            return;
        }
        recorded = s.recorded;
        known = s.known.size();
        s.process = nullptr;
    }
    Flush(s);
    std::size_t written = 0;
    {
        std::scoped_lock lock{s.file_mutex};
        written = s.written;
        s.file.Close();
    }
    {
        std::scoped_lock lock{s.mutex};
        s.known = {};
        s.pending = {};
    }
    LOG_INFO(Core_ARM,
             "[Preforged] CPU code list: recorded {} new blocks this session ({} written), {} in "
             "the list now",
             recorded, written, known);
}

} // namespace Core::Preforged::CpuCode
