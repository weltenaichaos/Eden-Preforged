// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <cstring>

#include "common/cityhash.h"
#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {
namespace {
constexpr std::array<char, 4> STORE_MAGIC{'P', 'F', 'L', 'S'};
constexpr u32 STORE_VERSION = 1;

/// A library made from stored driver code slower than this was compiled by the driver again.
constexpr double STORED_LIBRARY_MS = 3.0;

struct StoreHeader {
    std::array<char, 4> magic;
    u32 version;
    u32 driver_id;
    u32 driver_version;
};

struct StoreRecordHeader {
    u64 code_id;
    /// The shader and state the library was translated from, 0 when it was made in play.
    u64 source;
    u32 size;
    u32 reserved;
};
} // Anonymous namespace

PipelineLibraries::PipelineLibraries(const Device& device_) : device{device_} {}

PipelineLibraries::~PipelineLibraries() = default;

std::string_view PipelineLibraries::SourceName(Source source) {
    switch (source) {
    case Source::Memory:
        return "from memory";
    case Source::Storage:
        return "from storage";
    case Source::Compiled:
        return "compiled";
    case Source::Failed:
        return "failed";
    }
    return "?";
}

u64 PipelineLibraries::CodeId(LibraryHalf half, u64 code) {
    const std::array<u64, 2> words{static_cast<u64>(half), code};
    return Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words));
}

void PipelineLibraries::OpenStore(const std::filesystem::path& base_dir) {
    std::scoped_lock lock{store_mutex};
    store.clear();
    sources.clear();
    store_out.close();
    store_path = base_dir / "preforged_library_store.bin";
    const StoreHeader expected{
        .magic = STORE_MAGIC,
        .version = STORE_VERSION,
        .driver_id = static_cast<u32>(device.GetDriverID()),
        .driver_version = device.GetDriverVersion(),
    };
    std::error_code ec;
    u64 valid_end{0};
    u64 bytes{0};
    {
        const u64 file_size{std::filesystem::file_size(store_path, ec)};
        std::ifstream file{store_path, std::ios::binary};
        StoreHeader header{};
        if (!ec && file && file.read(reinterpret_cast<char*>(&header), sizeof(header)) &&
            std::memcmp(&header, &expected, sizeof(header)) == 0) {
            valid_end = sizeof(header);
            StoreRecordHeader record{};
            while (valid_end + sizeof(record) <= file_size &&
                   file.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                const u64 offset{valid_end + sizeof(record)};
                if (offset + record.size > file_size) {
                    break;
                }
                // Records without code only note a source.
                if (record.size != 0) {
                    store.insert_or_assign(record.code_id, StoreRecord{offset, record.size});
                }
                if (record.source != 0) {
                    sources.insert(record.source);
                }
                valid_end = offset + record.size;
                bytes += record.size;
                file.seekg(static_cast<std::streamoff>(valid_end));
            }
        } else if (!ec && file) {
            LOG_INFO(Render_Vulkan,
                     "[Preforged] library store is from another driver or build, starting over");
        }
    }
    if (valid_end == 0) {
        std::ofstream file{store_path, std::ios::binary | std::ios::trunc};
        file.write(reinterpret_cast<const char*>(&expected), sizeof(expected));
        valid_end = sizeof(expected);
    } else if (std::filesystem::file_size(store_path, ec) != valid_end) {
        // A write that was cut off, for example by a crash, is dropped.
        std::filesystem::resize_file(store_path, valid_end, ec);
    }
    store_out.open(store_path, std::ios::binary | std::ios::app);
    LOG_INFO(Render_Vulkan, "[Preforged] library store on storage: {} libraries, {} MiB",
             store.size(), bytes >> 20);
}

std::optional<std::vector<char>> PipelineLibraries::ReadStored(u64 code_id) const {
    StoreRecord record;
    std::filesystem::path path;
    {
        std::scoped_lock lock{store_mutex};
        const auto it{store.find(code_id)};
        if (it == store.end()) {
            return std::nullopt;
        }
        record = it->second;
        path = store_path;
    }
    std::vector<char> data(record.size);
    std::ifstream file{path, std::ios::binary};
    if (!file.seekg(static_cast<std::streamoff>(record.offset)) ||
        !file.read(data.data(), record.size)) {
        return std::nullopt;
    }
    return data;
}

void PipelineLibraries::Append(u64 code_id, u64 source, const vk::PipelineCache& cache) {
    size_t size{};
    if (cache.Read(&size, nullptr) != VK_SUCCESS || size == 0) {
        return;
    }
    std::vector<char> data(size);
    if (cache.Read(&size, data.data()) != VK_SUCCESS) {
        return;
    }
    std::scoped_lock lock{store_mutex};
    if (!store_out.is_open() || store.contains(code_id)) {
        return;
    }
    const StoreRecordHeader record{code_id, source, static_cast<u32>(size), 0};
    const u64 offset{static_cast<u64>(store_out.tellp()) + sizeof(record)};
    store_out.write(reinterpret_cast<const char*>(&record), sizeof(record));
    store_out.write(data.data(), static_cast<std::streamsize>(size));
    // Flushed at once, since a library made again in play reads its code back from the file.
    store_out.flush();
    if (store_out) {
        store.insert_or_assign(code_id, StoreRecord{offset, static_cast<u32>(size)});
        if (source != 0) {
            sources.insert(source);
        }
    }
}

vk::PipelineCache PipelineLibraries::MakeCache(std::span<const char> data) const {
    return device.GetLogical().CreatePipelineCache({
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .initialDataSize = data.size(),
        .pInitialData = data.empty() ? nullptr : data.data(),
    });
}

PipelineLibraries::Result PipelineLibraries::Get(LibraryHalf half, u64 code, u64 state,
                                                 const Make& make) {
    const std::array<u64, 3> words{static_cast<u64>(half), code, state};
    const u64 id{Common::CityHash64(reinterpret_cast<const char*>(words.data()), sizeof(words))};
    std::shared_ptr<Entry> entry;
    bool is_new{};
    {
        std::scoped_lock lock{libraries_mutex};
        auto [it, inserted] = libraries.try_emplace(id);
        if (inserted) {
            it->second = std::make_shared<Entry>();
        }
        entry = it->second;
        is_new = inserted;
    }
    if (!is_new) {
        const auto wait_start{std::chrono::steady_clock::now()};
        std::unique_lock lock{entry->mutex};
        entry->condvar.wait(lock, [&entry] { return entry->done; });
        const VkPipeline pipeline{*entry->library.pipeline};
        const std::chrono::duration<double, std::milli> waited{std::chrono::steady_clock::now() -
                                                               wait_start};
        ++(pipeline ? count_memory : count_failed);
        return Result{pipeline, pipeline ? Source::Memory : Source::Failed, waited.count()};
    }
    const auto start{std::chrono::steady_clock::now()};
    const u64 code_id{CodeId(half, code)};
    PipelineLibrary library;
    Source source{Source::Failed};
    try {
        if (const auto data{ReadStored(code_id)}) {
            const vk::PipelineCache cache{MakeCache(*data)};
            library = make(*cache);
            source = Source::Storage;
        } else {
            const vk::PipelineCache cache{MakeCache({})};
            library = make(*cache);
            source = Source::Compiled;
            Append(code_id, 0, cache);
        }
    } catch (const vk::Exception& exception) {
        LOG_ERROR(Render_Vulkan, "[Preforged] pipeline library {:016x} failed: {}", code,
                  exception.what());
        library = {};
        source = Source::Failed;
    }
    const std::chrono::duration<double, std::milli> time{std::chrono::steady_clock::now() - start};
    const VkPipeline pipeline{*library.pipeline};
    {
        std::scoped_lock lock{entry->mutex};
        entry->library = std::move(library);
        entry->done = true;
    }
    entry->condvar.notify_all();
    switch (source) {
    case Source::Storage:
        ++count_storage;
        if (time.count() > STORED_LIBRARY_MS) {
            ++count_slow_storage;
        }
        break;
    case Source::Compiled:
        ++count_compiled;
        break;
    default:
        ++count_failed;
        break;
    }
    return Result{pipeline, source, time.count()};
}

void PipelineLibraries::NoteSource(u64 source) {
    std::scoped_lock lock{store_mutex};
    if (source == 0 || !store_out.is_open() || sources.contains(source)) {
        return;
    }
    const StoreRecordHeader record{0, source, 0, 0};
    store_out.write(reinterpret_cast<const char*>(&record), sizeof(record));
    store_out.flush();
    sources.insert(source);
}

bool PipelineLibraries::HasSource(u64 source) const {
    std::scoped_lock lock{store_mutex};
    return sources.contains(source);
}

bool PipelineLibraries::Stored(LibraryHalf half, u64 code) const {
    std::scoped_lock lock{store_mutex};
    return store.contains(CodeId(half, code));
}

bool PipelineLibraries::CompileIntoStore(LibraryHalf half, u64 code, u64 source, const Make& make) {
    const u64 code_id{CodeId(half, code)};
    {
        std::scoped_lock lock{store_mutex};
        if (store.contains(code_id)) {
            // Another shader or state gave the same SPIR-V. Noted so the next boot skips it
            // without translating it again.
            if (source != 0 && store_out.is_open() && !sources.contains(source)) {
                const StoreRecordHeader record{code_id, source, 0, 0};
                store_out.write(reinterpret_cast<const char*>(&record), sizeof(record));
                store_out.flush();
                sources.insert(source);
            }
            return true;
        }
    }
    try {
        const vk::PipelineCache cache{MakeCache({})};
        const PipelineLibrary library{make(*cache)};
        if (!library.pipeline) {
            return false;
        }
        Append(code_id, source, cache);
    } catch (const vk::Exception& exception) {
        LOG_DEBUG(Render_Vulkan, "[Preforged] pipeline library {:016x} failed: {}", code,
                  exception.what());
        return false;
    }
    return true;
}

PipelineLibraries::Counts PipelineLibraries::PlayCounts() const {
    return Counts{
        .memory = count_memory.load(),
        .storage = count_storage.load(),
        .compiled = count_compiled.load(),
        .failed = count_failed.load(),
        .slow_storage = count_slow_storage.load(),
    };
}

size_t PipelineLibraries::StoredCount() const {
    std::scoped_lock lock{store_mutex};
    return store.size();
}

} // namespace Vulkan
