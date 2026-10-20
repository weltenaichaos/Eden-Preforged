// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

/// A pipeline library with the layouts it was made with.
struct PipelineLibrary {
    vk::DescriptorSetLayout set_layout;
    vk::PipelineLayout layout;
    vk::Pipeline pipeline;
};

/// The pipeline libraries the game used this session, and the driver code of every library
/// compiled so far, kept on storage. A pipeline the game needs in play is joined from the
/// libraries of its two halves. A library is compiled only when neither has it; one made again
/// from its stored driver code takes the driver a fraction of a millisecond.
class PipelineLibraries {
public:
    /// Where a library came from.
    enum class Source : u8 {
        Memory,
        Storage,
        Compiled,
        Failed,
    };

    struct Result {
        VkPipeline pipeline{};
        Source source{Source::Failed};
        double ms{};
    };

    /// Makes a library with the driver cache it gets.
    using Make = std::function<PipelineLibrary(VkPipelineCache)>;

    explicit PipelineLibraries(const Device& device);
    ~PipelineLibraries();

    PipelineLibraries(const PipelineLibraries&) = delete;
    PipelineLibraries& operator=(const PipelineLibraries&) = delete;

    /// Opens the store of a game, starting it over when it is from another driver or build.
    void OpenStore(const std::filesystem::path& base_dir);

    /// The library of a half with the SPIR-V `code` and the other state `state`, made with
    /// `make` the first time it is needed this session. Thread safe; a second caller waits for
    /// the first.
    Result Get(LibraryHalf half, u64 code, u64 state, const Make& make);

    /// Whether a library made from `source` (a shader with the state it is translated with)
    /// was compiled and stored on an earlier boot, so it needs no translating.
    [[nodiscard]] bool HasSource(u64 source) const;

    /// Notes `source` with no library, for a shader that can't have one, so later boots skip it
    /// without translating it again.
    void NoteSource(u64 source);

    /// Whether the store has the driver code of a library with this SPIR-V.
    [[nodiscard]] bool Stored(LibraryHalf half, u64 code) const;

    /// Compiles a library with a driver cache of its own and stores its driver code, without
    /// keeping it, for the game file shaders compiled while the game loads. With the code
    /// already stored, only `source` is noted. Returns false when the driver failed.
    bool CompileIntoStore(LibraryHalf half, u64 code, u64 source, const Make& make);

    /// How the libraries asked for in play were found.
    struct Counts {
        size_t memory{};
        size_t storage{};
        size_t compiled{};
        size_t failed{};
        /// Made from stored driver code but slow, so the driver compiled it again.
        size_t slow_storage{};
    };
    [[nodiscard]] Counts PlayCounts() const;

    [[nodiscard]] size_t StoredCount() const;

    [[nodiscard]] static std::string_view SourceName(Source source);

private:
    struct Entry {
        std::mutex mutex;
        std::condition_variable condvar;
        bool done{};
        PipelineLibrary library;
    };

    struct StoreRecord {
        u64 offset{};
        u32 size{};
    };

    [[nodiscard]] static u64 CodeId(LibraryHalf half, u64 code);

    [[nodiscard]] std::optional<std::vector<char>> ReadStored(u64 code_id) const;

    /// Appends the driver code in `cache` under `code_id`.
    void Append(u64 code_id, u64 source, const vk::PipelineCache& cache);

    [[nodiscard]] vk::PipelineCache MakeCache(std::span<const char> data) const;

    const Device& device;

    std::mutex libraries_mutex;
    std::unordered_map<u64, std::shared_ptr<Entry>> libraries;

    mutable std::mutex store_mutex;
    std::filesystem::path store_path;
    std::unordered_map<u64, StoreRecord> store;
    std::unordered_set<u64> sources;
    std::ofstream store_out;

    std::atomic<size_t> count_memory{};
    std::atomic<size_t> count_storage{};
    std::atomic<size_t> count_compiled{};
    std::atomic<size_t> count_failed{};
    std::atomic<size_t> count_slow_storage{};
};

} // namespace Vulkan
