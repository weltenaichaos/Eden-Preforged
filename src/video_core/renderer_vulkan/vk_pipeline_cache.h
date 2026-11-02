// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include "common/container/unordered_map.h"
#include <vector>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/preforged_state_variants.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_pipeline_library.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_cache.h"

namespace Core {
class System;
}

namespace Shader::IR {
struct Program;
}

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

using Maxwell = Tegra::Engines::Maxwell3D::Regs;

struct ComputePipelineCacheKey {
    u64 unique_hash;
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;

    size_t Hash() const noexcept;

    bool operator==(const ComputePipelineCacheKey& rhs) const noexcept;

    bool operator!=(const ComputePipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }
};
static_assert(std::has_unique_object_representations_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<ComputePipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<ComputePipelineCacheKey>);

} // namespace Vulkan

namespace std {

template <>
struct hash<Vulkan::ComputePipelineCacheKey> {
    size_t operator()(const Vulkan::ComputePipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};

} // namespace std

namespace Vulkan {

class ComputePipeline;
class DescriptorPool;
class Device;
class PipelineStatistics;
class RenderPassCache;
class Scheduler;

using VideoCommon::ShaderInfo;

struct ShaderPools {
    void ReleaseContents() {
        flow_block.ReleaseContents();
        block.ReleaseContents();
        inst.ReleaseContents();
    }

    Shader::ObjectPool<Shader::IR::Inst> inst{8192};
    Shader::ObjectPool<Shader::IR::Block> block{32};
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_block{32};
};

class PipelineCache : public VideoCommon::ShaderCache {
public:
    explicit PipelineCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, const Device& device,
                           Scheduler& scheduler, DescriptorPool& descriptor_pool,
                           GuestDescriptorQueue& guest_descriptor_queue,
                           DescriptorBufferRing& descriptor_buffer_ring,
                           RenderPassCache& render_pass_cache, BufferCache& buffer_cache,
                           TextureCache& texture_cache, VideoCore::ShaderNotify& shader_notify_);
    ~PipelineCache();

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipeline();

    [[nodiscard]] ComputePipeline* CurrentComputePipeline();

    void LoadDiskResources(u64 title_id, std::stop_token stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback);

    struct CachedShaderCode {
        u64 unique_hash;
        Shader::Stage stage;
        std::vector<u64> code;
        std::vector<Shader::TextureType> texture_types; ///< Types the game used at runtime
        size_t cbuf_value_count;                        ///< Constant buffer reads it needed
        std::shared_ptr<VideoCommon::FileEnvironment> env; ///< Everything the game used at runtime
    };

    /// Reads the guest code of every distinct shader stored in a pipeline cache file.
    /// Pass a private copy of the file: a damaged file is deleted while it is read.
    [[nodiscard]] static std::vector<CachedShaderCode> ReadCachedShaderCode(
        const std::filesystem::path& filename, std::stop_token stop_loading,
        std::vector<GraphicsPipelineCacheKey>* graphics_keys = nullptr);

    /// Hashes the parts of a pipeline's state that change the translated shaders.
    [[nodiscard]] static u64 ShaderStateHash(const GraphicsPipelineCacheKey& key);

private:
    /// Compiles the shader pairs found in the game files that no cached pipeline uses yet, so the
    /// driver keeps their compiled code. Pipelines with other states reuse that code.
    /// Shader pairs from the game files, with the shader state they were compiled with.
    std::unordered_map<u64, u64> preforged_pairs;
    /// Shaders from the game files in a pair, and the others from the game files.
    std::unordered_set<u64> preforged_shaders;
    std::unordered_set<u64> preforged_other_shaders;
    /// Shader pairs from the game files left out by the pre-compile share.
    std::unordered_set<u64> preforged_skipped_pairs;

public:
    /// The parts of a pipeline's state that change the SPIR-V of a vertex and fragment pair.
    struct PreforgedShaderInputs {
        /// Type of each attribute the vertex shader reads, 0xFF for attributes it doesn't read.
        std::array<u8, 32> attributes;
        u32 point_size;
        u32 alpha_test_ref;
        u8 points;
        u8 xfb;
        u8 ndc_minus_one_to_one;
        u8 alpha_test_func;
        u8 dual_source_blend;
        u8 early_z;
        u8 y_negate;
        u8 reserved;

        bool operator==(const PreforgedShaderInputs&) const = default;
    };
    /// What a vertex and fragment pair was translated to.
    struct PreforgedCode {
        u64 code_hash{};
        PreforgedShaderInputs inputs{};
    };

private:
    /// Where a compiled pair's driver code sits in the store file on storage.
    struct PreforgedStoreEntry {
        u64 offset;
        u64 state_hash;
        u32 size;
        PreforgedCode code;
        /// Attributes whose type was read from the vertex shader's code, one bit each.
        u32 code_typed{};
    };
    /// Driver code of the pairs compiled from the game files, one small driver cache per pair,
    /// kept on storage and read only when the game draws with that pair. Only the index is kept
    /// in memory.
    std::filesystem::path preforged_store_path;
    std::mutex preforged_store_mutex;
    std::unordered_map<u64, PreforgedStoreEntry> preforged_store;
    std::ofstream preforged_store_out;
    std::atomic<size_t> preforged_store_reads{};

    /// Reads the store index, starts the store over for another driver, and clears the driver
    /// pipeline cache once if earlier builds compiled the game file pairs into it.
    void OpenPreforgedStore(const std::filesystem::path& base_dir);

    /// Appends the code in `cache` for a pair. Called while compiling, from worker threads.
    /// `code_typed` has a bit for each attribute whose type was read from the shader's code.
    void AppendPreforgedStore(u64 pair_id, u64 state_hash, const PreforgedCode& code,
                              u32 code_typed, const vk::PipelineCache& cache);

    /// How often the game feeds each dynamic attribute type (off, float, int, uint) at each
    /// attribute when a played vertex shader reads it. Counted while the played pipelines build.
    std::array<std::array<size_t, 4>, 32> preforged_attribute_votes{};

    /// Counts the attribute types a played pipeline's vertex shader reads. Called under a lock.
    void CountPreforgedAttributeTypes(const GraphicsPipelineCacheKey& key,
                                      const PreforgedCode& code);

    /// Reads a pair's driver code from storage when the key draws with a stored pair that was
    /// translated to the same SPIR-V. `match` tells how the stored pair compares, for the log.
    [[nodiscard]] std::shared_ptr<vk::PipelineCache> LoadPreforgedStoreEntry(
        const GraphicsPipelineCacheKey& key, const PreforgedCode& code, std::string& match);

    void PrecompilePreforgedPairs(const std::filesystem::path& base_dir,
                                  std::span<const GraphicsPipelineCacheKey> played_keys,
                                  std::stop_token stop_loading,
                                  const VideoCore::DiskResourceLoadCallback& callback);

    [[nodiscard]] GraphicsPipeline* CurrentGraphicsPipelineSlowPath();

    [[nodiscard]] GraphicsPipeline* BuiltPipeline(GraphicsPipeline* pipeline) const noexcept;

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline();

    std::unique_ptr<GraphicsPipeline> CreateGraphicsPipeline(
        ShaderPools& pools, const GraphicsPipelineCacheKey& key,
        std::span<Shader::Environment* const> envs, PipelineStatistics* statistics,
        bool build_in_parallel, std::shared_ptr<vk::PipelineCache> build_cache = {},
        PreforgedCode* code_out = nullptr, bool store_libraries = false);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(const ComputePipelineCacheKey& key,
                                                           const ShaderInfo* shader);

    std::unique_ptr<ComputePipeline> CreateComputePipeline(ShaderPools& pools,
                                                           const ComputePipelineCacheKey& key,
                                                           Shader::Environment& env,
                                                           PipelineStatistics* statistics,
                                                           bool build_in_parallel);

    void SerializeVulkanPipelineCache(const std::filesystem::path& filename,
                                      const vk::PipelineCache& pipeline_cache, u32 cache_version);

    vk::PipelineCache LoadVulkanPipelineCache(const std::filesystem::path& filename,
                                              u32 expected_cache_version);

    void QueueVulkanPipelineCacheFlush();

    /// Logs, for a pipeline compiled during play, whether its shaders had a pipeline already
    /// and which parts of the draw state set the two apart.
    void LogStateVariant(const GraphicsPipelineCacheKey& key);

    /// How the pipelines that went to be joined were joined, and how their libraries were found.
    [[nodiscard]] std::string LibraryJoinSummary();

    const Device& device;
    Scheduler& scheduler;
    DescriptorPool& descriptor_pool;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    RenderPassCache& render_pass_cache;
    BufferCache& buffer_cache;
    TextureCache& texture_cache;
    VideoCore::ShaderNotify& shader_notify;
    bool use_asynchronous_shaders{};
    bool use_vulkan_pipeline_cache{};

    GraphicsPipelineCacheKey graphics_key{};
    GraphicsPipeline* current_pipeline{};

    ::Common::unordered_map<ComputePipelineCacheKey, std::unique_ptr<ComputePipeline>> compute_cache;
    /// Pipeline libraries of single shaders, which pipelines the game needs in play are joined
    /// from. Declared before the pipelines, so it goes after them.
    PipelineLibraries pipeline_libraries;
    bool use_pipeline_libraries{};
    /// Test: every pipeline the game drew and every pipeline built in play is joined.
    bool join_every_pipeline{};
    /// Pipelines the game needed in play that were joined from libraries, that didn't need
    /// them because the driver had their code, and that couldn't be joined, by reason.
    std::atomic<size_t> library_joins{};
    std::atomic<size_t> library_not_needed{};
    std::array<std::atomic<size_t>, 5> library_fallbacks{};
    /// The SPIR-V of every pipeline compiled this session, in the driver pipeline cache. A
    /// pipeline with the same SPIR-V and another state gets the shaders from there and needs no
    /// libraries.
    std::mutex built_codes_mutex;
    std::unordered_set<u64> built_codes;

    ::Common::unordered_map<GraphicsPipelineCacheKey, std::unique_ptr<GraphicsPipeline>> graphics_cache;

    ShaderPools main_pools;

    Shader::Profile profile;
    Shader::HostTranslateInfo host_info;

    std::filesystem::path pipeline_cache_filename;

    std::filesystem::path vulkan_pipeline_cache_filename;
    vk::PipelineCache vulkan_pipeline_cache;
    size_t pipelines_since_flush{};
    std::chrono::steady_clock::time_point last_flush{};
    std::atomic<size_t> last_cache_size{};
    std::atomic_bool flush_in_flight{};

    Common::ThreadWorker workers;
    /// Compiles the full pipeline of each joined pipeline, one at a time in the background.
    /// Declared after the pipelines, so it stops before they go.
    Common::ThreadWorker full_worker;
    Common::ThreadWorker serialization_thread;
    DynamicFeatures dynamic_features;

    PreforgedState::Tally state_tally;
    size_t state_examples{};
    std::chrono::steady_clock::time_point last_state_summary{};
    std::chrono::steady_clock::duration state_search_time{};
};

} // namespace Vulkan
