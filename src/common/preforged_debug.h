// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>

namespace Common::Preforged {

/// Switches for testing Eden Preforged without a new build, set in the debug settings.
struct DebugOptions {
    /// Bakes the values of constant buffer 1 into shaders, as upstream Eden does. Off makes
    /// shaders built from the game files match the game's own.
    bool bake_driver_constants = false;
    /// Searches the game files for shaders once per game.
    bool scan_game_files = true;
    /// Searches the game files again on every start while on.
    bool rescan_game_files = false;
    /// Compiles the shader pairs found in the game files while the game loads.
    bool precompile_game_shaders = true;
    /// Logs every shader compile during play, with its timing and where its shaders came from.
    bool log_shader_compiles = true;
    /// Share of the game file shader pairs to compile, in percent, most likely first. Compiled
    /// pairs are kept on storage and read when the game draws with them.
    unsigned precompile_percent = 100;
    /// Hands the driver a pair's stored code even when the game draws it with another state
    /// than it was compiled with, in case the driver can reuse part of it.
    bool store_any_state = false;
    /// Logs memory use every 30 seconds in game.
    bool log_memory = true;
    /// Logs every frame that took more than twice as long as the frame before it, with what ran
    /// during it (see FrameStats).
    bool log_slow_frames = true;
    /// Logs which allocation sizes fill the malloc heap and which mappings the app holds, each
    /// time the heap grows past another 2 GiB. Walking the heap holds up the emulator a moment.
    bool log_allocation_sizes = true;
    /// Records the CPU code the game compiles to a list, and compiles that list while the game
    /// loads the next time (Core::Preforged::CpuCode).
    bool precompile_cpu_code = true;
    /// Starts running the game only after the shader file scan, the shader cache and the CPU
    /// code have loaded, instead of while they load.
    bool start_after_loading = true;
    /// Records a sample of the heap allocations with the code that made them, and logs which
    /// code holds the most memory with every memory log. Costs a little speed.
    bool track_allocations = true;
};

[[nodiscard]] DebugOptions& Options();

/// Takes the switches from the settings and logs them. Call when a game starts.
void LoadOptions();

/// Logs how many memory mappings the app holds and which files or heaps hold the most, since
/// Android ends the app once it has about 65,000. Shows the top `top` names.
void LogMappingSummary(size_t top);

/// Counts the GPU memory allocations (VkDeviceMemory) the memory allocator makes and frees, so
/// the memory log shows how many the emulator holds. Some drivers map each one on its own, and
/// Android ends the app once it holds about 65,000 mappings.
void NoteGpuAllocation(std::uint32_t memory_type, std::uint64_t size);
void NoteGpuFree(std::uint32_t memory_type, std::uint64_t size);

/// Vulkan objects the emulator creates, counted so the memory log shows which ones pile up.
/// Drivers give some of them GPU memory of their own (command buffers, query pools, descriptor
/// pools, events), which the GPU allocation counts above don't see.
enum class VulkanObject : std::uint8_t {
    CommandPool,
    CommandBuffer,
    DescriptorPool,
    DescriptorSet,
    QueryPool,
    Event,
    Fence,
    Semaphore,
    DeviceMemory,
    Buffer,
    BufferView,
    Image,
    ImageView,
    Framebuffer,
    RenderPass,
    Sampler,
    Pipeline,
    ShaderModule,
    Count,
};

/// Adds `change` to the number of live objects of `type`.
void NoteVulkanObjects(VulkanObject type, std::int64_t change);

/// Counts a queue submit. `ahead` is how many earlier submits the GPU still had to finish.
void NoteQueueSubmit(std::uint64_t ahead);

/// Notes how many command chunks wait for the thread that records and submits them.
void NoteWorkQueue(std::uint64_t waiting);

/// Logs resident, virtual and malloc memory and the GPU allocations, tagged with `when`, and the
/// number of mappings when `count_mappings` is set. Past 20,000 mappings it also logs which
/// names hold them, again after each 5,000 more.
void LogMemoryUse(const char* when, bool count_mappings = true);

/// Logs the allocation sizes that hold the most malloc memory, tagged with `when`. Walks the
/// whole heap, which holds up every thread that allocates while it runs.
void LogAllocationSizes(const char* when);

} // namespace Common::Preforged
