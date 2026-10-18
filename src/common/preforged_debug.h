// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace Common::Preforged {

/// Switches for testing Eden Preforged without a new build, set in the debug settings.
struct DebugOptions {
    /// Bakes the values of constant buffer 1 into shaders, as upstream Eden does. Off makes
    /// shaders built from the game files match the game's own.
    bool bake_driver_constants = false;
    /// Searches the game files for shaders once per game, and again when they change.
    bool scan_game_files = true;
    /// Searches the game files again on every start while on.
    bool rescan_game_files = false;
    /// Unpacks the compressed parts of Unreal Engine packages during the game file search.
    bool unpack_unreal_packages = true;
    /// Unpacks the Oodle blocks of those packages too (Kraken, Mermaid, Selkie, Leviathan).
    bool unpack_oodle = true;
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
    /// Compiles each vertex shader from the game files with its own attribute types: the types
    /// the game drew it with, or for a shader the game hasn't drawn, float or an integer type by
    /// what its code does with each attribute. Off gives every shader the most common types.
    bool input_types_per_shader = true;
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
    /// Waits for the GPU before a queue submit once it has more than MaxGpuLag submits left to
    /// finish. Each of those holds command buffers and queries, which the driver keeps in memory
    /// mappings of its own, and the emulator never frees them.
    bool limit_gpu_lag = true;
    /// While the GPU still has earlier work to finish, adds the GPU work of a finished command
    /// list or draw batch to the next submit instead of submitting it on its own.
    bool merge_gpu_work = true;
};

/// How many queue submits the GPU may have left to finish before the next one waits for it.
constexpr std::uint64_t MaxGpuLag = 32;

/// How many queue submits the GPU must have left to finish before command list and draw batch
/// work is merged into the next submit.
constexpr std::uint64_t MergeGpuLag = 3;

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

/// What made the emulator submit GPU work, counted in the memory log.
enum class SubmitCause : std::uint8_t {
    Other,
    Fence,       ///< A fence the game or a download waits for
    CommandList, ///< A finished command list from the game
    DrawBatch,   ///< Many draws recorded since the last submit
    Wait,        ///< Waiting for the GPU (downloads, reads back, cleanups)
    Frame,       ///< A finished frame
    Count,
};

/// Marks the queue submits made on this thread while it lives with `cause`.
class SubmitCauseScope {
public:
    explicit SubmitCauseScope(SubmitCause cause);
    ~SubmitCauseScope();

    SubmitCauseScope(const SubmitCauseScope&) = delete;
    SubmitCauseScope& operator=(const SubmitCauseScope&) = delete;

private:
    SubmitCause previous;
};

/// Counts a submit the GPU thread handed to the command worker, under the current cause.
/// `pending` is how many earlier submits the GPU had left to finish.
void NoteSubmit(std::uint64_t tick, std::uint64_t pending);

/// Counts command list or draw batch work merged into a later submit.
void NoteMergedWork();

/// Counts a submit left out because nothing was recorded since the last one.
void NoteEmptySubmitSkipped();

/// Counts a wait for the GPU to catch up before a submit.
void NoteGpuLagWait(std::chrono::steady_clock::duration time);

/// The GPU's progress, for the freeze report: the last finished tick the GPU thread knew of.
void NoteGpuTick(std::uint64_t known_tick);

/// Notes that the game showed a frame, for the freeze report.
void NoteFrame();

/// Counts the time without a frame from now, when a game starts.
void ResetFrameClock();

/// Seconds since the game last showed a frame, or a negative value before the first one.
[[nodiscard]] double SecondsSinceFrame();

/// Describes the GPU's progress for the freeze report: submits, how many are left, and what is
/// waiting right now.
[[nodiscard]] std::string DescribeGpuProgress();

/// Logs resident, virtual and malloc memory and the GPU allocations, tagged with `when`, and the
/// number of mappings when `count_mappings` is set. Past 20,000 mappings it also logs which
/// names hold them, again after each 5,000 more.
void LogMemoryUse(const char* when, bool count_mappings = true);

/// Logs the allocation sizes that hold the most malloc memory, tagged with `when`. Walks the
/// whole heap, which holds up every thread that allocates while it runs.
void LogAllocationSizes(const char* when);

} // namespace Common::Preforged
