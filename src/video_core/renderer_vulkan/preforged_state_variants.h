// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <string>
#include <utility>

#include <fmt/format.h>

#include "common/common_types.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"

// Tells, for a pipeline compiled during play whose shaders already have a pipeline, which parts
// of the draw state made it a new pipeline, and which of Eden's dynamic state settings would
// have let the two share one. Only for the log.
namespace Vulkan::PreforgedState {

/// What takes a part of the state out of the pipeline, so pipelines that differ only there
/// share one.
enum class Merge : u8 {
    Eds1,
    Eds1OrVertexInput,
    Eds2,
    VertexInput,
    Eds3,
    Never,
};

enum Part : u8 {
    Topology,
    TopologyClass,
    PolygonMode,
    Tessellation,
    Msaa,
    DualSourceBlend,
    TransformFeedback,
    DepthRange,
    AlphaTest,
    EarlyZ,
    DepthBuffer,
    YNegate,
    ProvokingVertex,
    ConservativeRaster,
    SmoothLines,
    AlphaToCoverage,
    EngineHint,
    ColorFormats,
    AlphaTestReference,
    PointSize,
    ViewportSwizzles,
    InstanceDivisors,
    VertexAttributes,
    VertexAttributeTypes,
    CullMode,
    FrontFace,
    Stencil,
    DepthTest,
    DepthWrite,
    DepthBoundsTest,
    VertexStrides,
    PrimitiveRestart,
    DepthBias,
    RasterizerDiscard,
    LogicOp,
    LogicOpEnable,
    DepthClamp,
    LineStipple,
    Blend,
    ColorMasks,
    Features,
    PART_COUNT,
};

struct PartInfo {
    const char* name;
    Merge merge;
};

inline constexpr std::array<PartInfo, PART_COUNT> PARTS{{
    {"topology in one class", Merge::Eds2},
    {"topology class", Merge::Never},
    {"polygon mode", Merge::Never},
    {"tessellation", Merge::Never},
    {"multisampling", Merge::Never},
    {"dual source blend", Merge::Never},
    {"transform feedback", Merge::Never},
    {"depth range mode", Merge::Never},
    {"alpha test function", Merge::Never},
    {"early depth test", Merge::Never},
    {"depth buffer format", Merge::Never},
    {"y flip", Merge::Never},
    {"provoking vertex", Merge::Never},
    {"conservative raster", Merge::Never},
    {"smooth lines", Merge::Never},
    {"alpha to coverage", Merge::Never},
    {"engine hint", Merge::Never},
    {"color formats", Merge::Never},
    {"alpha test reference", Merge::Never},
    {"point size", Merge::Never},
    {"viewport swizzles", Merge::Never},
    {"instancing divisors", Merge::VertexInput},
    {"vertex attribute formats", Merge::VertexInput},
    {"vertex attribute types", Merge::Never},
    {"cull mode", Merge::Eds1},
    {"front face", Merge::Eds1},
    {"stencil", Merge::Eds1},
    {"depth test", Merge::Eds1},
    {"depth write", Merge::Eds1},
    {"depth bounds test", Merge::Eds1},
    {"vertex strides", Merge::Eds1OrVertexInput},
    {"primitive restart", Merge::Eds2},
    {"depth bias", Merge::Eds2},
    {"rasterizer discard", Merge::Eds2},
    {"logic op", Merge::Eds2},
    {"logic op enable", Merge::Eds3},
    {"depth clamp", Merge::Eds3},
    {"line stipple", Merge::Eds3},
    {"blending", Merge::Eds3},
    {"color write masks", Merge::Eds3},
    {"dynamic state flags or driver", Merge::Never},
}};

/// Ways two states can come to share one pipeline, from the fewest settings on.
enum Way : u8 {
    WayEds1,
    WayEds2,
    WayVertexInput,
    WayEdsAndVertexInput,
    WayEds3,
    WayNever,
    WAY_COUNT,
};

inline constexpr std::array<const char*, WAY_COUNT> WAY_NAMES{
    "Extended Dynamic State 1",   "Extended Dynamic State 2",
    "Vertex Input Dynamic State", "Extended Dynamic State 1 or 2 with Vertex Input Dynamic State",
    "Extended Dynamic State 3",   "no setting",
};

/// The topology Eden keeps in the pipeline when Extended Dynamic State 1 and 2 are on.
inline Maxwell::PrimitiveTopology ClassOf(Maxwell::PrimitiveTopology topology) {
    using T = Maxwell::PrimitiveTopology;
    switch (topology) {
    case T::TriangleStrip:
    case T::TriangleFan:
    case T::Quads:
    case T::QuadStrip:
    case T::Polygon:
        return T::Triangles;
    case T::LineStripAdjacency:
        return T::LinesAdjacency;
    case T::TriangleStripAdjacency:
        return T::TrianglesAdjacency;
    default:
        return topology;
    }
}

/// The type class of a vertex attribute that stays in the pipeline with Vertex Input Dynamic
/// State: float (normalized and scaled too), signed or unsigned integer.
inline u32 AttributeTypeClass(const FixedPipelineState::VertexAttribute& attribute) {
    if (attribute.enabled == 0) {
        return 0;
    }
    switch (attribute.type.Value()) {
    case 3:
        return 2;
    case 4:
        return 3;
    case 0:
        return 0;
    default:
        return 1;
    }
}

template <typename T>
bool Same(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// The parts in which two states differ, one bit per Part.
inline u64 DifferingParts(const FixedPipelineState& a, const FixedPipelineState& b) {
    u64 parts = 0;
    const auto mark = [&parts](Part part, bool differs) {
        if (differs) {
            parts |= u64{1} << part;
        }
    };
    if (a.topology.Value() != b.topology.Value()) {
        mark(ClassOf(a.topology.Value()) == ClassOf(b.topology.Value()) ? Topology : TopologyClass,
             true);
    }
    mark(PolygonMode, a.polygon_mode != b.polygon_mode);
    mark(Tessellation, a.tessellation_primitive != b.tessellation_primitive ||
                           a.tessellation_spacing != b.tessellation_spacing ||
                           a.tessellation_clockwise != b.tessellation_clockwise ||
                           a.patch_control_points_minus_one != b.patch_control_points_minus_one);
    mark(Msaa, a.msaa_mode.Value() != b.msaa_mode.Value());
    mark(DualSourceBlend, a.attachment0_dual_source_blend != b.attachment0_dual_source_blend);
    mark(TransformFeedback,
         a.xfb_enabled != b.xfb_enabled || (a.xfb_enabled != 0 && !Same(a.xfb_state, b.xfb_state)));
    mark(DepthRange, a.ndc_minus_one_to_one != b.ndc_minus_one_to_one);
    mark(AlphaTest, a.alpha_test_func != b.alpha_test_func);
    mark(EarlyZ, a.early_z != b.early_z);
    mark(DepthBuffer, a.depth_enabled != b.depth_enabled || a.depth_format != b.depth_format);
    mark(YNegate, a.y_negate != b.y_negate);
    mark(ProvokingVertex, a.provoking_vertex_last != b.provoking_vertex_last);
    mark(ConservativeRaster, a.conservative_raster_enable != b.conservative_raster_enable);
    mark(SmoothLines, a.smooth_lines != b.smooth_lines);
    mark(AlphaToCoverage, a.alpha_to_coverage_enabled != b.alpha_to_coverage_enabled ||
                              a.alpha_to_one_enabled != b.alpha_to_one_enabled);
    mark(EngineHint, a.app_stage.Value() != b.app_stage.Value());
    mark(ColorFormats, a.color_formats != b.color_formats);
    mark(AlphaTestReference, a.alpha_test_ref != b.alpha_test_ref);
    mark(PointSize, a.point_size != b.point_size);
    mark(ViewportSwizzles, a.viewport_swizzles != b.viewport_swizzles);
    // The dynamic state flags (bits 0 to 5 and 20 of raw1) and the driver.
    mark(Features, ((a.raw1 ^ b.raw1) & 0x0010003F) != 0 || a.driver_id != b.driver_id ||
                       a.driver_version != b.driver_version);
    if (a.dynamic_vertex_input != 0 && b.dynamic_vertex_input != 0) {
        mark(VertexAttributeTypes, a.attribute_types != b.attribute_types);
    } else {
        mark(InstanceDivisors,
             a.enabled_divisors != b.enabled_divisors || a.binding_divisors != b.binding_divisors);
        bool formats = false;
        bool types = false;
        for (size_t i = 0; i < a.attributes.size(); ++i) {
            if (a.attributes[i].raw == b.attributes[i].raw) {
                continue;
            }
            formats = true;
            types =
                types || AttributeTypeClass(a.attributes[i]) != AttributeTypeClass(b.attributes[i]);
        }
        mark(VertexAttributes, formats);
        mark(VertexAttributeTypes, types);
    }
    const auto& da = a.dynamic_state;
    const auto& db = b.dynamic_state;
    mark(CullMode, da.cull_face != db.cull_face || da.cull_enable != db.cull_enable);
    mark(FrontFace, da.front_face != db.front_face);
    mark(Stencil,
         da.stencil_enable != db.stencil_enable || (da.raw2 & 0xFFFFFF) != (db.raw2 & 0xFFFFFF));
    mark(DepthTest,
         da.depth_test_enable != db.depth_test_enable || da.depth_test_func != db.depth_test_func);
    mark(DepthWrite, da.depth_write_enable != db.depth_write_enable);
    mark(DepthBoundsTest, da.depth_bounds_enable != db.depth_bounds_enable);
    if (a.extended_dynamic_state == 0 || b.extended_dynamic_state == 0) {
        mark(VertexStrides, a.vertex_strides != b.vertex_strides);
    }
    mark(PrimitiveRestart, da.primitive_restart_enable != db.primitive_restart_enable);
    mark(DepthBias, da.depth_bias_enable != db.depth_bias_enable);
    mark(RasterizerDiscard, da.rasterize_enable != db.rasterize_enable);
    mark(LogicOp, da.logic_op != db.logic_op);
    mark(LogicOpEnable, da.logic_op_enable != db.logic_op_enable);
    mark(DepthClamp, da.depth_clamp_disabled != db.depth_clamp_disabled);
    mark(LineStipple, da.line_stipple_enable != db.line_stipple_enable);
    if (a.extended_dynamic_state_3_blend == 0 || b.extended_dynamic_state_3_blend == 0) {
        bool blend = false;
        bool masks = false;
        for (size_t i = 0; i < a.attachments.size(); ++i) {
            const u32 x = a.attachments[i].raw ^ b.attachments[i].raw;
            masks = masks || (x & 0xF) != 0;
            blend = blend || (x & ~u32{0xF}) != 0;
        }
        mark(Blend, blend);
        mark(ColorMasks, masks);
    }
    return parts;
}

/// The fewest settings with which states that differ in `parts` share one pipeline.
inline Way WayFor(u64 parts) {
    bool eds1 = false;
    bool strides = false;
    bool eds2 = false;
    bool vertex_input = false;
    bool eds3 = false;
    for (u32 part = 0; part < PART_COUNT; ++part) {
        if (((parts >> part) & 1) == 0) {
            continue;
        }
        switch (PARTS[part].merge) {
        case Merge::Eds1:
            eds1 = true;
            break;
        case Merge::Eds1OrVertexInput:
            strides = true;
            break;
        case Merge::Eds2:
            eds2 = true;
            break;
        case Merge::VertexInput:
            vertex_input = true;
            break;
        case Merge::Eds3:
            eds3 = true;
            break;
        case Merge::Never:
            return WayNever;
        }
    }
    if (eds3) {
        return WayEds3;
    }
    if (vertex_input) {
        return eds1 || eds2 ? WayEdsAndVertexInput : WayVertexInput;
    }
    return eds2 ? WayEds2 : (eds1 || strides ? WayEds1 : WayNever);
}

inline std::string PartNames(u64 parts) {
    std::string names;
    for (u32 part = 0; part < PART_COUNT; ++part) {
        if (((parts >> part) & 1) != 0) {
            names += fmt::format("{}{}", names.empty() ? "" : ", ", PARTS[part].name);
        }
    }
    return names.empty() ? std::string{"nothing"} : names;
}

/// The values of the differing parts, for the log, such as "vertex strides 0: 32 vs 48".
inline std::string DescribeDifference(const FixedPipelineState& a, const FixedPipelineState& b,
                                      u64 parts) {
    std::string out;
    const auto add = [&out](std::string text) {
        out += fmt::format("{}{}", out.empty() ? "" : "; ", text);
    };
    const auto has = [parts](Part part) { return ((parts >> part) & 1) != 0; };
    if (has(Topology) || has(TopologyClass)) {
        add(fmt::format("topology {} vs {}", static_cast<u32>(a.topology.Value()),
                        static_cast<u32>(b.topology.Value())));
    }
    if (has(VertexStrides)) {
        std::string list;
        for (size_t i = 0; i < a.vertex_strides.size(); ++i) {
            if (a.vertex_strides[i] != b.vertex_strides[i]) {
                list += fmt::format(" {}: {} vs {}", i, a.vertex_strides[i], b.vertex_strides[i]);
            }
        }
        add(fmt::format("vertex strides{}", list));
    }
    if (has(VertexAttributes) || has(VertexAttributeTypes)) {
        std::string list;
        size_t shown = 0;
        for (size_t i = 0; i < a.attributes.size() && shown < 6; ++i) {
            const auto& x = a.attributes[i];
            const auto& y = b.attributes[i];
            if (x.raw == y.raw) {
                continue;
            }
            ++shown;
            list +=
                fmt::format(" {}: buffer {} offset {} type {} size {}{} vs buffer {} offset "
                            "{} type {} size {}{}",
                            i, x.buffer.Value(), x.offset.Value(), x.type.Value(), x.size.Value(),
                            x.enabled ? "" : " off", y.buffer.Value(), y.offset.Value(),
                            y.type.Value(), y.size.Value(), y.enabled ? "" : " off");
        }
        add(fmt::format("vertex attributes{}", list));
    }
    const auto& da = a.dynamic_state;
    const auto& db = b.dynamic_state;
    if (has(CullMode)) {
        add(fmt::format("cull {} face {} vs {} face {}", da.cull_enable ? "on" : "off",
                        da.cull_face.Value(), db.cull_enable ? "on" : "off", db.cull_face.Value()));
    }
    if (has(DepthTest) || has(DepthWrite)) {
        add(fmt::format("depth test {} function {} write {} vs test {} function {} write {}",
                        da.depth_test_enable.Value(), da.depth_test_func.Value(),
                        da.depth_write_enable.Value(), db.depth_test_enable.Value(),
                        db.depth_test_func.Value(), db.depth_write_enable.Value()));
    }
    if (has(Stencil)) {
        add(fmt::format("stencil {} {:06x} vs {} {:06x}", da.stencil_enable.Value(),
                        da.raw2 & 0xFFFFFF, db.stencil_enable.Value(), db.raw2 & 0xFFFFFF));
    }
    if (has(Blend) || has(ColorMasks)) {
        std::string list;
        for (size_t i = 0; i < a.attachments.size(); ++i) {
            if (a.attachments[i].raw != b.attachments[i].raw) {
                list += fmt::format(" {}: {:08x} vs {:08x}", i, a.attachments[i].raw,
                                    b.attachments[i].raw);
            }
        }
        add(fmt::format("blending and masks{}", list));
    }
    if (has(ColorFormats)) {
        std::string list;
        for (size_t i = 0; i < a.color_formats.size(); ++i) {
            if (a.color_formats[i] != b.color_formats[i]) {
                list +=
                    fmt::format(" {}: {:#x} vs {:#x}", i, a.color_formats[i], b.color_formats[i]);
            }
        }
        add(fmt::format("color formats{}", list));
    }
    if (has(AlphaTest) || has(AlphaTestReference)) {
        float x = 0;
        float y = 0;
        std::memcpy(&x, &a.alpha_test_ref, sizeof(x));
        std::memcpy(&y, &b.alpha_test_ref, sizeof(y));
        add(fmt::format("alpha test {} ref {} vs {} ref {}", a.alpha_test_func.Value(), x,
                        b.alpha_test_func.Value(), y));
    }
    if (has(PointSize)) {
        float x = 0;
        float y = 0;
        std::memcpy(&x, &a.point_size, sizeof(x));
        std::memcpy(&y, &b.point_size, sizeof(y));
        add(fmt::format("point size {} vs {}", x, y));
    }
    if (has(DepthBuffer)) {
        add(fmt::format("depth buffer {} format {} vs {} format {}", a.depth_enabled.Value(),
                        a.depth_format.Value(), b.depth_enabled.Value(), b.depth_format.Value()));
    }
    if (has(EngineHint)) {
        add(fmt::format("engine hint {} vs {}", static_cast<u32>(a.app_stage.Value()),
                        static_cast<u32>(b.app_stage.Value())));
    }
    const u64 described = (u64{1} << Topology) | (u64{1} << TopologyClass) |
                          (u64{1} << VertexStrides) | (u64{1} << VertexAttributes) |
                          (u64{1} << VertexAttributeTypes) | (u64{1} << CullMode) |
                          (u64{1} << DepthTest) | (u64{1} << DepthWrite) | (u64{1} << Stencil) |
                          (u64{1} << Blend) | (u64{1} << ColorMasks) | (u64{1} << ColorFormats) |
                          (u64{1} << AlphaTest) | (u64{1} << AlphaTestReference) |
                          (u64{1} << PointSize) | (u64{1} << DepthBuffer) | (u64{1} << EngineHint);
    if ((parts & ~described) != 0) {
        add(PartNames(parts & ~described));
    }
    return out;
}

/// Of the states `others` (the earlier pipelines of the same shaders), the one a new state
/// would most easily share a pipeline with: the fewest settings, then the fewest parts.
template <typename States>
std::pair<const FixedPipelineState*, u64> Closest(const FixedPipelineState& state,
                                                  const States& others) {
    const FixedPipelineState* best = nullptr;
    u64 best_parts = 0;
    std::pair<u32, u32> best_rank{WAY_COUNT, 0};
    for (const FixedPipelineState* other : others) {
        const u64 parts = DifferingParts(state, *other);
        const std::pair<u32, u32> rank{static_cast<u32>(WayFor(parts)),
                                       static_cast<u32>(std::popcount(parts))};
        if (best == nullptr || rank < best_rank) {
            best = other;
            best_parts = parts;
            best_rank = rank;
        }
    }
    return {best, best_parts};
}

/// Counts, for the pipelines compiled during play, how many were for shaders with no pipeline
/// yet and how many were one more state of shaders that had one, and what set those apart.
struct Tally {
    size_t new_shaders = 0;
    size_t variants = 0;
    std::array<size_t, PART_COUNT> by_part{};
    std::array<size_t, WAY_COUNT> by_way{};
    /// For the variants no setting merges: the parts that keep them apart.
    std::array<size_t, PART_COUNT> kept_apart_by{};
    size_t only_other_bytes = 0;

    void Add(u64 parts) {
        ++variants;
        if (parts == 0) {
            ++only_other_bytes;
            return;
        }
        const Way way = WayFor(parts);
        ++by_way[way];
        for (u32 part = 0; part < PART_COUNT; ++part) {
            if (((parts >> part) & 1) == 0) {
                continue;
            }
            ++by_part[part];
            if (way == WayNever && PARTS[part].merge == Merge::Never) {
                ++kept_apart_by[part];
            }
        }
    }

    template <size_t N>
    static std::string Counts(const std::array<size_t, N>& counts, const auto& name_of) {
        std::array<std::pair<size_t, size_t>, N> order{};
        for (size_t i = 0; i < N; ++i) {
            order[i] = {counts[i], N - i};
        }
        std::sort(order.rbegin(), order.rend());
        std::string out;
        for (const auto& [count, reverse_index] : order) {
            if (count == 0) {
                break;
            }
            out +=
                fmt::format("{}{} x{}", out.empty() ? "" : ", ", name_of(N - reverse_index), count);
        }
        return out.empty() ? std::string{"none"} : out;
    }

    std::string Summary() const {
        return fmt::format(
            "{} for shaders with no pipeline yet, {} more states of shaders that had one; the "
            "closest earlier state differs in: {}{}; they would share a pipeline with: {}; kept "
            "apart with every setting by: {}",
            new_shaders, variants, Counts(by_part, [](size_t i) { return PARTS[i].name; }),
            only_other_bytes > 0 ? fmt::format(", only bytes not read x{}", only_other_bytes)
                                 : std::string{},
            Counts(by_way, [](size_t i) { return WAY_NAMES[i]; }),
            Counts(kept_apart_by, [](size_t i) { return PARTS[i].name; }));
    }
};

} // namespace Vulkan::PreforgedState
