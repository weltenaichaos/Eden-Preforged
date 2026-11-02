// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// What ooz's kraken.cpp took from Windows and SSE, in plain C++ so it builds for ARM64. The
// vector functions do exactly what the SSE instructions of the same name do, lane by lane, on a
// little-endian 16-byte value.

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef OOZ_DEBUG
#define OOZ_ASSERT(x) assert(x)
#else
#define OOZ_ASSERT(x) ((void)0)
#endif

#define OOZ_FORCEINLINE inline __attribute__((always_inline))

namespace Preforged::Ooz {

typedef unsigned char byte;
typedef unsigned char uint8;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef int64_t int64;
typedef int32_t int32;
typedef uint16_t uint16;
typedef int16_t int16;
typedef unsigned int uint;

inline uint16 OozByteSwap16(uint16 x) {
    return __builtin_bswap16(x);
}
inline uint32 OozByteSwap32(uint32 x) {
    return __builtin_bswap32(x);
}
inline uint64 OozByteSwap64(uint64 x) {
    return __builtin_bswap64(x);
}

/// Index of the highest set bit, as MSVC's _BitScanReverse. That leaves the index undefined for
/// 0; this sets it to 0 instead of calling __builtin_clz(0).
template <typename T>
inline unsigned char OozBitScanReverse(T* index, uint32 x) {
    *index = static_cast<T>(x == 0 ? 0 : 31 - __builtin_clz(x));
    return x != 0;
}
/// Index of the lowest set bit, as MSVC's _BitScanForward, 0 for 0.
template <typename T>
inline unsigned char OozBitScanForward(T* index, uint32 x) {
    *index = static_cast<T>(x == 0 ? 0 : __builtin_ctz(x));
    return x != 0;
}

inline uint32 OozRotl(uint32 x, int n) {
    n &= 31;
    return n == 0 ? x : (x << n) | (x >> (32 - n));
}

// --- The SSE2 operations the decoder uses ---

struct ooz_m128i {
    uint8 b[16];
};
/// The decoder casts to a float vector only to store the high half.
using ooz_m128 = ooz_m128i;
/// Only used as a pointer type for the 8-byte store.
struct ooz_m64 {
    uint8 b[8];
};

constexpr int OOZ_HINT_T0 = 0;

inline uint16 Lane16(const ooz_m128i& v, int i) {
    uint16 x;
    std::memcpy(&x, v.b + 2 * i, 2);
    return x;
}
inline void SetLane16(ooz_m128i& v, int i, uint16 x) {
    std::memcpy(v.b + 2 * i, &x, 2);
}
inline uint32 Lane32(const ooz_m128i& v, int i) {
    uint32 x;
    std::memcpy(&x, v.b + 4 * i, 4);
    return x;
}
inline void SetLane32(ooz_m128i& v, int i, uint32 x) {
    std::memcpy(v.b + 4 * i, &x, 4);
}

inline ooz_m128i ooz_mm_loadu_si128(const ooz_m128i* p) {
    ooz_m128i v;
    std::memcpy(v.b, p, 16);
    return v;
}
inline void ooz_mm_storeu_si128(ooz_m128i* p, const ooz_m128i& v) {
    std::memcpy(p, v.b, 16);
}
inline ooz_m128i ooz_mm_loadl_epi64(const ooz_m128i* p) {
    ooz_m128i v{};
    std::memcpy(v.b, p, 8);
    return v;
}
inline void ooz_mm_storel_epi64(ooz_m128i* p, const ooz_m128i& v) {
    std::memcpy(p, v.b, 8);
}
inline void ooz_mm_storeh_pi(ooz_m64* p, const ooz_m128& v) {
    std::memcpy(p, v.b + 8, 8);
}
inline ooz_m128 ooz_mm_castsi128_ps(const ooz_m128i& v) {
    return v;
}

inline ooz_m128i ooz_mm_set1_epi8(int x) {
    ooz_m128i v;
    std::memset(v.b, static_cast<uint8>(x), 16);
    return v;
}
inline ooz_m128i ooz_mm_set1_epi16(int x) {
    ooz_m128i v;
    for (int i = 0; i < 8; i++) {
        SetLane16(v, i, static_cast<uint16>(x));
    }
    return v;
}
inline ooz_m128i ooz_mm_set1_epi32(int x) {
    ooz_m128i v;
    for (int i = 0; i < 4; i++) {
        SetLane32(v, i, static_cast<uint32>(x));
    }
    return v;
}
/// Arguments go from the highest lane to the lowest, as in SSE.
inline ooz_m128i ooz_mm_set_epi16(int e7, int e6, int e5, int e4, int e3, int e2, int e1,
                                  int e0) {
    const int lanes[8] = {e0, e1, e2, e3, e4, e5, e6, e7};
    ooz_m128i v;
    for (int i = 0; i < 8; i++) {
        SetLane16(v, i, static_cast<uint16>(lanes[i]));
    }
    return v;
}
inline ooz_m128i ooz_mm_cvtsi32_si128(int x) {
    ooz_m128i v{};
    SetLane32(v, 0, static_cast<uint32>(x));
    return v;
}

inline ooz_m128i ooz_mm_add_epi8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = static_cast<uint8>(a.b[i] + c.b[i]);
    }
    return r;
}
inline ooz_m128i ooz_mm_sub_epi8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = static_cast<uint8>(a.b[i] - c.b[i]);
    }
    return r;
}
inline ooz_m128i ooz_mm_add_epi16(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        SetLane16(r, i, static_cast<uint16>(Lane16(a, i) + Lane16(c, i)));
    }
    return r;
}
inline ooz_m128i ooz_mm_sub_epi16(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        SetLane16(r, i, static_cast<uint16>(Lane16(a, i) - Lane16(c, i)));
    }
    return r;
}
inline ooz_m128i ooz_mm_and_si128(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = a.b[i] & c.b[i];
    }
    return r;
}
inline ooz_m128i ooz_mm_xor_si128(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = a.b[i] ^ c.b[i];
    }
    return r;
}
inline ooz_m128i ooz_mm_max_epu8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = a.b[i] > c.b[i] ? a.b[i] : c.b[i];
    }
    return r;
}
inline ooz_m128i ooz_mm_cmpeq_epi8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        r.b[i] = a.b[i] == c.b[i] ? 0xFF : 0;
    }
    return r;
}
inline ooz_m128i ooz_mm_cmpgt_epi16(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        const bool greater = static_cast<int16>(Lane16(a, i)) > static_cast<int16>(Lane16(c, i));
        SetLane16(r, i, greater ? 0xFFFF : 0);
    }
    return r;
}

/// Shifts the whole value up by `bytes` bytes, filling with zeros.
inline ooz_m128i ooz_mm_slli_si128(const ooz_m128i& a, int bytes) {
    ooz_m128i r{};
    if (bytes < 16) {
        std::memcpy(r.b + bytes, a.b, 16 - bytes);
    }
    return r;
}
inline ooz_m128i ooz_mm_srli_epi16(const ooz_m128i& a, int bits) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        SetLane16(r, i, bits > 15 ? 0 : static_cast<uint16>(Lane16(a, i) >> bits));
    }
    return r;
}
inline ooz_m128i ooz_mm_srai_epi16(const ooz_m128i& a, int bits) {
    ooz_m128i r;
    const int shift = bits > 15 ? 15 : bits;
    for (int i = 0; i < 8; i++) {
        SetLane16(r, i, static_cast<uint16>(static_cast<int16>(Lane16(a, i)) >> shift));
    }
    return r;
}

inline ooz_m128i ooz_mm_unpacklo_epi8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        r.b[2 * i] = a.b[i];
        r.b[2 * i + 1] = c.b[i];
    }
    return r;
}
inline ooz_m128i ooz_mm_unpackhi_epi8(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 8; i++) {
        r.b[2 * i] = a.b[8 + i];
        r.b[2 * i + 1] = c.b[8 + i];
    }
    return r;
}
inline ooz_m128i ooz_mm_unpacklo_epi16(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 4; i++) {
        SetLane16(r, 2 * i, Lane16(a, i));
        SetLane16(r, 2 * i + 1, Lane16(c, i));
    }
    return r;
}
/// Lane i of the result is lane ((order >> 2i) & 3) of `a`.
inline ooz_m128i ooz_mm_shuffle_epi32(const ooz_m128i& a, int order) {
    ooz_m128i r;
    for (int i = 0; i < 4; i++) {
        SetLane32(r, i, Lane32(a, (order >> (2 * i)) & 3));
    }
    return r;
}
/// Narrows signed 16-bit lanes to signed bytes with saturation: `a` fills bytes 0 to 7, `c` 8 to 15.
inline ooz_m128i ooz_mm_packs_epi16(const ooz_m128i& a, const ooz_m128i& c) {
    ooz_m128i r;
    for (int i = 0; i < 16; i++) {
        const int x = static_cast<int16>(Lane16(i < 8 ? a : c, i & 7));
        r.b[i] = static_cast<uint8>(static_cast<int8_t>(x < -128 ? -128 : x > 127 ? 127 : x));
    }
    return r;
}
/// Bit i of the result is the top bit of byte i.
inline int ooz_mm_movemask_epi8(const ooz_m128i& a) {
    int mask = 0;
    for (int i = 0; i < 16; i++) {
        mask |= (a.b[i] >> 7) << i;
    }
    return mask;
}

inline void ooz_mm_prefetch(const char* p, int) {
    __builtin_prefetch(p);
}

} // namespace Preforged::Ooz
