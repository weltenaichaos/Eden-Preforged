// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>

// Oodle decompression from ooz (jni/ooz/kraken.cpp, GPL-3.0-or-later, Copyright 2016 Powzix).
namespace Preforged::Ooz {

/// The decoder may write up to this many bytes past the end of its output.
constexpr size_t OUTPUT_SLACK = 64;

/// Unpacks a whole Oodle stream packed with Kraken, Mermaid, Selkie or Leviathan into exactly
/// `dst_len` bytes. `dst` must have OUTPUT_SLACK bytes of room after `dst_len`. Returns the
/// bytes written, or -1 when the data doesn't unpack. ooz was written for files it could trust;
/// the port adds the checks that fuzzing showed were missing, but hand it data whose sizes were
/// already checked.
int Kraken_Decompress(const unsigned char* src, size_t src_len, unsigned char* dst,
                      size_t dst_len);

} // namespace Preforged::Ooz
