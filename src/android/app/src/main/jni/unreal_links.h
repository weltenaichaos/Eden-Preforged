// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "common/common_types.h"

// Logs how the packages of an Unreal Engine 3 game refer to the shader code they hold, to learn
// which vertex and fragment shaders the game draws together. The shaders found in the game files
// can only be compiled ahead of time in pairs, and Unreal games give no program tables to pair
// them by. XCOM 2 keeps all its code in one package (UniShaderBuffer).
namespace Preforged::Unreal {

class LinkSurvey {
public:
    enum class Stage : u8 { Other, Vertex, Fragment };

    struct CodeHeader {
        u64 unpacked_offset; ///< Where Unpack handed out the code header
        u64 hash;            ///< The shader's hash, as in the pipeline cache
        Stage stage;
    };

    /// `overlap` is how many bytes each piece handed to SearchPiece repeats from the one before.
    explicit LinkSurvey(size_t overlap);
    ~LinkSurvey();

    /// Counts the objects of a package by class. `first_piece` is the start of what Unpack
    /// handed out for it, or the file itself when it is stored unpacked.
    void AddPackage(std::string_view name, std::span<const u8> file,
                    std::span<const u8> first_piece);

    /// Reads the package that holds the most shader code into its uncompressed layout, logs
    /// its layout and searches it for the offsets and indexes of its code. `headers` are the
    /// code headers the search found in it; `played` the pairs of shader hashes the game drew.
    void StudyCodePackage(std::string_view name, std::span<const u8> file,
                          std::span<const CodeHeader> headers,
                          std::span<const std::pair<u64, u64>> played,
                          const std::function<bool()>& stopped);

    /// Searches what another package unpacked to for the references StudyCodePackage learned.
    void SearchPiece(std::string_view name, std::span<const u8> piece, u64 offset);

    /// Logs what SearchPiece found and how the packages' objects divide into classes.
    void Report() const;

    /// Vertex and fragment shader hashes that a field the played pairs share ties together,
    /// most likely first. Empty when StudyCodePackage found no such field.
    [[nodiscard]] const std::vector<std::pair<u64, u64>>& LearnedPairs() const;

    /// Vertex and fragment shader hashes that ids SearchPiece found outside shader and material
    /// objects tie together, when one kind of id there lists the played pairs, most likely
    /// first. Call once, after all packages are searched.
    [[nodiscard]] const std::vector<std::pair<u64, u64>>& LearnFromOtherPackages();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Preforged::Unreal
