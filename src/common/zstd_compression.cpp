// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstring>
#include <zstd.h>

#include "common/zstd_compression.h"

namespace Common::Compression {

std::vector<u8> CompressDataZSTD(const u8* source, std::size_t source_size, s32 compression_level) {
    compression_level = std::clamp(compression_level, 1, ZSTD_maxCLevel());

    const std::size_t max_compressed_size = ZSTD_compressBound(source_size);
    std::vector<u8> compressed(max_compressed_size);

    const std::size_t compressed_size =
        ZSTD_compress(compressed.data(), compressed.size(), source, source_size, compression_level);

    if (ZSTD_isError(compressed_size)) {
        // Compression failed
        return {};
    }

    compressed.resize(compressed_size);

    return compressed;
}

std::vector<u8> CompressDataZSTDDefault(const u8* source, std::size_t source_size) {
    return CompressDataZSTD(source, source_size, ZSTD_CLEVEL_DEFAULT);
}

std::vector<u8> DecompressDataZSTD(std::span<const u8> compressed) {
    const std::size_t decompressed_size =
        ZSTD_getFrameContentSize(compressed.data(), compressed.size());
    std::vector<u8> decompressed(decompressed_size);

    const std::size_t uncompressed_result_size = ZSTD_decompress(
        decompressed.data(), decompressed.size(), compressed.data(), compressed.size());

    if (decompressed_size != uncompressed_result_size || ZSTD_isError(uncompressed_result_size)) {
        // Decompression failed
        return {};
    }
    return decompressed;
}

bool IsZstdDictionary(std::span<const u8> data) {
    constexpr u32 DICTIONARY_MAGIC = 0xEC30A437;
    u32 magic = 0;
    if (data.size() < 8) {
        return false;
    }
    std::memcpy(&magic, data.data(), sizeof(magic));
    return magic == DICTIONARY_MAGIC;
}

u32 ZstdFrameDictionaryId(std::span<const u8> frame) {
    return ZSTD_getDictID_fromFrame(frame.data(), frame.size());
}

struct ZstdDictionaries::Impl {
    ZSTD_DCtx* context = ZSTD_createDCtx();
    std::map<u32, ZSTD_DDict*> dictionaries;
    std::map<u32, size_t> sizes;

    ~Impl() {
        for (const auto& [id, dictionary] : dictionaries) {
            ZSTD_freeDDict(dictionary);
        }
        ZSTD_freeDCtx(context);
    }

    std::vector<u8> Run(std::span<const u8> in, size_t max_size, const ZSTD_DDict* dictionary) {
        if (context == nullptr) {
            return {};
        }
        ZSTD_DCtx_reset(context, ZSTD_reset_session_and_parameters);
        if (ZSTD_isError(ZSTD_DCtx_refDDict(context, dictionary))) {
            return {};
        }
        const unsigned long long content = ZSTD_getFrameContentSize(in.data(), in.size());
        if (content == ZSTD_CONTENTSIZE_ERROR ||
            (content != ZSTD_CONTENTSIZE_UNKNOWN && content > max_size)) {
            return {};
        }
        // The size the first frame records, plus room for more frames after it.
        std::vector<u8> out(content == ZSTD_CONTENTSIZE_UNKNOWN
                                ? std::min<size_t>(max_size, size_t{1} << 20)
                                : std::max<size_t>(static_cast<size_t>(content), 1));
        ZSTD_inBuffer input{in.data(), in.size(), 0};
        size_t written = 0;
        while (true) {
            ZSTD_outBuffer output{out.data() + written, out.size() - written, 0};
            const size_t result = ZSTD_decompressStream(context, &output, &input);
            if (ZSTD_isError(result)) {
                return {};
            }
            written += output.pos;
            if (result == 0 && input.pos == input.size) {
                break;
            }
            if (output.pos == output.size) {
                if (out.size() >= max_size) {
                    return {};
                }
                out.resize(std::min(max_size, out.size() * 2));
            } else if (input.pos == input.size) {
                return {}; // The data ends inside a frame.
            }
        }
        out.resize(written);
        return out;
    }
};

ZstdDictionaries::ZstdDictionaries() : impl{std::make_unique<Impl>()} {}

ZstdDictionaries::~ZstdDictionaries() = default;

u32 ZstdDictionaries::Add(std::span<const u8> dictionary) {
    if (!IsZstdDictionary(dictionary)) {
        return 0;
    }
    const u32 id = ZSTD_getDictID_fromDict(dictionary.data(), dictionary.size());
    if (id == 0 || impl->dictionaries.contains(id)) {
        return 0;
    }
    ZSTD_DDict* const digested = ZSTD_createDDict(dictionary.data(), dictionary.size());
    if (digested == nullptr) {
        return 0;
    }
    impl->dictionaries.emplace(id, digested);
    impl->sizes.emplace(id, dictionary.size());
    return id;
}

bool ZstdDictionaries::Has(u32 id) const {
    return impl->dictionaries.contains(id);
}

const std::map<u32, size_t>& ZstdDictionaries::Sizes() const {
    return impl->sizes;
}

std::vector<u8> ZstdDictionaries::Decompress(std::span<const u8> compressed, size_t max_size,
                                             u32& used) {
    used = 0;
    const u32 id = ZstdFrameDictionaryId(compressed);
    if (id != 0) {
        const auto it = impl->dictionaries.find(id);
        if (it == impl->dictionaries.end()) {
            return {};
        }
        std::vector<u8> out = impl->Run(compressed, max_size, it->second);
        used = out.empty() ? 0 : id;
        return out;
    }
    if (std::vector<u8> out = impl->Run(compressed, max_size, nullptr); !out.empty()) {
        return out;
    }
    // A frame packed with a dictionary does not have to name it. Another dictionary than the one
    // it was packed with often unpacks it too, into other bytes, so only a frame with a checksum
    // tells which dictionary is right.
    constexpr size_t DESCRIPTOR = 4;
    constexpr u8 CHECKSUM_FLAG = 1 << 2;
    if (compressed.size() <= DESCRIPTOR || (compressed[DESCRIPTOR] & CHECKSUM_FLAG) == 0) {
        return {};
    }
    for (const auto& [dictionary_id, dictionary] : impl->dictionaries) {
        if (std::vector<u8> out = impl->Run(compressed, max_size, dictionary); !out.empty()) {
            used = dictionary_id;
            return out;
        }
    }
    return {};
}

} // namespace Common::Compression
