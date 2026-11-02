// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <sstream>
#include <utility>

#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "common/dynamic_library.h"
#include "common/logging.h"
#include "jni/gpu_driver_check.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Preforged {

namespace {

// ELF parsing (ListVersionedImports and its helpers). Little-endian 64-bit only, which is what
// Android drivers for arm64 are.
constexpr u32 SHT_DYNSYM_TYPE = 11;
constexpr u32 SHT_GNU_VERNEED_TYPE = 0x6ffffffe;
constexpr u32 SHT_GNU_VERSYM_TYPE = 0x6fffffff;
constexpr u8 STB_WEAK_BIND = 2;
constexpr size_t SECTION_HEADER_SIZE = 64;
constexpr size_t SYMBOL_SIZE = 24;
constexpr size_t VERSION_ENTRY_SIZE = 16;
// More than any library has; stops a broken chain from looping.
constexpr size_t MAX_VERSION_ENTRIES = 4096;

template <typename T>
std::optional<T> ReadAt(std::span<const u8> data, u64 offset) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) {
        return std::nullopt;
    }
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

struct Section {
    u32 type = 0;
    u64 offset = 0;
    u64 size = 0;
    u32 link = 0;
};

std::optional<Section> ReadSection(std::span<const u8> elf, u64 table, u16 index) {
    const u64 at = table + u64{index} * SECTION_HEADER_SIZE;
    const auto type = ReadAt<u32>(elf, at + 0x04);
    const auto offset = ReadAt<u64>(elf, at + 0x18);
    const auto size = ReadAt<u64>(elf, at + 0x20);
    const auto link = ReadAt<u32>(elf, at + 0x28);
    if (!type || !offset || !size || !link || *offset > elf.size() ||
        elf.size() - *offset < *size) {
        return std::nullopt;
    }
    return Section{*type, *offset, *size, *link};
}

// A NUL-ended string at index in a string table, or nothing when it runs off the table.
std::optional<std::string> ReadString(std::span<const u8> elf, const Section& table, u64 index) {
    if (index >= table.size) {
        return std::nullopt;
    }
    const auto* const begin = reinterpret_cast<const char*>(elf.data() + table.offset + index);
    const size_t room = table.size - index;
    const size_t length = strnlen(begin, room);
    if (length == room) {
        return std::nullopt;
    }
    return std::string(begin, length);
}

} // namespace

std::vector<VersionedImport> ListVersionedImports(std::span<const u8> elf) {
    std::vector<VersionedImport> imports;
    if (elf.size() < 0x40 || std::memcmp(elf.data(), "\x7f" "ELF", 4) != 0 || elf[4] != 2 ||
        elf[5] != 1) {
        return imports;
    }
    const auto table = ReadAt<u64>(elf, 0x28);
    const auto entry_size = ReadAt<u16>(elf, 0x3a);
    const auto count = ReadAt<u16>(elf, 0x3c);
    if (!table || !entry_size || !count || *entry_size != SECTION_HEADER_SIZE) {
        return imports;
    }

    std::optional<Section> symbols;
    std::optional<Section> versions;
    std::optional<Section> needs;
    for (u16 index = 0; index < *count; ++index) {
        const auto section = ReadSection(elf, *table, index);
        if (!section) {
            continue;
        }
        if (section->type == SHT_DYNSYM_TYPE) {
            symbols = section;
        } else if (section->type == SHT_GNU_VERSYM_TYPE) {
            versions = section;
        } else if (section->type == SHT_GNU_VERNEED_TYPE) {
            needs = section;
        }
    }
    if (!symbols || !versions || !needs || symbols->link >= *count || needs->link >= *count) {
        return imports;
    }
    const auto symbol_names = ReadSection(elf, *table, static_cast<u16>(symbols->link));
    const auto need_names = ReadSection(elf, *table, static_cast<u16>(needs->link));
    if (!symbol_names || !need_names) {
        return imports;
    }

    // Version index -> (library, version name), from the chain of needed libraries, each with a
    // chain of versions.
    std::map<u16, std::pair<std::string, std::string>> named_versions;
    u64 need_at = needs->offset;
    const u64 needs_end = needs->offset + needs->size;
    for (size_t step = 0; step < MAX_VERSION_ENTRIES && need_at + VERSION_ENTRY_SIZE <= needs_end;
         ++step) {
        const auto version_count = ReadAt<u16>(elf, need_at + 0x02);
        const auto file_name = ReadAt<u32>(elf, need_at + 0x04);
        const auto first_aux = ReadAt<u32>(elf, need_at + 0x08);
        const auto next = ReadAt<u32>(elf, need_at + 0x0c);
        if (!version_count || !file_name || !first_aux || !next) {
            break;
        }
        const auto file = ReadString(elf, *need_names, *file_name);
        u64 aux_at = need_at + *first_aux;
        for (u16 aux = 0; file && aux < *version_count && aux_at + VERSION_ENTRY_SIZE <= needs_end;
             ++aux) {
            const auto index = ReadAt<u16>(elf, aux_at + 0x06);
            const auto version_name = ReadAt<u32>(elf, aux_at + 0x08);
            const auto aux_next = ReadAt<u32>(elf, aux_at + 0x0c);
            if (!index || !version_name || !aux_next) {
                break;
            }
            if (const auto version = ReadString(elf, *need_names, *version_name)) {
                named_versions[*index & 0x7fff] = {*file, *version};
            }
            if (*aux_next == 0) {
                break;
            }
            aux_at += *aux_next;
        }
        if (*next == 0) {
            break;
        }
        need_at += *next;
    }

    const u64 symbol_count = std::min(symbols->size / SYMBOL_SIZE, versions->size / 2);
    for (u64 index = 1; index < symbol_count; ++index) {
        const u64 at = symbols->offset + index * SYMBOL_SIZE;
        const auto name_index = ReadAt<u32>(elf, at);
        const auto info = ReadAt<u8>(elf, at + 0x04);
        const auto section_index = ReadAt<u16>(elf, at + 0x06);
        const auto version = ReadAt<u16>(elf, versions->offset + index * 2);
        if (!name_index || !info || !section_index || !version || *section_index != 0 ||
            *name_index == 0 || (*info >> 4) == STB_WEAK_BIND) {
            continue;
        }
        const auto found = named_versions.find(static_cast<u16>(*version & 0x7fff));
        if (found == named_versions.end()) {
            continue;
        }
        if (auto name = ReadString(elf, *symbol_names, *name_index)) {
            imports.push_back({found->second.first, found->second.second, std::move(*name)});
        }
    }
    return imports;
}

namespace {

// Whether the process has the file mapped, which it has once the linker loaded it.
bool IsMapped(const std::string& path) {
    char resolved[PATH_MAX];
    const std::string real_path = realpath(path.c_str(), resolved) ? resolved : path;
    struct stat file_stat {};
    const bool have_inode = stat(path.c_str(), &file_stat) == 0;
    const std::string file_name = path.substr(path.find_last_of('/') + 1);

    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        std::istringstream fields(line);
        std::string address, permissions, offset, device;
        u64 inode = 0;
        fields >> address >> permissions >> offset >> device >> inode;
        std::string mapped;
        std::getline(fields >> std::ws, mapped);
        if (mapped.empty()) {
            continue;
        }
        if (mapped == real_path || mapped == path) {
            return true;
        }
        if (have_inode && inode == static_cast<u64>(file_stat.st_ino) &&
            mapped.size() > file_name.size() && mapped.ends_with("/" + file_name)) {
            return true;
        }
    }
    return false;
}

std::string DescribeGpu(const Vulkan::vk::Instance& instance,
                        const Vulkan::vk::InstanceDispatch& dld) {
    const std::vector<VkPhysicalDevice> devices = instance.EnumeratePhysicalDevices();
    if (devices.empty()) {
        return "no GPU";
    }
    const Vulkan::vk::PhysicalDevice device(devices.front(), dld);
    const VkPhysicalDeviceProperties properties = device.GetProperties();
    std::string text = fmt::format("{}, Vulkan {}.{}.{}", properties.deviceName,
                                   VK_API_VERSION_MAJOR(properties.apiVersion),
                                   VK_API_VERSION_MINOR(properties.apiVersion),
                                   VK_API_VERSION_PATCH(properties.apiVersion));
    if (properties.apiVersion >= VK_API_VERSION_1_2) {
        VkPhysicalDeviceDriverProperties driver{};
        driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &driver;
        device.GetProperties2(properties2);
        std::string info(driver.driverInfo, strnlen(driver.driverInfo, VK_MAX_DRIVER_INFO_SIZE));
        for (char& c : info) {
            if (c == '\n' || c == '\r') {
                c = ' ';
            }
        }
        text += fmt::format(", driver '{}' '{}'",
                            std::string_view(driver.driverName,
                                             strnlen(driver.driverName, VK_MAX_DRIVER_NAME_SIZE)),
                            info);
    }
    return text;
}

// The functions the driver needs from libc, libm or libdl that aren't here in that version.
// A driver built for a newer Android than the device runs fails to load this way.
void LogMissingSystemFunctions(const std::string& driver_path, const std::string& driver_name) {
    std::ifstream file(driver_path, std::ios::binary | std::ios::ate);
    if (!file) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: could not open {} to check it",
                    driver_path);
        return;
    }
    const std::streamoff size = file.tellg();
    constexpr std::streamoff MAX_DRIVER_SIZE = 256LL << 20;
    if (size <= 0 || size > MAX_DRIVER_SIZE) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: {} is {} bytes, not checked", driver_name,
                    static_cast<s64>(size));
        return;
    }
    std::vector<u8> data(static_cast<size_t>(size));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), size);
    if (!file) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: could not read {}", driver_path);
        return;
    }

    const std::vector<VersionedImport> imports = ListVersionedImports(data);
    std::map<std::string, void*> libraries;
    size_t checked = 0;
    std::vector<std::string> missing;
    for (const VersionedImport& import : imports) {
        if (import.file != "libc.so" && import.file != "libm.so" && import.file != "libdl.so") {
            continue;
        }
        auto [entry, added] = libraries.try_emplace(import.file, nullptr);
        if (added) {
            entry->second = dlopen(import.file.c_str(), RTLD_NOW | RTLD_NOLOAD);
        }
        if (!entry->second) {
            continue;
        }
        ++checked;
        if (!dlvsym(entry->second, import.name.c_str(), import.version.c_str())) {
            missing.push_back(fmt::format("{}@{}", import.name, import.version));
        }
    }
    for (const auto& [name, handle] : libraries) {
        if (handle) {
            dlclose(handle);
        }
    }

    if (imports.empty()) {
        LOG_WARNING(Frontend,
                    "[Preforged] GPU driver: {} lists no versioned functions, so it may not be "
                    "an arm64 library",
                    driver_name);
    } else if (missing.empty()) {
        LOG_INFO(Frontend,
                 "[Preforged] GPU driver: all {} versioned libc, libm and libdl functions {} "
                 "needs exist on this device",
                 checked, driver_name);
    } else {
        LOG_WARNING(Frontend,
                    "[Preforged] GPU driver: {} needs {} functions this Android version doesn't "
                    "have, so it can't load on this device: {}",
                    driver_name, missing.size(), fmt::join(missing, ", "));
    }
}

// adrenotools' hook writes why a driver failed to the system log, not to ours.
void LogLoaderErrors() {
    const std::string command =
        fmt::format("logcat -d -v brief --pid={} -s hook_impl:V adrenotools:V 2>&1", getpid());
    FILE* const pipe = popen(command.c_str(), "r");
    if (!pipe) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: could not read the system log ({})",
                    std::strerror(errno));
        return;
    }
    std::vector<std::string> errors;
    char buffer[1024];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        std::string line(buffer);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        if (line.find("failed") != std::string::npos) {
            errors.push_back(std::move(line));
        }
    }
    pclose(pipe);

    if (errors.empty()) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: the system log has no driver loader "
                              "error for this app");
        return;
    }
    // The last ones belong to this start.
    constexpr size_t MAX_LINES = 4;
    const size_t first = errors.size() > MAX_LINES ? errors.size() - MAX_LINES : 0;
    for (size_t i = first; i < errors.size(); ++i) {
        LOG_WARNING(Frontend, "[Preforged] GPU driver: system log: {}", errors[i]);
    }
}

} // namespace

void LogCustomDriverCheck(const Common::DynamicLibrary& library, const std::string& driver_dir,
                          const std::string& driver_name) {
    const std::string driver_path = driver_dir + driver_name;
    const auto start = std::chrono::steady_clock::now();
    std::string gpu;
    bool in_use = false;
    try {
        Vulkan::vk::InstanceDispatch dld;
        const Vulkan::vk::Instance instance =
            Vulkan::CreateInstance(library, dld, VK_API_VERSION_1_1);
        // Vulkan loads the driver while it starts, so the file is mapped now if it loaded.
        in_use = IsMapped(driver_path);
        gpu = DescribeGpu(instance, dld);
    } catch (const std::exception& e) {
        in_use = IsMapped(driver_path);
        gpu = fmt::format("nothing, Vulkan did not start ({})", e.what());
    }
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count();

    if (in_use) {
        LOG_INFO(Frontend,
                 "[Preforged] GPU driver: custom driver {} is in use, the GPU reports {} "
                 "(checked in {} ms)",
                 driver_name, gpu, ms);
        return;
    }
    LOG_WARNING(Frontend,
                "[Preforged] GPU driver: custom driver {} did not load, so the system driver "
                "runs instead; the GPU reports {} (checked in {} ms)",
                driver_name, gpu, ms);
    LogMissingSystemFunctions(driver_path, driver_name);
    LogLoaderErrors();
}

} // namespace Preforged
