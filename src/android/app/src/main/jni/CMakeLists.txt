# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# SPDX-FileCopyrightText: 2023 yuzu Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

add_library(yuzu-android SHARED
    emu_window/emu_window.cpp
    emu_window/emu_window.h
    native.cpp
    native.h
    native_config.cpp
    native_freedreno.cpp
    android_settings.cpp
    game_metadata.cpp
    native_log.cpp
    android_config.cpp
    android_config.h
    native_input.cpp
    native_post_processing.cpp
    crash_report.cpp
    crash_report.h
    gpu_driver_check.cpp
    gpu_driver_check.h
    shader_file_scan.cpp
    shader_file_scan.h
    unreal_links.cpp
    unreal_links.h
    unreal_package.cpp
    unreal_package.h
    unreal_pak.cpp
    unreal_pak.h
    unity_assets.cpp
    unity_assets.h
    embedded_streams.cpp
    embedded_streams.h
    ooz/kraken.cpp
    ooz/ooz.h
    ooz/ooz_platform.h
)

# ooz is C-style code from elsewhere that reads bytes through other pointer types, so it is
# built without strict aliasing and without the project's warnings.
set_source_files_properties(ooz/kraken.cpp PROPERTIES COMPILE_OPTIONS "-w;-fno-strict-aliasing")

set_property(TARGET yuzu-android PROPERTY IMPORTED_LOCATION ${FFmpeg_LIBRARY_DIR})

target_link_libraries(yuzu-android PRIVATE audio_core common core input_common frontend_common video_core)
target_link_libraries(yuzu-android PRIVATE android camera2ndk EGL glad jnigraphics log GLESv2)
target_link_libraries(yuzu-android PRIVATE ZLIB::ZLIB)
if (ARCHITECTURE_arm64)
    target_link_libraries(yuzu-android PRIVATE adrenotools)
endif()

target_link_libraries(yuzu-android PRIVATE ${FFmpeg_LIBRARIES} OpenSSL::SSL cpp-jwt::cpp-jwt)
if (ENABLE_UPDATE_CHECKER)
    target_compile_definitions(yuzu-android PUBLIC ENABLE_UPDATE_CHECKER)
endif()

set(CPACK_PACKAGE_EXECUTABLES ${CPACK_PACKAGE_EXECUTABLES} yuzu-android)

target_link_options(yuzu-android PRIVATE "-Wl,-Bsymbolic")
