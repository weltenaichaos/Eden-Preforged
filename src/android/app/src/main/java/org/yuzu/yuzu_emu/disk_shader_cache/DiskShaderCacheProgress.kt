// SPDX-FileCopyrightText: 2023 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

package org.yuzu.yuzu_emu.disk_shader_cache

import androidx.annotation.Keep
import androidx.lifecycle.ViewModelProvider
import org.yuzu.yuzu_emu.NativeLibrary
import org.yuzu.yuzu_emu.R
import org.yuzu.yuzu_emu.activities.EmulationActivity
import org.yuzu.yuzu_emu.model.EmulationViewModel
import org.yuzu.yuzu_emu.utils.Log

@Keep
object DiskShaderCacheProgress {
    private lateinit var emulationViewModel: EmulationViewModel

    private fun prepareViewModel() {
        emulationViewModel =
            ViewModelProvider(
                NativeLibrary.sEmulationActivity.get() as EmulationActivity
            )[EmulationViewModel::class.java]
    }

    @JvmStatic
    fun loadProgress(stage: Int, progress: Int, max: Int) {
        val emulationActivity = NativeLibrary.sEmulationActivity.get()
        if (emulationActivity == null) {
            Log.error("[DiskShaderCacheProgress] EmulationActivity not present")
            return
        }

        emulationActivity.runOnUiThread {
            when (LoadCallbackStage.values()[stage]) {
                LoadCallbackStage.Prepare -> {
                    prepareViewModel()
                    emulationViewModel.setShaderLoading(true)
                }

                LoadCallbackStage.Build -> emulationViewModel.updateProgress(
                    emulationActivity.getString(R.string.building_shaders),
                    progress,
                    max
                )

                LoadCallbackStage.Complete -> {
                    if (::emulationViewModel.isInitialized) {
                        emulationViewModel.setShaderLoading(false)
                    }
                }

                LoadCallbackStage.ScanGameFiles -> emulationViewModel.updateProgress(
                    emulationActivity.getString(R.string.scanning_game_files),
                    progress,
                    max
                )

                LoadCallbackStage.PrecompileGameShaders -> emulationViewModel.updateProgress(
                    withCount(
                        emulationActivity.getString(R.string.precompiling_game_shaders),
                        progress,
                        max
                    ),
                    progress,
                    max
                )

                LoadCallbackStage.PrecompileCpuCode -> emulationViewModel.updateProgress(
                    withCount(
                        emulationActivity.getString(R.string.precompiling_cpu_code),
                        progress,
                        max
                    ),
                    progress,
                    max
                )
            }
        }
    }

    // Shows how far along a compile stage is, e.g. "Precompiling game shaders  4321 / 24000", so a
    // long first start reads as real progress instead of a bar that looks stuck.
    private fun withCount(message: String, progress: Int, max: Int): String =
        if (max > 0) "$message  $progress / $max" else message

    // Equivalent to VideoCore::LoadCallbackStage
    enum class LoadCallbackStage {
        Prepare, Build, Complete, ScanGameFiles, PrecompileGameShaders, PrecompileCpuCode
    }
}
