// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

package org.yuzu.yuzu_emu.updater

import android.widget.Toast
import androidx.fragment.app.FragmentActivity
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.android.material.textview.MaterialTextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import org.yuzu.yuzu_emu.BuildConfig
import org.yuzu.yuzu_emu.R
import java.io.File

/**
 * Installs the newest Eden Preforged test build. CI publishes every finished build as the
 * GitHub release tagged [TAG], with the build number in the APK name.
 */
object PreforgedUpdater {
    private const val TAG = "preforged-latest"
    private const val RELEASE_URL =
        "https://api.github.com/repos/weltenaichaos/Eden-Preforged/releases/tags/$TAG"
    private val BUILD_IN_NAME = Regex("""build-(\d+)\.apk$""")

    private data class Build(val number: Int, val notes: String, val url: String)

    val installedBuild: Int get() = BuildConfig.PREFORGED_BUILD

    fun checkAndInstall(activity: FragmentActivity) {
        Toast.makeText(activity, R.string.preforged_update_checking, Toast.LENGTH_SHORT).show()
        CoroutineScope(Dispatchers.IO).launch {
            val newest = try {
                fetchNewest()
            } catch (e: Exception) {
                null
            }
            withContext(Dispatchers.Main) {
                when {
                    newest == null -> MaterialAlertDialogBuilder(activity)
                        .setTitle(R.string.preforged_update)
                        .setMessage(R.string.preforged_update_check_failed)
                        .setPositiveButton(android.R.string.ok, null)
                        .show()

                    newest.number <= installedBuild -> MaterialAlertDialogBuilder(activity)
                        .setTitle(R.string.preforged_update)
                        .setMessage(
                            activity.getString(R.string.preforged_update_newest, installedBuild)
                        )
                        .setPositiveButton(android.R.string.ok, null)
                        .show()

                    else -> MaterialAlertDialogBuilder(activity)
                        .setTitle(R.string.preforged_update)
                        .setMessage(
                            activity.getString(
                                R.string.preforged_update_available,
                                newest.number,
                                installedBuild,
                                newest.notes
                            )
                        )
                        .setPositiveButton(R.string.preforged_update_install) { _, _ ->
                            download(activity, newest)
                        }
                        .setNegativeButton(android.R.string.cancel, null)
                        .show()
                }
            }
        }
    }

    private fun fetchNewest(): Build? {
        val request = Request.Builder()
            .url(RELEASE_URL)
            .header("Accept", "application/vnd.github+json")
            .build()
        OkHttpClient().newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                return null
            }
            val release = JSONObject(response.body?.string() ?: return null)
            val assets = release.getJSONArray("assets")
            for (i in 0 until assets.length()) {
                val asset = assets.getJSONObject(i)
                val match = BUILD_IN_NAME.find(asset.getString("name")) ?: continue
                return Build(
                    number = match.groupValues[1].toInt(),
                    notes = release.optString("body", ""),
                    url = asset.getString("browser_download_url")
                )
            }
        }
        return null
    }

    private fun download(activity: FragmentActivity, build: Build) {
        val view = activity.layoutInflater.inflate(R.layout.dialog_download_progress, null)
        val bar = view.findViewById<LinearProgressIndicator>(R.id.download_progress_bar)
        val message = view.findViewById<MaterialTextView>(R.id.download_progress_message)
        val dialog = MaterialAlertDialogBuilder(activity)
            .setTitle(activity.getString(R.string.preforged_update_downloading, build.number))
            .setView(view)
            .setCancelable(false)
            .show()

        val apkFile = File(activity.cacheDir, "eden-preforged-build-${build.number}.apk")
        APKDownloader(build.url, apkFile).download(
            onProgress = { progress ->
                activity.runOnUiThread {
                    bar.progress = progress
                    message.text = activity.getString(R.string.percent, progress)
                }
            },
            onComplete = { success ->
                activity.runOnUiThread {
                    dialog.dismiss()
                    if (!success) {
                        Toast.makeText(
                            activity,
                            R.string.update_download_failed,
                            Toast.LENGTH_LONG
                        ).show()
                        return@runOnUiThread
                    }
                    APKInstaller(activity).install(
                        apkFile,
                        onComplete = {},
                        onFailure = { exception ->
                            Toast.makeText(
                                activity,
                                activity.getString(
                                    R.string.update_install_failed,
                                    exception.message
                                ),
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    )
                }
            }
        )
    }
}
