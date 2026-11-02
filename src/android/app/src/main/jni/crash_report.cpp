// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/prctl.h>
#include <ucontext.h>
#include <unistd.h>
#include <unwind.h>

#include "jni/crash_report.h"

namespace {

constexpr std::array SIGNALS{SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE};
std::array<struct sigaction, SIGNALS.size()> previous_actions{};
std::array<char, 1024> log_path{};
std::atomic_bool reporting{false};

struct Frames {
    std::array<uintptr_t, 64> pcs{};
    size_t count = 0;
};

_Unwind_Reason_Code CollectFrame(_Unwind_Context* context, void* arg) {
    auto* const frames = static_cast<Frames*>(arg);
    const uintptr_t pc = _Unwind_GetIP(context);
    if (pc != 0) {
        frames->pcs[frames->count++] = pc;
    }
    return frames->count < frames->pcs.size() ? _URC_NO_REASON : _URC_END_OF_STACK;
}

const char* SignalName(int sig) {
    switch (sig) {
    case SIGSEGV:
        return "SIGSEGV";
    case SIGBUS:
        return "SIGBUS";
    case SIGABRT:
        return "SIGABRT";
    case SIGILL:
        return "SIGILL";
    case SIGFPE:
        return "SIGFPE";
    default:
        return "signal";
    }
}

void WriteFrame(int fd, const char* label, uintptr_t pc) {
    char line[512];
    Dl_info info{};
    int length;
    if (dladdr(reinterpret_cast<void*>(pc), &info) != 0 && info.dli_fname != nullptr) {
        const char* module = std::strrchr(info.dli_fname, '/');
        module = module ? module + 1 : info.dli_fname;
        const uintptr_t offset = pc - reinterpret_cast<uintptr_t>(info.dli_fbase);
        if (info.dli_sname != nullptr) {
            length = std::snprintf(line, sizeof(line), "  %s %s+0x%zx (%s+0x%zx)\n", label, module,
                                   static_cast<size_t>(offset), info.dli_sname,
                                   static_cast<size_t>(
                                       pc - reinterpret_cast<uintptr_t>(info.dli_saddr)));
        } else {
            length = std::snprintf(line, sizeof(line), "  %s %s+0x%zx\n", label, module,
                                   static_cast<size_t>(offset));
        }
    } else {
        length = std::snprintf(line, sizeof(line), "  %s 0x%zx (unknown module)\n", label,
                               static_cast<size_t>(pc));
    }
    if (length > 0) {
        void(write(fd, line, static_cast<size_t>(length)));
    }
}

void HandleCrash(int sig, siginfo_t* info, void* raw_context) {
    if (!reporting.exchange(true)) {
        const int fd = open(log_path.data(), O_WRONLY | O_APPEND | O_CLOEXEC);
        if (fd >= 0) {
            char thread_name[32]{};
            prctl(PR_GET_NAME, thread_name);
            char header[512];
            const int length = std::snprintf(
                header, sizeof(header),
                "\n=== Eden Preforged crash: %s (code %d) at address %p on thread '%s' (tid %d) "
                "===\n",
                SignalName(sig), info ? info->si_code : 0, info ? info->si_addr : nullptr,
                thread_name, gettid());
            if (length > 0) {
                void(write(fd, header, static_cast<size_t>(length)));
            }
#if defined(__aarch64__)
            if (raw_context != nullptr) {
                const auto* const context = static_cast<const ucontext_t*>(raw_context);
                WriteFrame(fd, "pc", static_cast<uintptr_t>(context->uc_mcontext.pc));
                WriteFrame(fd, "lr", static_cast<uintptr_t>(context->uc_mcontext.regs[30]));
            }
#endif
            Frames frames;
            _Unwind_Backtrace(CollectFrame, &frames);
            for (size_t i = 0; i < frames.count; ++i) {
                char label[16];
                std::snprintf(label, sizeof(label), "#%02zu", i);
                WriteFrame(fd, label, frames.pcs[i]);
            }
            constexpr char footer[] = "=== end of crash report ===\n";
            void(write(fd, footer, sizeof(footer) - 1));
            close(fd);
        }
    }

    // Hand the crash on as if this handler was never there.
    for (size_t i = 0; i < SIGNALS.size(); ++i) {
        if (SIGNALS[i] != sig) {
            continue;
        }
        const struct sigaction& previous = previous_actions[i];
        if ((previous.sa_flags & SA_SIGINFO) != 0 && previous.sa_sigaction != nullptr) {
            previous.sa_sigaction(sig, info, raw_context);
            return;
        }
        if (previous.sa_handler != SIG_DFL && previous.sa_handler != SIG_IGN &&
            previous.sa_handler != nullptr) {
            previous.sa_handler(sig);
            return;
        }
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

} // Anonymous namespace

void InstallCrashReport(const std::filesystem::path& log_file) {
    static std::atomic_bool installed{false};
    if (installed.exchange(true)) {
        return;
    }
    const std::string path = log_file.string();
    std::strncpy(log_path.data(), path.c_str(), log_path.size() - 1);

    struct sigaction action{};
    action.sa_sigaction = &HandleCrash;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    for (size_t i = 0; i < SIGNALS.size(); ++i) {
        sigaction(SIGNALS[i], &action, &previous_actions[i]);
    }
}
