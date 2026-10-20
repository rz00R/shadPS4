// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
//
// SHADPS4_DLSS_FG: NVIDIA DLSS Frame Generation (NGX DLSS-G) next to the Streamline DLSS 4.5
// upscaler. It runs through the optional tlg_dlss.dll bridge (MIT) and NVIDIA's own
// nvngx_dlss.dll / nvngx_dlssg.dll, all loaded at run time from the dlssg\ folder next to the
// EXE; no NVIDIA code is compiled in. Without the folder, on other GPUs or with SHADPS4_DLSS_FG=0
// nothing changes.
//
//   SHADPS4_DLSS_FG=1 / 2 / 3: generated frames per rendered frame (2x / 3x / 4x; capped at what
//                             the GPU supports). Default 1 when the dlssg folder is complete.
//   Shift+Home: frame generation on / off live.
//
// Inputs (Second Son): the DLSS resolve's depth (0x26b200000, reversed Z) and motion vectors
// (0x3059b0000, UV, current minus previous) are copied at the resolve; the camera's clip to
// previous clip matrix comes from the game's camera motion pass constants (fs_2bb2320e, 12
// floats). Frame generation runs on frames DLSS upscaled (gameplay); menus and videos show the
// rendered frames only.

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <vector>

#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"

struct VmaAllocator_T;

namespace Vulkan::DlssFg {

constexpr u64 ReprojectionPsHash = 0x2bb2320e;
constexpr u32 MaxGenerated = 3;

/// Generated frames wanted per rendered frame (0 = off: no files or SHADPS4_DLSS_FG=0).
int Wanted();
/// Shift+Home.
inline std::atomic<bool> g_user_on{true};

// ---- the camera's reprojection (parsing threads) ----
inline std::atomic<u64> g_reproj_ps{0};
inline std::atomic<u32> g_reproj_ud{~0u};
inline std::atomic<Shader::PFN_SrtWalker> g_reproj_walker{nullptr};
inline std::atomic<u32> g_reproj_flat_dw{0};
inline std::mutex g_reproj_mutex;
inline std::array<float, 12> g_reproj{};
inline u64 g_reproj_count = 0;

inline void NoteInfo(const Shader::Info& info, VAddr base) {
    if (info.pgm_hash != ReprojectionPsHash || info.stage != Shader::Stage::Fragment) [[likely]] {
        return;
    }
    if (Wanted() == 0 || g_reproj_ps.load(std::memory_order_relaxed) == base) {
        return;
    }
    for (const auto& b : info.buffers) {
        if (b.IsSpecial() || b.inline_cbuf) {
            continue;
        }
        if (b.sharp_idx + 4 <= 16) {
            g_reproj_walker.store(nullptr, std::memory_order_relaxed);
        } else if (info.srt_info.walker_func != nullptr &&
                   b.sharp_idx + 4 <= info.srt_info.flattened_bufsize_dw &&
                   info.srt_info.flattened_bufsize_dw <= 1024) {
            g_reproj_flat_dw.store(info.srt_info.flattened_bufsize_dw, std::memory_order_relaxed);
            g_reproj_walker.store(info.srt_info.walker_func, std::memory_order_relaxed);
        } else {
            continue;
        }
        g_reproj_ud.store(b.sharp_idx, std::memory_order_relaxed);
        g_reproj_ps.store(base, std::memory_order_release);
        return; // the first buffer is the constants
    }
}

inline void NoteDraw(const AmdGpu::Regs& regs) {
    const u64 p = g_reproj_ps.load(std::memory_order_relaxed);
    if (p == 0 || regs.ps_program.Address<VAddr>() != p) [[likely]] {
        return;
    }
    const u32 ud = g_reproj_ud.load(std::memory_order_relaxed);
    AmdGpu::Buffer vsharp;
    if (ud + 4 <= 16) {
        std::memcpy(&vsharp, &regs.ps_program.user_data[ud], sizeof(vsharp));
    } else {
        const auto walker = g_reproj_walker.load(std::memory_order_relaxed);
        const u32 flat_dw = g_reproj_flat_dw.load(std::memory_order_relaxed);
        if (walker == nullptr || ud + 4 > flat_dw) {
            return;
        }
        thread_local std::vector<u32> flat;
        flat.assign(flat_dw, 0u);
        std::memcpy(flat.data(), regs.ps_program.user_data.data(), 16 * sizeof(u32));
        walker(regs.ps_program.user_data.data(), flat.data());
        std::memcpy(&vsharp, &flat[ud], sizeof(vsharp));
    }
    if (vsharp.base_address == 0 || vsharp.GetSize() < 48) {
        return;
    }
    std::array<float, 12> rows;
    std::memcpy(rows.data(), reinterpret_cast<const void*>(vsharp.base_address), sizeof(rows));
    std::scoped_lock lock{g_reproj_mutex};
    g_reproj = rows;
    ++g_reproj_count;
}

// ---- device (vk_platform / vk_instance) ----
void AddInstanceExtensions(std::vector<const char*>& extensions);
/// The device extensions frame generation needs on this GPU (empty: it stays off).
const std::vector<const char*>& DeviceExtensions(vk::Instance instance,
                                                 vk::PhysicalDevice physical);
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
void Shutdown();
/// The instance's allocator (the snapshot images).
void SetAllocator(VmaAllocator_T* allocator);
/// The bridge is up and the GPU can generate frames.
bool Ready();
/// Generated frames this frame (min of the wish and the GPU), 0 = none.
u32 Count();

// ---- per frame ----
/// DLSS resolve: copies depth (depth aspect) and motion, both in TransferSrcOptimal (guest
/// layout), into the frame generation's own images.
void Snapshot(vk::CommandBuffer cmdbuf, vk::Image depth, vk::Format depth_format,
              vk::ImageLayout depth_layout, vk::Image motion, vk::Format motion_format,
              vk::ImageLayout motion_layout, u32 width, u32 height);
/// The snapshot of this frame exists (taken since the last Consume) - the presenter's test.
bool TakeSnapshot();

struct Output {
    vk::Image image;
    vk::ImageView view;
};
/// The presenter's frame, General layout. False: nothing written (the caller shows the frame).
bool Record(vk::CommandBuffer cmdbuf, vk::Device device, vk::Image backbuffer,
            vk::ImageView backbuffer_view, vk::Format format, u32 width, u32 height,
            const Output* outputs, u32 count);
/// A feature exists for another size / format: the caller waits for the GPU work that used it
/// and calls ReleaseFeature before Record.
bool NeedsRecreate(u32 width, u32 height, vk::Format format);
void ReleaseFeature();
/// The next frame starts a new sequence (a cut, a pause).
void Reset();

} // namespace Vulkan::DlssFg
