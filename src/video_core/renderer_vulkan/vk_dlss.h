// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include "common/logging/log.h"
#include "common/types.h"
#include "shader_recompiler/info.h"
#include "video_core/amdgpu/regs.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_dlss_fg.h"

// SHADPS4_DLSS (2026-10-17, LATER-PLAN 2; plan: notes/study-dlss-plan-2026-10-14.md, build D1;
// Streamline facts: notes/dlss-streamline-checklist-2026-10-17.md).
// Second Son's SMAA T2x temporal resolve (fs_a6d23116) reads exactly DLSS's inputs and writes the
// anti-aliased frame the HUD is drawn onto; that one draw per frame becomes a back-side action
// (Rasterizer::DrawClass::DlssResolve), so the worker paths carry it like a resolve or a depth copy
// and nothing is added per draw beyond one address compare in ClassifyDraw.
//   =1: DLAA through NVIDIA Streamline. The player's own signed DLLs are loaded from dlss\ next to
//       the EXE (sl.interposer, sl.common, sl.dlss, nvngx_dlss); only MIT headers are compiled in.
//       Missing / unsigned DLLs, a GPU without DLSS or any failure: the draw stays the game's own.
//   =2: diagnostic - the action copies the un-anti-aliased tonemap output (the game without T2x).
// Live keys: Home = DLSS on / off (instant A/B with the game's AA), Page Up = jitter sign combination.
namespace Vulkan::Dlss {

constexpr u64 ResolvePsHash = 0xa6d23116; ///< T2x temporal resolve (writes 0x26bd70000)
// Measured in every capture since 09-24 (study sec. 1.1-1.2): fixed guest addresses.
constexpr VAddr ColorInAddr = 0x2f9e10000; ///< tonemap output, RGBA8 1920x1080, not anti-aliased
constexpr VAddr MotionAddr = 0x3059b0000;  ///< RG16F motion vectors (UV, current minus previous)
constexpr VAddr DepthAddr = 0x26b200000;   ///< D32S8, reversed Z

inline int Mode() {
    static const int mode = [] {
        const char* v = std::getenv("SHADPS4_DLSS");
        return v && (v[0] == '1' || v[0] == '2') ? v[0] - '0' : 0;
    }();
    return mode;
}
inline bool Enabled() {
    return Mode() != 0;
}

/// Streamline is up and the feature exists on this device (=1), or the diagnostic copy (=2).
inline std::atomic<bool> g_ready{false};
/// Home key: the player's on / off (the game's own T2x runs while off).
inline std::atomic<bool> g_user_on{true};
/// Page Up: jitter sign combination (bit 0 flips x, bit 1 flips y).
inline std::atomic<u32> g_jitter_combo{3}; ///< D (-x -y): the user's live pick, 2026-10-17
/// Insert (SHADPS4_DLSS_UPSCALE): 1 = DLSS writes the window size, 1.5 = 1.5x the window (3840x2160
/// on a 1440p screen), scaled down by the post pass (supersampling; user idea 2026-10-17).
inline std::atomic<float> g_output_scale{1.0f};

/// Guest code address of fs_a6d23116 once a pipeline lookup has met it (0 = not yet / off).
inline std::atomic<u64> g_resolve_ps{0};

/// PipelineCache::GetProgram: remembers where the resolve shader lives.
inline void NoteProgram(bool fragment, u64 hash, VAddr base) {
    if (fragment && hash == ResolvePsHash && Enabled()) [[unlikely]] {
        g_resolve_ps.store(base, std::memory_order_relaxed);
    }
}

/// ClassifyDraw: is this draw's pixel shader the resolve, and is DLSS to replace it now?
inline bool IsResolve(VAddr ps_address) {
    const u64 a = g_resolve_ps.load(std::memory_order_relaxed);
    return a != 0 && a == ps_address && g_ready.load(std::memory_order_relaxed) &&
           g_user_on.load(std::memory_order_relaxed);
}

// The frame's jitter phase (study sec. 1.4): the game alternates two quarter-pixel offsets and
// writes the matching T2x subsample index (1 or 2; 0 = T2x off) into the SMAA blend-weights
// draw's 80-byte CB, float 12. That draw runs two draws before the resolve in the same command
// buffer, so the thread parsing it keeps the value and the resolve's record carries it to the back
// (Rasterizer::SplitDraw::dlss_phase): exact per frame, no guess from a frame counter that a
// loading screen could flip.
constexpr u64 WeightsPsHash = 0x961c93ba; ///< SMAA blend weights
inline std::atomic<u64> g_weights_ps{0};
inline std::atomic<u32> g_weights_ud{~0u}; ///< dword of its CB's V# in the flattened user data
/// When the V# is not in the 16 user-data registers: the shader's SRT walker (it reads the guest
/// pointer tables into the flattened buffer, as binding does) and that buffer's size.
inline std::atomic<Shader::PFN_SrtWalker> g_weights_walker{nullptr};
inline std::atomic<u32> g_weights_flat_dw{0};
/// The parsing thread's last weights draw: 1 / 2 = subsample index, 3 = T2x off, 0 = unknown.
inline thread_local u8 t_phase = 0;

/// Where the weights program's Info is known: remember its address and its CB's user-data slot.
inline void NoteWeightsInfo(const Shader::Info& info, VAddr base) {
    DlssFg::NoteInfo(info, base); // SHADPS4_DLSS_FG: the camera reprojection's CB
    if (info.pgm_hash != WeightsPsHash || info.stage != Shader::Stage::Fragment || !Enabled())
        [[likely]] {
        return;
    }
    if (g_weights_ps.load(std::memory_order_relaxed) == base) {
        return;
    }
    for (const auto& b : info.buffers) {
        if (b.IsSpecial() || b.inline_cbuf) {
            continue;
        }
        if (b.sharp_idx + 4 <= 16) {
            g_weights_walker.store(nullptr, std::memory_order_relaxed);
        } else if (info.srt_info.walker_func != nullptr &&
                   b.sharp_idx + 4 <= info.srt_info.flattened_bufsize_dw &&
                   info.srt_info.flattened_bufsize_dw <= 1024) {
            // Play log 2026-10-17: the phase stayed "unknown" on every frame - the V# comes
            // through a pointer table, not the 16 registers.
            g_weights_flat_dw.store(info.srt_info.flattened_bufsize_dw, std::memory_order_relaxed);
            g_weights_walker.store(info.srt_info.walker_func, std::memory_order_relaxed);
        } else {
            continue;
        }
        g_weights_ud.store(b.sharp_idx, std::memory_order_relaxed);
        g_weights_ps.store(base, std::memory_order_release);
        LOG_INFO(Render_Vulkan, "DLSS: jitter source fs_961c93ba at {:#x}, CB V# at dword {} ({})",
                 base, b.sharp_idx,
                 b.sharp_idx + 4 <= 16 ? "user data" : "pointer table");
        return;
    }
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        LOG_WARNING(Render_Vulkan, "DLSS: fs_961c93ba has no readable CB ({} buffers) - jitter "
                                   "phase unknown",
                    info.buffers.size());
    }
}

/// Next to every ClassifyDraw call that parses the stream in order (front, lean workers, inline).
inline void NoteDraw(const AmdGpu::Regs& regs) {
    DlssFg::NoteDraw(regs);
    const u64 w = g_weights_ps.load(std::memory_order_relaxed);
    if (w == 0 || regs.ps_program.Address<VAddr>() != w) [[likely]] {
        return;
    }
    const u32 ud = g_weights_ud.load(std::memory_order_relaxed);
    AmdGpu::Buffer vsharp;
    if (ud + 4 <= 16) {
        std::memcpy(&vsharp, &regs.ps_program.user_data[ud], sizeof(vsharp));
    } else {
        const auto walker = g_weights_walker.load(std::memory_order_relaxed);
        const u32 flat_dw = g_weights_flat_dw.load(std::memory_order_relaxed);
        if (walker == nullptr || ud + 4 > flat_dw) {
            t_phase = 0;
            return;
        }
        thread_local std::vector<u32> flat;
        flat.assign(flat_dw, 0u);
        std::memcpy(flat.data(), regs.ps_program.user_data.data(), 16 * sizeof(u32));
        walker(regs.ps_program.user_data.data(), flat.data());
        std::memcpy(&vsharp, &flat[ud], sizeof(vsharp));
    }
    if (vsharp.base_address == 0 || vsharp.GetSize() < 52) {
        t_phase = 0;
        return;
    }
    float f;
    std::memcpy(&f, reinterpret_cast<const void*>(vsharp.base_address + 48), sizeof(f));
    t_phase = (f > 0.5f && f < 1.5f)   ? 1
              : (f > 1.5f && f < 2.5f) ? 2
              : (f > -0.5f && f < 0.5f) ? 3
                                        : 0;
}

// ---- Halton jitter (SHADPS4_DLSS_HALTON, 2026-10-20) ----
// The game's own jitter is two diagonal quarter-pixel positions: too few for DLSS to rebuild detail,
// above all from 1080p to 2160p. Ours adds a sub-pixel shift to the viewport of every pass drawn
// with the scene depth (DepthAddr at 1920x1080), chosen so that game + shift walks a Halton (2, 3)
// sequence of 8 * (output / render)^2 phases (8 for DLAA, 15 for 1440p, 32 for 2160p), as the
// iss_pc temporal upscaler did. The back runs the stream in order (MTB is off whenever the native
// paths are on), so one back-side value is "this frame's": the resolve of frame N-1 picks the
// shift frame N draws with, and the resolve of frame N passes DLSS the jitter that was really
// applied (the game's phase it read + our shift), so a wrong guess of the game's next phase costs
// distribution, never a trembling picture. Motion vectors stay unjittered (the camera pass
// reprojects from depth), as before.
//   SHADPS4_DLSS_HALTON=0: off (the game's two positions only, the v1.0.2 behaviour); default on.
//   SHADPS4_DLSS_HALTON_SIGNS=0..3: our shift's sign versus DLSS's convention (bit 0 x, bit 1 y).
// Live keys: Shift+Page Down = on / off (A/B), Shift+Insert = cycle the signs.
inline std::atomic<bool> g_halton_on{[] {
    const char* v = std::getenv("SHADPS4_DLSS_HALTON");
    return !(v && v[0] == '0');
}()};
inline std::atomic<u32> g_halton_signs{[] {
    const char* v = std::getenv("SHADPS4_DLSS_HALTON_SIGNS");
    return v ? static_cast<u32>(std::atoi(v)) & 3u : 0u;
}()};
/// A nonzero shift is in effect: the A2 translator leaves scene-depth draws to the back (its
/// workers compute viewports, even into pre-recorded secondaries, where the shift cannot go).
inline std::atomic<bool> g_scene_jitter{false};

inline float Halton(u32 index, u32 base) {
    float f = 1.0f, result = 0.0f;
    for (u32 i = index; i > 0; i /= base) {
        f /= static_cast<float>(base);
        result += f * static_cast<float>(i % base);
    }
    return result;
}
/// 8 * (output / render)^2 between 8 and 64 (DLSS programming guide; iss_pc JitterPhases).
inline u32 JitterPhases(u32 render_width, u32 output_width) {
    if (render_width == 0 || output_width <= render_width) {
        return 8;
    }
    const float r = static_cast<float>(output_width) / static_cast<float>(render_width);
    const u32 n = static_cast<u32>(8.0f * r * r + 0.999f);
    return n < 8 ? 8 : n > 64 ? 64 : n;
}
/// The scene depth this jitter belongs to (the same address also holds the 768x768 minimap depth).
inline bool IsSceneDepth(VAddr address, u32 width, u32 height) {
    return address == DepthAddr && width == 1920 && height == 1080;
}

/// Back thread only (Rasterizer::BeginRendering / RenderStateMemoBegin / DlssResolve).
struct HaltonState {
    u32 index{};    ///< 1..phases, the frame's Halton sample
    u32 phases{8};
    float shift_x{}, shift_y{}; ///< the viewport shift of this frame's scene passes, pixels
    float dlss_x{}, dlss_y{};   ///< the same shift in DLSS's convention (signs applied)
    u64 frames{};               ///< resolves that applied a shift (sampled log)
};
inline HaltonState g_halton{};

/// The viewport shift for a pass, by its depth attachment (none: 0).
inline std::pair<float, float> SceneShift(bool scene_depth) {
    if (!scene_depth || !g_halton_on.load(std::memory_order_relaxed) ||
        !g_user_on.load(std::memory_order_relaxed) || !g_ready.load(std::memory_order_relaxed)) {
        return {0.0f, 0.0f};
    }
    return {g_halton.shift_x, g_halton.shift_y};
}

/// DlssResolve, before Evaluate: the jitter DLSS gets for this frame (`game_x/y` = the game's
/// own offset from the phase), then the shift of the next frame. `reset` is raised when the
/// sequence restarts (on / off, signs, phase count).
inline std::pair<float, float> HaltonResolve(float game_x, float game_y, u8 game_phase,
                                             u32 render_width, u32 output_width, bool& reset) {
    auto& h = g_halton;
    const bool on = g_halton_on.load(std::memory_order_relaxed);
    const u32 signs = g_halton_signs.load(std::memory_order_relaxed);
    static bool last_on = on;
    static u32 last_signs = signs;
    const u32 phases = JitterPhases(render_width, output_width);
    if (on != last_on || signs != last_signs || phases != h.phases) {
        reset = true;
        h.index = 0;
        h.phases = phases;
        last_on = on;
        last_signs = signs;
        // This frame's scene was drawn with the old shift; DLSS restarts its history anyway.
    }
    // What this frame's geometry really had: the game's offset plus our shift.
    const float jx = game_x + h.dlss_x;
    const float jy = game_y + h.dlss_y;
    if (!on) {
        h.shift_x = h.shift_y = h.dlss_x = h.dlss_y = 0.0f;
        g_scene_jitter.store(false, std::memory_order_relaxed);
        return {jx, jy};
    }
    // Next frame: the game alternates its two positions; aim game + shift at the Halton sample.
    const float next_gx = game_phase == 1 || game_phase == 2 ? -game_x : 0.0f;
    const float next_gy = game_phase == 1 || game_phase == 2 ? -game_y : 0.0f;
    h.index = h.index % h.phases + 1;
    const float tx = Halton(h.index, 2) - 0.5f;
    const float ty = Halton(h.index, 3) - 0.5f;
    // Under half a pixel: a full-screen quad shifted by less keeps covering every pixel centre.
    const auto clamp = [](float v) { return v < -0.49f ? -0.49f : v > 0.49f ? 0.49f : v; };
    h.dlss_x = clamp(tx - next_gx);
    h.dlss_y = clamp(ty - next_gy);
    h.shift_x = (signs & 1) ? -h.dlss_x : h.dlss_x;
    h.shift_y = (signs & 2) ? -h.dlss_y : h.dlss_y;
    g_scene_jitter.store(true, std::memory_order_relaxed);
    if ((++h.frames & 1023) == 1) {
        LOG_INFO(Render_Vulkan,
                 "DLSS: Halton frame {} sample {}/{} game phase {} -> DLSS jitter ({:.3f}, {:.3f}); "
                 "next shift ({:.3f}, {:.3f}) signs {}",
                 h.frames, h.index, h.phases, game_phase, jx, jy, h.shift_x, h.shift_y, signs);
    }
    return {jx, jy};
}

// ---- Streamline (vk_dlss.cpp; Windows only, stubs elsewhere) ----

/// vk_platform GetInstanceExtensions: loads + verifies the interposer, slInit, and appends the
/// instance extensions DLSS needs. Before vkCreateInstance (Streamline's required order).
void AddInstanceExtensions(std::vector<const char*>& extensions);
/// vk_instance, device selection: slIsFeatureSupported for this GPU; false = DLSS stays off.
bool DeviceSupported(vk::PhysicalDevice physical_device);
/// The device extensions DLSS asked for (valid until Shutdown).
const std::vector<const char*>& DeviceExtensions();
/// vk_instance, after the device and its queues exist: slSetVulkanInfo, present proxy, DLSS
/// functions; sets g_ready on success.
void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device,
                   u32 graphics_family, u32 graphics_index);
/// vk_swapchain: the present to use (Streamline's proxy runs its per-frame bookkeeping and
/// forwards); null = present as usual.
PFN_vkQueuePresentKHR PresentFunction();
/// Instance destruction (GPU idle): slShutdown before the device goes.
void Shutdown();

struct EvalImage {
    vk::Image image;
    vk::ImageView view;
    vk::Format format;
    vk::ImageLayout layout;
    vk::ImageUsageFlags usage;
    u32 width;
    u32 height;
    bool depth;
};
/// One DLAA evaluate into `cmdbuf` (outside rendering; the images already in `layout`, with
/// their producer barriers). jitter in pixels. False = not run (the caller falls back).
bool Evaluate(vk::CommandBuffer cmdbuf, u64 frame, const EvalImage& color, const EvalImage& motion,
              const EvalImage& depth, const EvalImage& output, float jitter_x, float jitter_y,
              bool reset);

} // namespace Vulkan::Dlss
