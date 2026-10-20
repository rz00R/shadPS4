// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include "common/types.h"

// Second Son's post effects the player can switch off (F3 -> Graphics). Each one ends in a single
// alpha-blended full-screen draw into the HDR scene (frame records of both performance modes):
//   fs_f47f183e  depth of field: the half-size blurred scene blended in by the linearized depth;
//   fs_5acb9dad  motion blur: the half-size reconstruction from 48x27 velocity tiles blended in;
//   fs_a291d5b2  lens glare: a lens dirt texture times the 240x135 bloom, added (ONE/ONE).
// Off: ClassifyDraw skips that draw, so the scene stays as it was before it.
namespace Vulkan::Effects {

constexpr u64 DepthOfFieldPs = 0xf47f183e;
constexpr u64 MotionBlurPs = 0x5acb9dad;
constexpr u64 LensFlarePs = 0xa291d5b2;

/// Guest code addresses of the three programs once a pipeline lookup has met them (0 = not yet).
inline std::atomic<u64> g_dof_ps{0}, g_motion_blur_ps{0}, g_lens_flare_ps{0};
/// The player's choices (EmulatorSettings, applied by the F3 dialog and at start-up).
inline std::atomic<bool> g_dof_on{true}, g_motion_blur_on{true}, g_lens_flare_on{true};

/// PipelineCache::GetProgram: remembers where the three programs live.
inline void NoteProgram(bool fragment, u64 hash, u64 base) {
    if (!fragment) [[likely]] {
        return;
    }
    switch (hash) {
    case DepthOfFieldPs:
        g_dof_ps.store(base, std::memory_order_relaxed);
        break;
    case MotionBlurPs:
        g_motion_blur_ps.store(base, std::memory_order_relaxed);
        break;
    case LensFlarePs:
        g_lens_flare_ps.store(base, std::memory_order_relaxed);
        break;
    default:
        break;
    }
}

/// ClassifyDraw: whether this draw is a switched-off effect (three relaxed loads + compares).
inline bool Skip(u64 ps_address) {
    return ps_address != 0 &&
           ((!g_dof_on.load(std::memory_order_relaxed) &&
             ps_address == g_dof_ps.load(std::memory_order_relaxed)) ||
            (!g_motion_blur_on.load(std::memory_order_relaxed) &&
             ps_address == g_motion_blur_ps.load(std::memory_order_relaxed)) ||
            (!g_lens_flare_on.load(std::memory_order_relaxed) &&
             ps_address == g_lens_flare_ps.load(std::memory_order_relaxed)));
}

inline void Apply(bool depth_of_field, bool motion_blur, bool lens_flare) {
    g_dof_on.store(depth_of_field, std::memory_order_relaxed);
    g_motion_blur_on.store(motion_blur, std::memory_order_relaxed);
    g_lens_flare_on.store(lens_flare, std::memory_order_relaxed);
}

} // namespace Vulkan::Effects
