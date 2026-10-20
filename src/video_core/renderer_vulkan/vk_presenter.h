// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>

#include "core/libraries/videoout/buffer.h"
#include "imgui/imgui_texture.h"
#include "video_core/renderer_vulkan/host_passes/fsr_pass.h"
#include "video_core/renderer_vulkan/host_passes/pp_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Frontend {
class WindowSDL;
}

namespace AmdGpu {
struct Liverpool;
}

namespace Vulkan {

struct Frame {
    u32 width;
    u32 height;
    VmaAllocation allocation;
    vk::Image image;
    vk::ImageView image_view;
    vk::Fence present_done;
    vk::Semaphore ready_semaphore;
    u64 ready_tick;
    bool is_hdr{false};
    u8 id{};

    ImTextureID imgui_texture;

    // SHADPS4_DLSS_FG (vk_dlss_fg.h): the frames generated before this one, shown first.
    struct Generated {
        VmaAllocation allocation{};
        vk::Image image{};
        vk::ImageView view{};
        ImTextureID imgui_texture{};
    };
    std::array<Generated, 3> generated{};
    u32 generated_count{};
    u32 generated_width{}, generated_height{};
    vk::Format generated_format{};
};

enum SchedulerType {
    Draw,
    Present,
    CpuFlip,
};

class Rasterizer;

class Presenter {
public:
    Presenter(Frontend::WindowSDL& window, AmdGpu::Liverpool* liverpool);
    ~Presenter();

    HostPasses::PostProcessingPass::Settings& GetPPSettingsRef() {
        return pp_settings;
    }

    HostPasses::FsrPass::Settings& GetFsrSettingsRef() {
        return fsr_settings;
    }

    Frontend::WindowSDL& GetWindow() const {
        return window;
    }

    Rasterizer& GetRasterizer() const {
        return *rasterizer.get();
    }

    bool IsHDRSupported() const {
        return swapchain.HasHDR();
    }

    void SetHDR(bool enable) {
        if (!IsHDRSupported()) {
            return;
        }
        swapchain.SetHDR(enable);
        pp_settings.hdr = enable ? 1 : 0;
    }

    VideoCore::Image& RegisterVideoOutSurface(
        const Libraries::VideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
        // Registering the same address twice would grow this vector without bound; it is only
        // ever linear-scanned by IsVideoOutSurface and nothing removes entries from it.
        if (std::ranges::find(vo_buffers_addr, cpu_address) == vo_buffers_addr.cend()) {
            vo_buffers_addr.emplace_back(cpu_address);
        }
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        const auto image_id = texture_cache.FindImage(desc);
        auto& image = texture_cache.GetImage(image_id);
        image.usage.vo_surface = 1u;
        return image;
    }

    bool IsVideoOutSurface(const AmdGpu::ColorBuffer& color_buffer) const;

    Frame* PrepareFrame(const Libraries::VideoOut::BufferAttributeGroup& attribute,
                        VAddr cpu_address);

    Frame* PrepareBlankFrame(bool present_thread);

    void Present(Frame* frame, bool is_reusing_frame = false);
    Frame* PrepareLastFrame();

    /// SHADPS4_BOOT_SCREEN: one overlay-only frame during WarmUp (inside this constructor, main
    /// thread); uses only the members built before the rasterizer. Waits for it on the GPU.
    void DrawBootScreen(std::string_view headline, std::string_view detail, float fraction,
                        bool one_time);

private:
    Frame* GetRenderFrame();

    void RecreateFrame(Frame* frame, u32 width, u32 height);

    void SetExpectedGameSize(s32 width, s32 height);

    /// SHADPS4_DLSS_FG: one swapchain present; `generated` >= 0 shows that generated frame
    /// (no fence, no flip count), -1 the frame itself.
    void PresentImpl(Frame* frame, bool is_reusing_frame, int generated);
    void EnsureGeneratedImages(Frame* frame, u32 count);
    void DestroyGeneratedImages(Frame* frame);
    /// PrepareFrame, after the post pass: the generated frames into `frame` (or none).
    void RecordFrameGeneration(vk::CommandBuffer cmdbuf, Frame* frame, bool upscaled);
    /// PrepareFrame, before the command buffer: frame generation runs this flip.
    bool FrameGenerationWanted(Frame* frame, bool upscaled);

private:
    u64 fg_last_tick{};
    std::chrono::steady_clock::time_point fg_last_present{};
    double fg_interval_ms{16.7};

    float expected_ratio{1920.0 / 1080.0f};
    u32 expected_frame_width{1920};
    u32 expected_frame_height{1080};

    Frontend::WindowSDL& window;
    Instance instance;
    HostPasses::FsrPass fsr_pass;
    HostPasses::FsrPass::Settings fsr_settings{};
    HostPasses::PostProcessingPass::Settings pp_settings{};
    HostPasses::PostProcessingPass pp_pass;
    AmdGpu::Liverpool* liverpool;
    Scheduler draw_scheduler;
    Scheduler present_scheduler;
    Scheduler flip_scheduler;
    Swapchain swapchain;
    std::unique_ptr<Rasterizer> rasterizer;
    VideoCore::TextureCache& texture_cache;
    vk::UniqueCommandPool command_pool;
    std::vector<Frame> present_frames;
    std::queue<Frame*> free_queue;
    Frame* last_submit_frame;
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::condition_variable_any frame_cv;
    std::optional<ImGui::RefCountedTexture> splash_img;
    std::vector<VAddr> vo_buffers_addr;
};

} // namespace Vulkan
