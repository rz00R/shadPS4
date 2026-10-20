// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>

#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_dlss_fg.h"

#include <vk_mem_alloc.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "video_core/renderer_vulkan/tlg_dlss_bridge.h"
#endif

namespace Vulkan::DlssFg {

#ifdef _WIN32

namespace {

std::filesystem::path Folder() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path() / "dlssg";
}

bool FilesPresent() {
    const auto dir = Folder();
    return std::filesystem::is_regular_file(dir / "tlg_dlss.dll") &&
           std::filesystem::is_regular_file(dir / "nvngx_dlss.dll") &&
           std::filesystem::is_regular_file(dir / "nvngx_dlssg.dll");
}

void BridgeLog(int warning, const char* message) {
    if (warning) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: {}", message);
    } else {
        LOG_INFO(Render_Vulkan, "DLSS FG: {}", message);
    }
}

struct OwnedImage {
    VkImage image{};
    VmaAllocation allocation{};
    vk::ImageView view{};
};

struct State {
    HMODULE module{};
    const TlgDlssApi* api{};
    bool loaded{}, instance_ok{}, device_ok{}, ready{};
    int max_frames{};
    std::vector<const char*> device_ext_names;
    std::vector<std::string> device_ext_storage;
    std::vector<std::string> instance_ext_storage;
    vk::Device device{};
    VmaAllocator allocator{};
    // Snapshots (render size).
    std::mutex mutex;
    OwnedImage depth{}, motion{};
    vk::Format depth_format{};
    u32 snap_width{}, snap_height{};
    bool snap_initialized{};
    bool snap_fresh{};
    std::vector<OwnedImage> graveyard;
    // Feature.
    std::optional<TlgDlssFgFeature> feature;
    bool reset{true};
    bool failed_logged{};
    u64 last_reproj{};
} g;

int ComputeWanted() {
    const char* v = std::getenv("SHADPS4_DLSS_FG");
    if (v && v[0] == '0') {
        return 0;
    }
    if (!FilesPresent()) {
        if (v && v[0] >= '1' && v[0] <= '3') {
            LOG_WARNING(Render_Vulkan,
                        "DLSS FG: SHADPS4_DLSS_FG is set but the dlssg folder is incomplete "
                        "(tlg_dlss.dll, nvngx_dlss.dll, nvngx_dlssg.dll)");
        }
        return 0;
    }
    if (v && v[0] >= '1' && v[0] <= '3') {
        return v[0] - '0';
    }
    return 1;
}

bool Load() {
    if (g.loaded) {
        return g.api != nullptr;
    }
    g.loaded = true;
    if (Wanted() == 0) {
        return false;
    }
    const auto dir = Folder();
    // nvngx_dlssg.dll and the bridge's own imports are found next to it.
    g.module = LoadLibraryExW((dir / "tlg_dlss.dll").c_str(), nullptr,
                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    const auto get_api =
        g.module ? reinterpret_cast<TlgDlssGetApiFn>(GetProcAddress(g.module, "TlgDlssGetApi"))
                 : nullptr;
    g.api = get_api ? get_api() : nullptr;
    if (!g.api || g.api->abi != TLG_DLSS_BRIDGE_ABI) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: dlssg\\tlg_dlss.dll missing or another version - off");
        g.api = nullptr;
        return false;
    }
    std::error_code ec;
    const auto data = dir / "data";
    std::filesystem::create_directories(data, ec);
    if (!g.api->Configure(dir.c_str(), data.c_str(), BridgeLog)) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: bridge configuration failed - off");
        g.api = nullptr;
        return false;
    }
    return true;
}

// 4x4 helpers (row-major, column vectors) for the reprojection.
using Mat4 = std::array<double, 16>;
Mat4 Mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) r[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
    return r;
}
bool Invert(const Mat4& m, Mat4& out) {
    std::array<double, 32> a{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) a[i * 8 + j] = m[i * 4 + j];
        a[i * 8 + 4 + i] = 1.0;
    }
    for (int c = 0; c < 4; ++c) {
        int p = c;
        for (int r = c + 1; r < 4; ++r)
            if (std::abs(a[r * 8 + c]) > std::abs(a[p * 8 + c])) p = r;
        if (std::abs(a[p * 8 + c]) < 1e-12) return false;
        if (p != c)
            for (int j = 0; j < 8; ++j) std::swap(a[c * 8 + j], a[p * 8 + j]);
        const double d = a[c * 8 + c];
        for (int j = 0; j < 8; ++j) a[c * 8 + j] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == c) continue;
            const double f = a[r * 8 + c];
            for (int j = 0; j < 8; ++j) a[r * 8 + j] -= f * a[c * 8 + j];
        }
    }
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = a[i * 8 + 4 + j];
    return true;
}
void StoreRowVector(const Mat4& m, float* out) {
    // v' = v * M with v a row vector: the transpose of the column-vector matrix.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = float(m[j * 4 + i]);
}

/// Clip (x, y up, z, w) to the previous frame's clip, from the game's current (u, v, depth, 1)
/// -> previous (u, v, w) reprojection rows (UV y down).
bool ClipToPrevClip(const std::array<float, 12>& r, float* to_prev, float* from_prev) {
    const Mat4 uv_from_clip{0.5, 0, 0, 0.5, 0, -0.5, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 1};
    Mat4 prev_from_uv{};
    for (int j = 0; j < 4; ++j) {
        prev_from_uv[0 * 4 + j] = 2.0 * r[0 * 4 + j] - r[2 * 4 + j];
        prev_from_uv[1 * 4 + j] = r[2 * 4 + j] - 2.0 * r[1 * 4 + j];
        prev_from_uv[3 * 4 + j] = r[2 * 4 + j];
    }
    prev_from_uv[2 * 4 + 2] = 1.0;
    const Mat4 forward = Mul(prev_from_uv, uv_from_clip);
    Mat4 inverse{};
    if (!Invert(forward, inverse)) return false;
    StoreRowVector(forward, to_prev);
    StoreRowVector(inverse, from_prev);
    return true;
}

void DestroyImage(OwnedImage& img) {
    if (img.view) g.device.destroyImageView(img.view);
    if (img.image) vmaDestroyImage(g.allocator, img.image, img.allocation);
    img = {};
}

bool CreateImage(OwnedImage& img, vk::Format format, u32 w, u32 h, vk::ImageUsageFlags usage,
                 vk::ImageAspectFlags view_aspect) {
    const VkImageCreateInfo info = static_cast<VkImageCreateInfo>(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = usage,
    });
    const VmaAllocationCreateInfo alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(g.allocator, &info, &alloc, &img.image, &img.allocation, nullptr) !=
        VK_SUCCESS) {
        img = {};
        return false;
    }
    auto [r, view] = g.device.createImageView(vk::ImageViewCreateInfo{
        .image = vk::Image{img.image},
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = {view_aspect, 0, 1, 0, 1},
    });
    if (r != vk::Result::eSuccess) {
        DestroyImage(img);
        return false;
    }
    img.view = view;
    return true;
}

TlgDlssImage Bridge(vk::Image image, vk::ImageView view, vk::Format format,
                    vk::ImageAspectFlags aspect, u32 w, u32 h) {
    const vk::ImageSubresourceRange range{aspect, 0, 1, 0, 1};
    return TlgDlssImage{static_cast<VkImage>(image), static_cast<VkImageView>(view),
                        static_cast<VkImageSubresourceRange>(range), static_cast<VkFormat>(format),
                        w, h};
}

} // namespace

int Wanted() {
    static const int wanted = ComputeWanted();
    return wanted;
}

void AddInstanceExtensions(std::vector<const char*>& extensions) {
    if (!Load()) {
        return;
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!g.api->InstanceExtensions(&count, &required)) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: instance extension query failed - off");
        return;
    }
    const auto [result, available] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        return;
    }
    g.instance_ext_storage.clear();
    for (u32 i = 0; i < count; ++i) {
        const bool found = std::ranges::any_of(available, [&](const auto& e) {
            return std::strcmp(e.extensionName.data(), required[i].extensionName) == 0;
        });
        if (!found) {
            LOG_WARNING(Render_Vulkan, "DLSS FG: missing instance extension {} - off",
                        required[i].extensionName);
            return;
        }
        g.instance_ext_storage.emplace_back(required[i].extensionName);
    }
    for (const auto& name : g.instance_ext_storage) {
        if (std::ranges::none_of(extensions, [&](const char* e) { return name == e; })) {
            extensions.push_back(name.c_str());
        }
    }
    g.instance_ok = true;
}

const std::vector<const char*>& DeviceExtensions(vk::Instance instance,
                                                 vk::PhysicalDevice physical) {
    g.device_ext_names.clear();
    if (!g.api || !g.instance_ok || g.device_ok) {
        return g.device_ext_names;
    }
    if (physical.getProperties().vendorID != 0x10de) {
        LOG_INFO(Render_Vulkan, "DLSS FG: not an NVIDIA GPU - off");
        return g.device_ext_names;
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!g.api->DeviceExtensions(static_cast<VkInstance>(instance),
                                 static_cast<VkPhysicalDevice>(physical), &count, &required)) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: this GPU / driver does not support DLSS - off");
        return g.device_ext_names;
    }
    g.device_ext_storage.clear();
    for (u32 i = 0; i < count; ++i) {
        g.device_ext_storage.emplace_back(required[i].extensionName);
    }
    for (const auto& s : g.device_ext_storage) {
        g.device_ext_names.push_back(s.c_str());
    }
    g.device_ok = true;
    return g.device_ext_names;
}

void DeviceCreated(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!g.api || !g.device_ok || g.ready) {
        return;
    }
    if (!g.api->Initialize(static_cast<VkInstance>(instance),
                           static_cast<VkPhysicalDevice>(physical), static_cast<VkDevice>(device),
                           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                           VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr)) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: NGX initialization failed - off");
        return;
    }
    g.max_frames = g.api->FgMaxFrames();
    if (g.max_frames <= 0) {
        LOG_WARNING(Render_Vulkan, "DLSS FG: frame generation is not available on this GPU - off");
        return;
    }
    g.device = device;
    g.ready = true;
    LOG_INFO(Render_Vulkan, "DLSS FG: ready ({} generated per frame wanted, GPU max {})", Wanted(),
             g.max_frames);
}

void Shutdown() {
    if (!g.api) {
        return;
    }
    {
        std::scoped_lock lock{g.mutex};
        if (g.allocator) {
            DestroyImage(g.depth);
            DestroyImage(g.motion);
            for (auto& img : g.graveyard) DestroyImage(img);
            g.graveyard.clear();
        }
    }
    g.feature.reset();
    if (g.ready) {
        g.api->Shutdown();
    }
    g.ready = false;
}

bool Ready() {
    return g.ready;
}

u32 Count() {
    if (!g.ready || !g_user_on.load(std::memory_order_relaxed)) {
        return 0;
    }
    return u32(std::clamp(std::min(Wanted(), g.max_frames), 0, int(MaxGenerated)));
}

void SetAllocator(VmaAllocator_T* allocator) {
    g.allocator = allocator;
}

void Snapshot(vk::CommandBuffer cmdbuf, vk::Image depth, vk::Format depth_format,
              vk::ImageLayout depth_layout, vk::Image motion, vk::Format motion_format,
              vk::ImageLayout motion_layout, u32 width, u32 height) {
    if (!g.ready || !g.allocator || Count() == 0) {
        return;
    }
    std::scoped_lock lock{g.mutex};
    if (g.snap_width != width || g.snap_height != height || g.depth_format != depth_format) {
        // The old images may be in use by submitted frames: kept until shutdown (a size change
        // is rare; no GPU wait from inside a recording).
        if (g.depth.image) g.graveyard.push_back(g.depth);
        if (g.motion.image) g.graveyard.push_back(g.motion);
        g.depth = {};
        g.motion = {};
        g.snap_initialized = false;
        g.snap_width = width;
        g.snap_height = height;
        g.depth_format = depth_format;
        if (!CreateImage(g.depth, depth_format, width, height,
                         vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
                             vk::ImageUsageFlagBits::eDepthStencilAttachment,
                         vk::ImageAspectFlagBits::eDepth) ||
            !CreateImage(g.motion, vk::Format::eR16G16Sfloat, width, height,
                         vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled |
                             vk::ImageUsageFlagBits::eStorage,
                         vk::ImageAspectFlagBits::eColor)) {
            LOG_WARNING(Render_Vulkan, "DLSS FG: snapshot images could not be created");
            DestroyImage(g.depth);
            DestroyImage(g.motion);
            g.snap_width = g.snap_height = 0;
            return;
        }
    }
    if (motion_format != vk::Format::eR16G16Sfloat) {
        return;
    }
    const vk::ImageLayout old = g.snap_initialized ? vk::ImageLayout::eGeneral
                                                   : vk::ImageLayout::eUndefined;
    const std::array to_dst{
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = old,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = vk::Image{g.depth.image},
            .subresourceRange = {vk::ImageAspectFlagBits::eDepth |
                                     vk::ImageAspectFlagBits::eStencil,
                                 0, 1, 0, 1},
        },
        vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = old,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = vk::Image{g.motion.image},
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        },
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = to_dst.data()});
    const vk::ImageCopy depth_copy{
        .srcSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1},
        .extent = {width, height, 1},
    };
    cmdbuf.copyImage(depth, depth_layout, vk::Image{g.depth.image},
                     vk::ImageLayout::eTransferDstOptimal, depth_copy);
    const vk::ImageCopy motion_copy{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .extent = {width, height, 1},
    };
    cmdbuf.copyImage(motion, motion_layout, vk::Image{g.motion.image},
                     vk::ImageLayout::eTransferDstOptimal, motion_copy);
    std::array to_general = to_dst;
    for (auto& b : to_general) {
        b.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
        b.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
        b.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
        b.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        b.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        b.newLayout = vk::ImageLayout::eGeneral;
    }
    cmdbuf.pipelineBarrier2(
        {.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = to_general.data()});
    g.snap_initialized = true;
    g.snap_fresh = true;
}

bool TakeSnapshot() {
    std::scoped_lock lock{g.mutex};
    const bool fresh = g.snap_fresh && g.snap_initialized;
    g.snap_fresh = false;
    if (!fresh) {
        g.reset = true;
    }
    return fresh;
}

bool NeedsRecreate(u32 width, u32 height, vk::Format format) {
    std::scoped_lock lock{g.mutex};
    return g.feature && (g.feature->width != width || g.feature->height != height ||
                         g.feature->backbuffer_format != static_cast<VkFormat>(format) ||
                         g.feature->render_width != g.snap_width ||
                         g.feature->render_height != g.snap_height);
}

void ReleaseFeature() {
    if (g.api && g.feature) {
        g.api->FgRelease();
    }
    g.feature.reset();
    g.reset = true;
}

void Reset() {
    g.reset = true;
}

bool Record(vk::CommandBuffer cmdbuf, vk::Device device, vk::Image backbuffer,
            vk::ImageView backbuffer_view, vk::Format format, u32 width, u32 height,
            const Output* outputs, u32 count) {
    if (!g.ready || count == 0) {
        return false;
    }
    std::scoped_lock lock{g.mutex};
    if (!g.snap_initialized) {
        return false;
    }
    const u32 rw = g.snap_width, rh = g.snap_height;
    if (!g.feature) {
        const TlgDlssFgFeature desc{width, height, static_cast<VkFormat>(format), rw, rh};
        if (!g.api->FgCreate(static_cast<VkCommandBuffer>(cmdbuf), &desc)) {
            if (!g.failed_logged) {
                g.failed_logged = true;
                LOG_WARNING(Render_Vulkan, "DLSS FG: feature creation failed ({}x{} {})", width,
                            height, vk::to_string(format));
            }
            return false;
        }
        g.feature = desc;
        g.reset = true;
        LOG_INFO(Render_Vulkan, "DLSS FG: {}x{} {} (depth / motion {}x{}), {} generated per frame",
                 width, height, vk::to_string(format), rw, rh, count);
    }

    // Camera.
    TlgDlssFgEvaluate e{};
    std::array<float, 12> rows;
    u64 reproj_count;
    {
        std::scoped_lock rl{g_reproj_mutex};
        rows = g_reproj;
        reproj_count = g_reproj_count;
    }
    bool have_matrix = reproj_count != 0 && reproj_count != g.last_reproj &&
                       ClipToPrevClip(rows, e.clip_to_prev_clip, e.prev_clip_to_clip);
    g.last_reproj = reproj_count;
    if (!have_matrix) {
        for (int i = 0; i < 4; ++i) {
            e.clip_to_prev_clip[i * 5] = 1.0f;
            e.prev_clip_to_clip[i * 5] = 1.0f;
        }
    }
    const float fov = 1.047f, aspect = float(rw) / float(rh), near_plane = 0.1f;
    const float f = 1.0f / std::tan(fov * 0.5f);
    const std::array<float, 16> view_to_clip{f / aspect, 0, 0, 0, 0, f, 0, 0,
                                             0, 0, 0, 1, 0, 0, near_plane, 0};
    const std::array<float, 16> clip_to_view{aspect / f, 0, 0, 0, 0, 1.0f / f, 0, 0,
                                             0, 0, 0, 1.0f / near_plane, 0, 0, 1, 0};
    std::memcpy(e.view_to_clip, view_to_clip.data(), sizeof(e.view_to_clip));
    std::memcpy(e.clip_to_view, clip_to_view.data(), sizeof(e.clip_to_view));
    static const float mv_sign = [] {
        const char* v = std::getenv("SHADPS4_DLSS_FG_MV");
        return v && v[0] == '+' ? 1.0f : -1.0f;
    }();
    // The game's vectors are screen UV, current minus previous (as the DLSS upscaler's -1).
    e.mvec_scale_x = mv_sign;
    e.mvec_scale_y = mv_sign;
    e.camera_near = near_plane;
    e.camera_far = 100000.0f;
    e.camera_fov = fov;
    e.camera_aspect = aspect;
    e.depth_inverted = 1;
    e.reset = g.reset ? 1 : 0;
    e.not_game_frame = 0;
    e.backbuffer =
        Bridge(backbuffer, backbuffer_view, format, vk::ImageAspectFlagBits::eColor, width, height);
    e.depth = Bridge(vk::Image{g.depth.image}, g.depth.view, g.depth_format,
                     vk::ImageAspectFlagBits::eDepth, rw, rh);
    e.motion = Bridge(vk::Image{g.motion.image}, g.motion.view, vk::Format::eR16G16Sfloat,
                      vk::ImageAspectFlagBits::eColor, rw, rh);

    std::array<vk::ImageMemoryBarrier2, MaxGenerated + 1> barriers{};
    barriers[0] = {
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = backbuffer,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    for (u32 k = 0; k < count; ++k) {
        barriers[1 + k] = {
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = outputs[k].image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
    }
    cmdbuf.pipelineBarrier2(
        {.imageMemoryBarrierCount = 1 + count, .pImageMemoryBarriers = barriers.data()});

    bool ok = true;
    for (u32 k = 0; k < count && ok; ++k) {
        e.output = Bridge(outputs[k].image, outputs[k].view, format,
                          vk::ImageAspectFlagBits::eColor, width, height);
        e.multi_frame_count = count;
        e.multi_frame_index = k + 1;
        ok = g.api->FgEvaluate(static_cast<VkCommandBuffer>(cmdbuf), &e) != 0;
    }
    // NGX leaves the command buffer state undefined; later commands rebind what they need.
    const vk::MemoryBarrier2 done{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &done});
    if (!ok) {
        if (!g.failed_logged) {
            g.failed_logged = true;
            LOG_WARNING(Render_Vulkan, "DLSS FG: evaluation failed");
        }
        return false;
    }
    g.reset = false;
    return true;
}

#else

int Wanted() {
    return 0;
}
void AddInstanceExtensions(std::vector<const char*>&) {}
const std::vector<const char*>& DeviceExtensions(vk::Instance, vk::PhysicalDevice) {
    static const std::vector<const char*> none;
    return none;
}
void DeviceCreated(vk::Instance, vk::PhysicalDevice, vk::Device) {}
void Shutdown() {}
bool Ready() {
    return false;
}
u32 Count() {
    return 0;
}
void SetAllocator(VmaAllocator_T*) {}
void Snapshot(vk::CommandBuffer, vk::Image, vk::Format, vk::ImageLayout, vk::Image, vk::Format,
              vk::ImageLayout, u32, u32) {}
bool TakeSnapshot() {
    return false;
}
bool NeedsRecreate(u32, u32, vk::Format) {
    return false;
}
void ReleaseFeature() {}
void Reset() {}
bool Record(vk::CommandBuffer, vk::Device, vk::Image, vk::ImageView, vk::Format, u32, u32,
            const Output*, u32) {
    return false;
}

#endif

} // namespace Vulkan::DlssFg
