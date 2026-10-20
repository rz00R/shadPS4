// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later
// tlg cross build (MinGW): comsuppw.lib is MSVC-only; _bstr_t (hwinfo's WMI queries) needs these.

#ifdef __MINGW32__
#include <windows.h>
#include <comutil.h>
#include <oleauto.h>

namespace _com_util {

BSTR WINAPI ConvertStringToBSTR(const char* src) {
    if (!src) {
        return nullptr;
    }
    const int n = MultiByteToWideChar(CP_ACP, 0, src, -1, nullptr, 0);
    if (n <= 0) {
        return nullptr;
    }
    BSTR out = SysAllocStringLen(nullptr, static_cast<UINT>(n - 1));
    if (out) {
        MultiByteToWideChar(CP_ACP, 0, src, -1, out, n);
    }
    return out;
}

char* WINAPI ConvertBSTRToString(BSTR src) {
    if (!src) {
        return nullptr;
    }
    const int n = WideCharToMultiByte(CP_ACP, 0, src, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        return nullptr;
    }
    char* out = new char[static_cast<size_t>(n)];
    WideCharToMultiByte(CP_ACP, 0, src, -1, out, n, nullptr, nullptr);
    return out;
}

} // namespace _com_util
#endif
