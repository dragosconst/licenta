import win32gui, win32ui, win32con
import ctypes
from ctypes import c_int, c_bool
import ctypes.wintypes
from ctypes.wintypes import HWND, DWORD

def grab_all_open_windows():
    win_names = []
    dwmapi = ctypes.WinDLL("dwmapi")
    DWMWA_CLOAKED = 14
    # annoying overlays, can't find a better way to skip them
    OVERLAYS = ["NVIDIA GeForce Overlay", "Program Manager"]
    isCloacked = c_int(0)
    notPeeking = c_bool(True)
    def winEnumHandler(hwnd, ctx):
        # code to check whether or not it is a suspended window
        if win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd) and win32gui.GetWindowText(hwnd) != '':
            dwmapi.DwmGetWindowAttribute(HWND(hwnd), DWORD(DWMWA_CLOAKED), ctypes.byref(isCloacked),
                                         ctypes.sizeof(isCloacked))
            if (isCloacked.value == 0) and win32gui.GetWindowText(hwnd) not in OVERLAYS:
                win_names.append(win32gui.GetWindowText(hwnd))

    win32gui.EnumWindows(winEnumHandler, None)
    return win_names