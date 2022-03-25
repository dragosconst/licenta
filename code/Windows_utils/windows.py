import time

import win32gui, win32ui, win32con
import ctypes
from ctypes import c_int, c_bool
import ctypes.wintypes
from ctypes.wintypes import HWND, DWORD
from ctypes import windll
import numpy as np
from PIL import Image

def grab_selected_window_contents(wName, w: int=800, h: int=600, x: int=0, y: int=0) -> np.ndarray:
    hwnd = win32gui.FindWindow(None, wName)
    if not hwnd:
        raise Exception('Window not found: {}'.format(wName))

    left, top, right, bot = win32gui.GetWindowRect(hwnd)
    w_wnd = right - left
    h_wnd = bot - top
    # print(w, h)

    # win32gui.SetForegroundWindow(hwnd)
    time.sleep(1e-3)

    hdesk = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hdesk)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w_wnd, h_wnd)
    cDC.SelectObject(dataBitMap)

    result = windll.user32.PrintWindow(hwnd, cDC.GetSafeHdc(), 2)

    # convert the raw data into a format opencv can read
    dataInfo = dataBitMap.GetInfo()
    # dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = Image.frombuffer('RGB', (dataInfo["bmWidth"], dataInfo["bmHeight"]), signedIntsArray, 'raw', 'BGRX', 0, 1)
    # img = img.resize((w, h))
    img = np.asarray(img)
    img = img[y:y+h, x:x+w, :]
    # img = img.reshape((h, w, 3))
    # free resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hdesk, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img


def grab_screen_area(x, y, w, h):
    hdesk = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hdesk)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (x, y), win32con.SRCCOPY)

    # convert the raw data into a format opencv can read
    dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img = img.reshape((h, w, 4))
    # free resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hdesk, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return img

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