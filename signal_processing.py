# -*- coding: utf-8 -*-
"""信号/图像处理：FFT 频谱、数字低通滤波、一维卷积、平滑去噪、Sobel 边缘检测。
AI 工具 + 页面（交叉领域，教学友好）。"""
import numpy as np
from scipy import signal as sg
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "signal_processing",
    "name": "信号处理",
    "version": "1.0",
    "author": "zoilzo",
    "description": "信号/图像处理：FFT 频谱分析、数字低通/高通滤波、一维卷积、平滑去噪、Sobel 边缘检测（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _nums(v):
    """解析一维序列：接受逗号/空格分隔字符串或列表。"""
    if isinstance(v, (list, tuple)):
        return np.asarray([float(x) for x in v], dtype=float)
    s = str(v).replace("，", ",").replace(";", ",").replace("\n", ",")
    arr = np.asarray([float(x) for x in s.split(",") if x.strip()], dtype=float)
    return arr


def _mat(v):
    """解析二维图像矩阵：行以 ; 或换行分隔，列以逗号分隔。"""
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    s = str(v).replace("，", ",").replace(";", "\n")
    rows = [r.strip() for r in s.split("\n") if r.strip()]
    if not rows:
        return np.zeros((1, 1))
    return np.asarray([[float(x) for x in r.replace(";", ",").split(",") if x.strip()] for r in rows], dtype=float)


def _default_wave(fs=100.0, T=2.0):
    """默认测试信号：5Hz 主频 + 20Hz 次频 + 轻微噪声。"""
    t = np.arange(0, T, 1.0 / fs)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.RandomState(0).randn(t.size)
    return t, x


# ------------------------------------------------------------
# 工具 1：FFT 频谱分析
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "signal": {"type": "string", "description": "信号采样序列（逗号分隔）；留空用默认测试信号"},
        "sample_rate": {"type": "number", "description": "采样率 fs（Hz）"},
    },
}, category="信号处理")
def fft_spectrum(signal="", sample_rate="100"):
    """FFT 频谱分析：对采样信号做快速傅里叶变换，输出功率谱与主要频率成分。用于定位周期/主频。"""
    fs = float(sample_rate or 100)
    if str(signal).strip():
        x = _nums(signal)
        if x.size < 2:
            raise ValueError("信号序列至少 2 个采样点。")
    else:
        _t, x = _default_wave(fs)
    n = x.size
    x = x - np.mean(x)  # 去除直流
    fft = np.fft.rfft(x)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(fft) / n
    power = amp ** 2
    # 去掉 0Hz 直流后找主导频率
    k = 1 if freq.size > 1 else 0
    if k < freq.size:
        main_idx = int(np.argmax(power[k:])) + k
        dominant = (float(freq[main_idx]), float(power[main_idx]))
    else:
        dominant = (float(freq[-1]), float(power[-1]))
    # 前 5 个峰（按功率降序，去除重复邻域）
    order = np.argsort(power)[::-1]
    seen = []
    for i in order:
        if i == 0:
            continue
        if all(abs(int(i) - idx) >= 2 for idx in seen):
            seen.append(int(i))
        if len(seen) >= 5:
            break
    lines = [f"FFT 频谱分析：N = {n} 点，采样率 fs = {fs:g} Hz，频率分辨率 Δf = {fs / n:.3g} Hz", "",
             f"主导频率 ≈ {dominant[0]:.4g} Hz（功率 {dominant[1]:.3g}）", "",
             f"{'频率 (Hz)':>14} {'幅值 (归一)':>14} {'功率':>14}"]
    for i in seen[:5]:
        lines.append(f"{freq[i]:>14.4g} {amp[i]:>14.4g} {power[i]:>14.4g}")
    lines.append("")
    lines.append("➤ 含义：谱峰对应的频率即信号的周期性成分；噪声表现为整个频带的低幅基底。")
    lines.append("➤ 奈奎斯特提示：可分辨的最高频率 = fs/2 = " + f"{fs / 2:g} Hz。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：数字低通滤波
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "signal": {"type": "string", "description": "信号采样序列（逗号分隔）；留空用默认测试信号"},
        "cutoff": {"type": "number", "description": "截止频率（Hz）"},
        "sample_rate": {"type": "number", "description": "采样率 fs（Hz）"},
        "order": {"type": "number", "description": "巴特沃斯滤波器阶数"},
        "mode": {"type": "string", "description": "low 低通 / high 高通 / band 带通"},
    },
}, category="信号处理")
def lowpass_filter(signal="", cutoff="10.0", sample_rate="100", order="2", mode="low"):
    """巴特沃斯数字滤波：low 低通、high 高通、band 带通。零相位（filtfilt）双向处理避免相位失真。"""
    fs = float(sample_rate or 100)
    fc = float(cutoff or 10)
    N = int(order or 2)
    if str(signal).strip():
        x = _nums(signal)
        if x.size < 10:
            raise ValueError("信号过短，至少 10 点以进行滤波。")
    else:
        _t, x = _default_wave(fs)
    m = str(mode or "low").strip().lower()
    nyq = 0.5 * fs
    if fc >= nyq:
        raise ValueError("截止频率必须小于奈奎斯特频率 fs/2。")
    sos = sg.butter(N, fc / nyq, btype=m, output="sos")
    y = sg.sosfiltfilt(sos, x)
    # 统计滤波前后高频能量（相邻差分）
    hf_in = float(np.mean(np.diff(x) ** 2))
    hf_out = float(np.mean(np.diff(y) ** 2))
    lines = [f"巴特沃斯 {m} 滤波：fs = {fs:g} Hz，截止 = {fc:g} Hz，阶数 = {N}", "",
             f"输入长度 = {x.size}，输出长度 = {y.size}", "",
             f"{'序号':>6} {'输入':>12} {'滤波后':>12}"]
    for i in range(min(10, x.size)):
        lines.append(f"{i:>6} {x[i]:>12.4g} {y[i]:>12.4g}")
    lines.append("")
    lines.append(f"高频（相邻差分能量）：滤波前 {hf_in:.4g} → 滤波后 {hf_out:.4g}")
    lines.append("➤ 含义：低通削弱高频噪声；高通保留快速变化；带通只留指定频段。filtfilt 零相位不引入延迟。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：一维卷积
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "signal": {"type": "string", "description": "输入信号（逗号分隔）"},
        "kernel": {"type": "string", "description": "卷积核（逗号分隔）"},
        "mode": {"type": "string", "description": "full 完整 / same 同长 / valid 有效"},
    },
}, category="信号处理")
def signal_conv(signal="1,2,3,4,5", kernel="1,1", mode="full"):
    """一维卷积：y[n]=Σ x[k]·h[n-k]。完整/同长/有效三种模式。卷积是线性时不变系统与图像模板(核)的基础。"""
    x = _nums(signal)
    h = _nums(kernel)
    if x.size == 0 or h.size == 0:
        raise ValueError("信号与卷积核都不能为空。")
    m = str(mode or "full").strip().lower()
    y = np.convolve(x, h, mode=m)
    lines = [f"一维卷积：信号 x 长度 = {x.size}，核 h 长度 = {h.size}，模式 = {m}", "",
             f"x = {np.array2string(x, precision=4, separator=', ')}",
             f"h = {np.array2string(h, precision=4, separator=', ')}", "",
             f"y = x * h（长度 {y.size}） = {np.array2string(y, precision=4, separator=', ')}"]
    lines.append("")
    lines.append("➤ 含义：卷积 = 滑动加权求和。框式滤波/均值滤波/图像模板卷积都基于此；由卷积定理等价于频域相乘。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：平滑去噪
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "signal": {"type": "string", "description": "含噪信号（逗号分隔）；留空用默认测试信号"},
        "window": {"type": "number", "description": "窗口大小（奇数）"},
        "method": {"type": "string", "description": "movavg 移动平均 / savgol Savitzky-Golay"},
    },
}, category="信号处理")
def signal_smooth(signal="", window="5", method="savgol"):
    """平滑去噪：移动平均或 Savitzky-Golay 多项式平滑。保留信号趋势、压制高频随机噪声。"""
    if str(signal).strip():
        x = _nums(signal)
        if x.size < 7:
            raise ValueError("信号过短，至少 7 点。")
    else:
        _t, x = _default_wave(fs=100)
    w = int(window or 5)
    if w % 2 == 0:
        w += 1
    w = max(3, min(w, x.size if x.size % 2 else x.size - 1))
    m = str(method or "savgol").strip().lower()
    if m.startswith("m"):
        y = np.convolve(x, np.ones(w) / w, mode="same")
        tag = "移动平均 (window=%d)" % w
    else:
        poly = min(3, w - 1)
        y = sg.savgol_filter(x, window_length=w, polyorder=poly)
        tag = "Savitzky-Golay (window=%d, poly=%d)" % (w, poly)
    lines = [f"平滑去噪（{tag}）：输入长度 = {x.size}", "",
             f"{'序号':>6} {'原信号':>12} {'平滑后':>12}"]
    for i in range(min(12, x.size)):
        lines.append(f"{i:>6} {x[i]:>12.4g} {y[i]:>12.4g}")
    lines.append("")
    lines.append(f"噪声（差分能量）：{np.mean(np.diff(x) ** 2):.4g} → 平滑后 {np.mean(np.diff(y) ** 2):.4g}")
    lines.append("➤ 含义：窗口越大越平滑但越失真；Savgol 在平滑同时更好保留峰的形状。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 5：Sobel 边缘检测
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "image": {"type": "string", "description": "灰度图像矩阵（行以分号/换行分隔，列以逗号分隔）；留空用默认正方形图"},
        "threshold": {"type": "string", "description": "归一化阈值；留空自动选 Otsu 近似"},
    },
}, category="信号处理")
def image_edge(image="", threshold=""):
    """Sobel 边缘检测：对灰度图计算水平+垂直梯度方向角的强度，输出边缘强度分布与检出比例。"""
    if str(image).strip():
        img = _mat(image)
    else:
        # 默认：100x100 暗背景 + 中央亮方块 → 清晰矩形边缘
        img = np.zeros((100, 100))
        img[30:70, 35:65] = 1.0
    if img.ndim != 2 or img.shape[0] < 3 or img.shape[1] < 3:
        raise ValueError("图像至少 3x3。")
    # Sobel 核
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    gx = sg.convolve2d(img, kx, mode="same")
    gy = sg.convolve2d(img, ky, mode="same")
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_n = mag / (mag.max() if mag.max() > 0 else 1.0)
    # 阈值（默认 Otsu 近似：均值 + 0.5 方差梯度的简单自适应）
    if str(threshold).strip():
        th = float(threshold)
    else:
        th = float(np.mean(mag_n) + 0.5 * np.std(mag_n))
    edges = mag_n > th
    ratio = float(edges.mean())
    lines = [f"Sobel 边缘检测：图像 {img.shape[0]}×{img.shape[1]}，阈值 = {th:.4g}", "",
             f"边缘强度最大值 = {mag.max():.4g}（灰度差异越大越高）",
             f"检出边缘像素占比 = {ratio * 100:.3g}%", "",
             f"{'行':>3} {'均值梯度':>10} {'边缘像素':>8}"]
    for i in range(min(8, img.shape[0])):
        lines.append(f"{i:>3} {mag[i].mean():>10.4g} {int(edges[i].sum()):>8}")
    lines.append("")
    lines.append("➤ 含义：边缘 = 灰度突变处。Sobel 用 3×3 卷积核分别测水平/垂直梯度再合成强度，是经典图像分割基础。")
    lines.append("➤ 提示：将左上 3×3 邻域与 Sobel 核点积即得该点梯度（卷积），与一维卷积同源。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "信号处理"
    _TOOLS = {}

    def __init__(self, master):
        super().__init__(master, layout=False)
        self._fields = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text=self._TITLE, font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="功能：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w", padx=(0, ui.SPACE["sm"]))
        self._names = [v[0] for v in self._TOOLS.values()]
        self._key_of = {v[0]: k for k, v in self._TOOLS.items()}
        self.mode_var = ctk.StringVar(value=self._names[0])
        ctk.CTkOptionMenu(row, values=self._names, variable=self.mode_var,
                          command=lambda _: self._rebuild()).grid(row=0, column=1, sticky="w")
        self.desc = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=ui.FONT["body"]), text_color="gray60",
                                 wraplength=560, justify="left")
        self.desc.pack(anchor="w", pady=(ui.SPACE["xs"], 0))
        self.field_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.field_frame.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(body, text="计算", height=h, fg_color=ui.body(), command=self._run).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self._rebuild()

    def _rebuild(self):
        for w in self.field_frame.winfo_children():
            w.destroy()
        self._fields = {}
        key = self._key_of[self.mode_var.get()]
        label, desc, fields, func = self._TOOLS[key]
        self.desc.configure(text=desc)
        self._current = func
        for r, (fkey, flabel, fdefault) in enumerate(fields):
            ctk.CTkLabel(self.field_frame, text=flabel, font=ctk.CTkFont(size=ui.FONT["body"])).grid(
                row=r, column=0, sticky="w", pady=(0, ui.SPACE["xs"]))
            e = ctk.CTkEntry(self.field_frame)
            e.insert(0, str(fdefault))
            e.grid(row=r, column=1, sticky="ew", pady=(0, ui.SPACE["xs"]))
            self.field_frame.grid_columnconfigure(1, weight=1)
            self._fields[fkey] = e

    def _run(self):
        vals = {k: e.get().strip() for k, e in self._fields.items()}
        out = self.out
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.configure(state="disabled")
        try:
            res = self._current(**vals)
            text = res.get("text", str(res)) if isinstance(res, dict) else str(res)
        except Exception as ex:
            text = f"计算出错：{ex}"
        out.configure(state="normal")
        out.insert("1.0", text)
        out.configure(state="disabled")


class SignalPage(_FormPage):
    NAME = "信号处理"
    EMOJI = "\U0001F5B8"
    _TITLE = "🌐 信号 / 图像处理"
    _TOOLS = {
        "fft_spectrum": (
            "FFT 频谱分析",
            "对采样信号做 FFT，输出主导频率与功率谱，定位周期成分。",
            [("signal", "信号序列（逗号分隔，留空=默认 5Hz+20Hz 测试波）", ""),
             ("sample_rate", "采样率 fs (Hz)", "100")],
            fft_spectrum),
        "lowpass_filter": (
            "数字低通/带通滤波",
            "巴特沃斯滤波：低通/高通/带通，零相位 filtfilt 去噪。",
            [("signal", "信号序列（逗号分隔，留空=默认测试波）", ""),
             ("cutoff", "截止频率 (Hz)", "10.0"),
             ("sample_rate", "采样率 fs (Hz)", "100"),
             ("order", "滤波器阶数", "2"),
             ("mode", "low 低通 / high 高通 / band 带通", "low")],
            lowpass_filter),
        "signal_conv": (
            "一维卷积",
            "y = x * h：滑动加权求和，三种模式。滤波与图像模板的基础。",
            [("signal", "输入信号（逗号分隔）", "1,2,3,4,5"),
             ("kernel", "卷积核（逗号分隔）", "1,1"),
             ("mode", "full 完整 / same 同长 / valid 有效", "full")],
            signal_conv),
        "signal_smooth": (
            "平滑去噪",
            "移动平均或 Savitzky-Golay 平滑，压高频噪声保趋势。",
            [("signal", "含噪信号（逗号分隔，留空=默认测试波）", ""),
             ("window", "窗口大小（奇数）", "5"),
             ("method", "movavg 移动平均 / savgol Savitzky-Golay", "savgol")],
            signal_smooth),
        "image_edge": (
            "Sobel 边缘检测",
            "对灰度图像用 Sobel 卷积核测水平/垂直梯度，输出边缘强度与检出比例。",
            [("image", "灰度图像矩阵（留空=默认方块图）", ""),
             ("threshold", "归一化阈值（留空=自动）", "")],
            image_edge),
    }


PAGES = [SignalPage]