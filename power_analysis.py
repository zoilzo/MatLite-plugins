# -*- coding: utf-8 -*-
"""统计功效与样本量：t 检验样本量、功效随效应量曲线、Cohen d、按误差定 n、比例样本量（工具 + 页面）。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "power_analysis",
    "name": "统计功效与样本量",
    "version": "1.0",
    "author": "zoilzo",
    "description": "样本量/功效/效应量/按误差定 n（工具 + 页面，试验设计与教学）",
}


def _ppf_std(p):
    """标准正态分布分位数（用数学近似，避免 scipy 依赖）。"""
    from scipy.stats import norm
    return float(norm.ppf(p))


@ai_tools._reg
@ai_tools._tool({"properties": {"d": {"type": "number", "default": 0.5}, "alpha": {"type": "number", "default": 0.05}, "power": {"type": "number", "default": 0.8}, "two_sided": {"type": "boolean", "default": True}}, "required": []}, category="功效分析")
def sample_size_ttest(d=0.5, alpha=0.05, power=0.8, two_sided=True):
    """两组独立样本 t 检验的每组样本量。需指定效应量 d、显著性 alpha、功效 power。"""
    za = _ppf_std(1 - alpha / 2 if two_sided else 1 - alpha)
    zb = _ppf_std(power)
    n = int(round(2 * ((za + zb) / d) ** 2) + 1)
    total = 2 * n
    return {"text": f"每组需 n={n}，共 {total} 例（效应量 d={d}，alpha={alpha}，功效={power}）。公式：n=2(z_a+z_b)^2/d^2"}


@ai_tools._reg
@ai_tools._tool({"properties": {"d": {"type": "number", "default": 0.5}, "alpha": {"type": "number", "default": 0.05}, "n": {"type": "integer", "default": 64}}, "required": []}, category="功效分析")
def power_curve_points(d=0.5, alpha=0.05, n=64):
    """给定 n，返回该效应量下的检验功效。"""
    za = _ppf_std(1 - alpha / 2)
    z = d / math.sqrt(2.0 / n) - za
    from scipy.stats import norm
    power = float(norm.cdf(z))
    return {"text": f"n={n}（每组），d={d}，alpha={alpha} 时功效 power={power:.4f}（{power*100:.1f}%）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"m1": {"type": "number", "default": 100}, "m2": {"type": "number", "default": 102}, "s1": {"type": "number", "default": 5}, "s2": {"type": "number", "default": 5}}, "required": []}, category="功效分析")
def cohen_d(m1=100, m2=102, s1=5, s2=5):
    """Cohen d 效应量 = (均差) / (合并标准差)。"""
    sp = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    d = (m1 - m2) / sp
    note = "小" if abs(d) < 0.5 else ("中" if abs(d) < 0.8 else "大")
    return {"text": f"Cohen d={d:.3f}（{note} 效应）。合并标准差 sp={sp:.4f}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"sigma": {"type": "number", "default": 5}, "half_width": {"type": "number", "default": 1}, "alpha": {"type": "number", "default": 0.05}}, "required": []}, category="功效分析")
def ci_sample_size(sigma=5, half_width=1, alpha=0.05):
    """估计总体均值，使置信区间半宽不超过 half_width 所需样本量。"""
    z = _ppf_std(1 - alpha / 2)
    n = int(round((z * sigma / half_width) ** 2) + 1)
    return {"text": f"需 n={n}，使均值 95% 置信区间半宽 <= {half_width}（sigma={sigma}）。"}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number", "default": 0.5}, "half_width": {"type": "number", "default": 0.05}, "alpha": {"type": "number", "default": 0.05}}, "required": []}, category="功效分析")
def proportion_sample_size(p=0.5, half_width=0.05, alpha=0.05):
    """估计比例 p 所需样本量（精确度 half_width）。"""
    z = _ppf_std(1 - alpha / 2)
    n = int(round((z / half_width) ** 2 * p * (1 - p)) + 1)
    return {"text": f"需 n={n}，使比例 p 的置信区间半宽 <= {half_width}（p={p}）。"}


class PowerPage(ui.BasePage):
    NAME = "统计功效与样本量"
    EMOJI = "\U0001F50E"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="\U0001F50E 统计功效与样本量", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="效应量 d：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
        self.d = ctk.CTkEntry(body); self.d.insert(0, "0.5"); self.d.pack(fill="x")
        ctk.CTkLabel(body, text="显著性 alpha：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
        self.al = ctk.CTkEntry(body); self.al.insert(0, "0.05"); self.al.pack(fill="x")
        btn = ctk.CTkFrame(body, fg_color="transparent")
        btn.pack(fill="x", pady=ui.SPACE["sm"])
        ops = [
            ("样本量(t检验)", lambda: self._run(sample_size_ttest, [float(self.d.get() or 0.5), float(self.al.get() or 0.05), 0.8])),
            ("功效", lambda: self._run(power_curve_points, [float(self.d.get() or 0.5), float(self.al.get() or 0.05), 64])),
            ("Cohen d", lambda: self._run(cohen_d, [100, 102, 5, 5])),
            ("按误差定n", lambda: self._run(ci_sample_size, [5, 1, 0.05])),
        ]
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btn, text=txt, command=fn, width=100).grid(row=0, column=i, padx=4, pady=4)
        self.out = ui.mono_textbox(body, ui.FONT["body"])
        self.out.pack(fill="both", expand=True, pady=(ui.SPACE["sm"], 0))

    def _run(self, fn, args):
        try:
            r = fn(*args)
        except Exception as e:
            r = {"text": "计算失败：" + repr(e)}
        try:
            self.out.configure(state="normal")
            self.out.delete("1.0", "end")
            self.out.insert("end", r.get("text", "") + "\n")
            self.out.configure(state="disabled")
        except Exception:
            pass


PAGES = [PowerPage]
