# -*- coding: utf-8 -*-
"""流行病学模型：SIR/SEIR 仓室模型、基本再生数 R0、Logistic 种群增长、哈代-温伯格平衡。AI 工具 + 页面。"""
import customtkinter as ctk
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "epi_models",
    "name": "流行病学模型",
    "version": "1.0",
    "author": "zoilzo",
    "description": "SIR/SEIR 传染病模型、基本再生数 R0、Logistic 种群增长、哈代-温伯格平衡（工具 + 页面）",
}


def _f(v, d):
    """稳健数值：容忍 None/字符串（含逗号、百分号），失败用默认值。"""
    if v is None:
        return float(d)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("，", "").replace("%", "").strip()
    if s in ("", "-", "无", "空", "None"):
        return float(d)
    try:
        return float(s)
    except Exception:
        return float(d)


def _n(v, d):
    """整数解析：保证非负。"""
    return max(0, int(_f(v, d)))


def _fmt(x):
    """数值显示：过大/过小用科学计数，其余保留 4 位小数去尾零。"""
    x = float(x)
    if abs(x) >= 1e6 or (abs(x) < 1e-4 and x != 0):
        return f"{x:.4g}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


@ai_tools._reg
@ai_tools._tool({"properties": {"S0": {"type": "number"}, "I0": {"type": "number"}, "beta": {"type": "number"}, "gamma": {"type": "number"}, "days": {"type": "number"}}, "required": []}, category="流行病学")
def sir_model(S0=999, I0=1, beta=0.3, gamma=0.1, days=120):
    """SIR 传染病模型。S/I 为易感与感染人数，beta 传染率，gamma 恢复率，days 模拟天数。返回峰值与最终规模。"""
    S, I, R = float(S0), float(I0), 0.0
    b, g, T = _f(beta, 0.3), _f(gamma, 0.1), _n(days, 120)
    peak, peak_day = I, 0
    for t in range(1, T + 1):
        new = b * S * I / max(1e-9, S + I + R)
        rec = g * I
        S, I, R = S - new, I + new - rec, R + rec
        if I > peak:
            peak, peak_day = I, t
    r0 = (b / g) if g else float("inf")
    return {"text": (f"SIR 模型（β={_fmt(b)}, γ={_fmt(g)}, R0={_fmt(r0)}）\n"
                     f"峰值感染: {_fmt(peak)} 人（第 {peak_day} 天）\n"
                     f"最终: 易感 {_fmt(S)}, 感染 {_fmt(I)}, 恢复 {_fmt(R)}")}


@ai_tools._reg
@ai_tools._tool({"properties": {"S0": {"type": "number"}, "E0": {"type": "number"}, "I0": {"type": "number"}, "beta": {"type": "number"}, "gamma": {"type": "number"}, "sigma": {"type": "number"}, "days": {"type": "number"}}, "required": []}, category="流行病学")
def seir_model(S0=999, E0=0, I0=1, beta=0.3, gamma=0.1, sigma=0.2, days=120):
    """SEIR 传染病模型（含潜伏期）。E 为潜伏者，sigma 为潜伏期转化率。返回峰值与最终规模。"""
    S, E, I, R = float(S0), float(E0), float(I0), 0.0
    b, g, sg, T = _f(beta, 0.3), _f(gamma, 0.1), _f(sigma, 0.2), _n(days, 120)
    peak, peak_day = I, 0
    for t in range(1, T + 1):
        new = b * S * I / max(1e-9, S + E + I + R)
        inc = sg * E
        rec = g * I
        S, E, I, R = S - new, E + new - inc, I + inc - rec, R + rec
        if I > peak:
            peak, peak_day = I, t
    r0 = (b / g) if g else float("inf")
    return {"text": (f"SEIR 模型（β={_fmt(b)}, γ={_fmt(g)}, R0={_fmt(r0)}）\n"
                     f"峰值感染: {_fmt(peak)} 人（第 {peak_day} 天）\n"
                     f"最终: 易感 {_fmt(S)}, 潜伏 {_fmt(E)}, 感染 {_fmt(I)}, 恢复 {_fmt(R)}")}


@ai_tools._reg
@ai_tools._tool({"properties": {"beta": {"type": "number"}, "gamma": {"type": "number"}, "growth_rate": {"type": "number"}, "serial_interval": {"type": "number"}}, "required": []}, category="流行病学")
def basic_reproduction_number(beta=0.3, gamma=0.1, growth_rate=None, serial_interval=5.0):
    """基本再生数 R0。仓室法 R0=β/γ；也可由流行增长速率 r 与代际间隔 T 推算 R0≈1+r·T。"""
    if growth_rate is not None:
        r = _f(growth_rate, 0.0)
        T = _f(serial_interval, 5.0)
        r0 = 1.0 + r * T
        return {"text": f"由流行增长率 r={_fmt(r)} 与代际间隔 T={_fmt(T)} 估算：R0 ≈ {_fmt(r0)}"}
    b, g = _f(beta, 0.3), _f(gamma, 0.1)
    r0 = (b / g) if g else float("inf")
    return {"text": f"由仓室模型参数估计：R0 = β/γ = {_fmt(b)} / {_fmt(g)} = {_fmt(r0)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"P0": {"type": "number"}, "r": {"type": "number"}, "K": {"type": "number"}, "years": {"type": "number"}}, "required": []}, category="流行病学")
def logistic_growth(P0=100, r=0.5, K=1000, years=10):
    """Logistic 种群增长：P(t)=K·P0·e^{rt}/(K+P0(e^{rt}-1))。返回 0..years 逐年种群。"""
    p0, rr, kk, T = _f(P0, 100.0), _f(r, 0.5), _f(K, 1000.0), _n(years, 10)
    if kk <= 0:
        return {"text": "环境容量 K 必须为正数。"}
    pts = []
    for t in range(0, T + 1):
        e = np.exp(rr * t)
        p = kk * p0 * e / max(1e-12, kk + p0 * (e - 1))
        pts.append(f"t={t}: {_fmt(p)}")
    return {"text": "Logistic 增长（K=%s）\n%s" % (_fmt(kk), "，".join(pts))}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number"}}, "required": []}, category="流行病学")
def hardy_weinberg(p=0.5):
    """哈代-温伯格平衡：由等位基因频率 p 计算基因型频率 (p², 2pq, q²)。"""
    pp = min(1.0, max(0.0, _f(p, 0.5)))
    q = 1.0 - pp
    aa = pp * pp
    ao = 2.0 * pp * q
    oo = q * q
    total = aa + ao + oo
    return {"text": (f"等位基因：p={_fmt(pp)}, q={_fmt(q)}\n"
                     f"基因型频率：AA={_fmt(aa)}, Aa={_fmt(ao)}, aa={_fmt(oo)}\n"
                     f"合计={_fmt(total)}（应=1）")}


class EpiPage(ui.BasePage):
    NAME = "流行病学模型"
    EMOJI = "\U0001F9A0"  # 🦠

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🦠 流行病学模型", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="SIR/SEIR 仓室模型、R0、Logistic 增长、哈代-温伯格",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["md"], 0))
        self.ent = {}
        pairs = [("S0", "易感"), ("I0", "感染"), ("beta", "传染率"), ("gamma", "恢复率"), ("days", "天数")]
        defaults = {"S0": "999", "I0": "1", "beta": "0.3", "gamma": "0.1", "days": "120"}
        for i, (k, lab) in enumerate(pairs):
            ctk.CTkLabel(row, text=lab).grid(row=i // 3, column=(i % 3) * 2, padx=(0, 2), pady=(0, ui.SPACE["sm"]))
            e = ctk.CTkEntry(row, width=90)
            e.grid(row=i // 3, column=(i % 3) * 2 + 1, padx=(0, ui.SPACE["sm"]), pady=(0, ui.SPACE["sm"]))
            e.insert(0, defaults[k])
            self.ent[k] = e
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(ui.SPACE["md"], 0))
        for text, cmd in (("运行 SIR", self.do_sir), ("运行 SEIR", self.do_seir), ("R0 估算", self.do_r0),
                          ("Logistic 增长", self.do_logistic), ("哈代-温伯格", self.do_hw)):
            ctk.CTkButton(btns, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _val(self, k):
        return self.ent[k].get().strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_sir(self):
        self._run(sir_model, _f(self._val("S0"), 999), _f(self._val("I0"), 1), _f(self._val("beta"), 0.3), _f(self._val("gamma"), 0.1), _n(self._val("days"), 120))

    def do_seir(self):
        self._run(seir_model, _f(self._val("S0"), 999), 0, _f(self._val("I0"), 1), _f(self._val("beta"), 0.3), _f(self._val("gamma"), 0.1), 0.2, _n(self._val("days"), 120))

    def do_r0(self):
        self._run(basic_reproduction_number, _f(self._val("beta"), 0.3), _f(self._val("gamma"), 0.1))

    def do_logistic(self):
        self._run(logistic_growth, 100, 0.5, 1000, _n(self._val("days"), 120))

    def do_hw(self):
        self._run(hardy_weinberg, 0.5)


PAGES = [EpiPage]
