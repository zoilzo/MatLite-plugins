# -*- coding: utf-8 -*-
"""概率分布：正态/二项/泊松/指数/均匀/t 分布的属性、概率、分位数。AI 工具 + 页面。"""
import customtkinter as ctk
from scipy import stats

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "probability_dist",
    "name": "概率分布",
    "version": "1.0",
    "author": "zoilzo",
    "description": "正态/二项/泊松/指数/均匀/t 分布的均值、方差、CDF、分位数（工具 + 页面）",
}

# 分布名 -> (中文名, 构造器(dist 的参数 a,b) -> scipy 冻结分布, 是否离散)
_DISTS: dict = {
    "norm": ("正态", lambda a, b: stats.norm(a, max(abs(float(b)), 1e-12)), False),
    "binom": ("二项", lambda a, b: stats.binom(max(0, int(a)), float(b)), True),
    "poisson": ("泊松", lambda a, b: stats.poisson(max(0.0, float(a))), True),
    "exp": ("指数", lambda a, b: stats.expon(scale=max(float(a), 1e-12)), False),
    "uniform": ("均匀", lambda a, b: stats.uniform(float(a), max(float(b) - float(a), 1e-12)), False),
    "t": ("t分布", lambda a, b: stats.t(max(float(a), 1e-6)), False),
}

# 默认参数：norm(a=均值,b=标差) binom(a=n,b=p) poisson(a=λ) exp(a=均值) uniform(a=下限,b=上限) t(a=自由度)
_DEFAULT = {
    "norm": (0, 1),
    "binom": (10, 0.5),
    "poisson": (4, 0),
    "exp": (1, 0),
    "uniform": (0, 1),
    "t": (10, 0),
}


def _get(dist, a, b):
    """按名字取分布对象。返回 (中文名, dist 对象, 是否离散)。"""
    key = str(dist).strip().lower()
    if key not in _DISTS:
        raise ValueError("dist 只支持 " + "/".join(_DISTS))
    name, build, disc = _DISTS[key]
    da, db = _DEFAULT[key]
    if a in (None, "") or (isinstance(a, str) and not a.strip()):
        a = da
    if b in (None, "") or (isinstance(b, str) and not b.strip()):
        b = db
    return name, build(float(a), float(b)), disc


def _fmt(x):
    try:
        return f"{float(x):.6g}"
    except (TypeError, ValueError):
        return str(x)


@ai_tools._reg
@ai_tools._tool({"properties": {"dist": {"type": "string", "description": "分布名: norm/binom/poisson/exp/uniform/t"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["dist"]}, category="概率分布")
def dist_props(dist, a=0, b=1):
    """某概率分布的均值/方差/标准差。参数：norm(a=均值,b=标准差)、binom(a=n,b=p)、poisson(a=λ)、exp(a=均值)、uniform(a=下限,b=上限)、t(a=自由度)。"""
    name, d, disc = _get(dist, a, b)
    s = f"{name}分布: 均值={_fmt(d.mean())}, 方差={_fmt(d.var())}, 标准差={_fmt(d.std())}"
    if disc:
        lo, hi = d.support()
        s += f"\n取值区间: {_fmt(lo)} ~ {_fmt(hi)}（整数）"
    else:
        lo, hi = d.support()
        s += f"\n支撑区间: {_fmt(lo)} ~ {_fmt(hi)}"
    return {"text": s}


@ai_tools._reg
@ai_tools._tool({"properties": {"dist": {"type": "string"}, "x": {"type": "number"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["dist", "x"]}, category="概率分布")
def dist_cdf(dist, x, a=0, b=1):
    """P(X ≤ x)，即小于等于某个值的概率。参数同 dist_props。"""
    name, d, _ = _get(dist, a, b)
    p = float(d.cdf(float(x)))
    return {"text": f"{name}分布, P(X ≤ {_fmt(x)}) = {p:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"dist": {"type": "string"}, "k": {"type": "number"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["dist", "k"]}, category="概率分布")
def dist_pmf(dist, k, a=0, b=1):
    """离散分布 P(X=k)；连续分布给出 x 处的概率密度 f(x)。参数同 dist_props。"""
    name, d, disc = _get(dist, a, b)
    if disc:
        p = float(d.pmf(int(float(k))))
        return {"text": f"{name}分布, P(X = {k}) = {p:.6g}"}
    p = float(d.pdf(float(k)))
    return {"text": f"{name}分布, 密度 f({_fmt(k)}) = {p:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"dist": {"type": "string"}, "lo": {"type": "number"}, "hi": {"type": "number"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["dist", "lo", "hi"]}, category="概率分布")
def dist_range(dist, lo, hi, a=0, b=1):
    """P(lo < X < hi) 落在某区间的概率。参数同 dist_props。"""
    name, d, _ = _get(dist, a, b)
    p = float(d.cdf(float(hi)) - d.cdf(float(lo)))
    return {"text": f"{name}分布, P({_fmt(lo)} < X < {_fmt(hi)}) = {p:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"dist": {"type": "string"}, "q": {"type": "number"}, "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["dist", "q"]}, category="概率分布")
def dist_quantile(dist, q, a=0, b=1):
    """分位数：使 P(X ≤ x) = q 的 x（q 取 0~1 之间）。参数同 dist_props。"""
    name, d, _ = _get(dist, a, b)
    q = float(q)
    if q < 0 or q > 1:
        raise ValueError("q 需在 0~1 之间")
    x = float(d.ppf(q))
    return {"text": f"{name}分布, 分位数({q:g}) x = {_fmt(x)}"}


class ProbPage(ui.BasePage):
    NAME = "概率分布"
    EMOJI = "\U0001F3B2"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🔬 概率分布", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="分布: norm/binom/poisson/exp/uniform/t（norm 需均值+标准差, 二项需 n+p, 泊松需 λ, 指数需均值, 均匀需下限+上限, t 需自由度）",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.f_dist = self._field(body, "分布名", "norm")
        self.f_a = self._field(body, "参数 a", "0")
        self.f_b = self._field(body, "参数 b", "1")
        self.f_x = self._field(body, "值 x（/k）", "2")
        self.f_lo = self._field(body, "区间下限 lo", "0")
        self.f_hi = self._field(body, "区间上限 hi", "1")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("属性", self.do_props), ("CDF P(X≤x)", self.do_cdf), ("PMF/密度 X=x", self.do_pmf),
                          ("区间 P(lo<X<hi)", self.do_range), ("分位数", self.do_quant)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def _field(self, frame, label, default):
        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        e = ctk.CTkEntry(frame)
        e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        e.insert(0, default)
        return e

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _d(self):
        return self.f_dist.get().strip() or "norm"

    def do_props(self):
        try:
            self.msg(dist_props(self._d(), self.f_a.get(), self.f_b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_cdf(self):
        try:
            self.msg(dist_cdf(self._d(), self.f_x.get(), self.f_a.get(), self.f_b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_pmf(self):
        try:
            self.msg(dist_pmf(self._d(), self.f_x.get(), self.f_a.get(), self.f_b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_range(self):
        try:
            self.msg(dist_range(self._d(), self.f_lo.get(), self.f_hi.get(), self.f_a.get(), self.f_b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_quant(self):
        try:
            self.msg(dist_quantile(self._d(), self.f_x.get(), self.f_a.get(), self.f_b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")


PAGES = [ProbPage]
