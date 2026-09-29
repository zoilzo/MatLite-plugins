# -*- coding: utf-8 -*-
"""级数与展开：幂级数收敛性判别、Taylor/Maclaurin 展开、递推求通项、裂项求和、Fourier 系数。AI 工具 + 页面。"""
import sympy as sp
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "series_pro",
    "name": "级数与展开",
    "version": "1.0",
    "author": "zoilzo",
    "description": "级数收敛判定（比值/根值）、Taylor/Maclaurin 展开、递推求通项、裂项求和与 Fourier 系数（工具 + 页面）",
}

_VAR_ALIASES = {"n": sp.Symbol("n"), "k": sp.Symbol("k"), "N": sp.Symbol("N")}
_VAR_CLASS = {"末项": "ratio", "常规": "ratio", "收敛": "ratio"}


def _sym(s):
    return sp.sympify(str(s).replace("^", "**"))


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "term": {"type": "string", "description": "通项 a_n（用 n 表示），如 1/n**2 或 x**n/n 或 n**(1/2)"},
        "n": {"type": "string", "description": "通项自变量，默认 n"},
        "test": {"type": "string", "description": "检验法：ratio（比值法）或 root（根值法）"},
    },
    "required": [],
}, category="级数与展开")
def series_converge(term="1/n**2", n="n", test="ratio"):
    """幂级数收敛性判别：计算比值法或根值法的极限并给出收敛区间/半径结论。"""
    nn = sp.Symbol(str(n))
    t = _sym(term)
    method = (str(test) or "ratio").lower()
    if method in ("root", "根值", "root-test"):
        lim = sp.limit(sp.Abs(t) ** (1 / nn), nn, sp.oo)
    else:
        lim = sp.limit(sp.Abs(t.subs(nn, nn + 1) / t), nn, sp.oo)
    L = sp.simplify(lim)
    txt = [f"通项 a_n = {sp.latex(t)}，自变量 {n}", f"{'根值' if method in ('root', '根值', 'root-test') else '比值'}法极限 L = lim → {sp.latex(L)} ≈ {complex(L.evalf()):.6g}"]
    free = L.free_symbols - {nn}
    if not free:
        try:
            v = float(L.evalf())
            if abs(v) < 1e-12:
                txt.append("L = 0：级数处处绝对收敛（收敛半径无穷大）。")
            elif abs(v) > 1:
                txt.append("L > 1：级数发散（仅可能 x=0 处收敛）。")
            else:
                txt.append("L = 1：比值/根值法失效，需换用其他方法（如 p 级数、积分判别）。")
        except Exception:
            txt.append("极限为符号表达式，请代入数值判断。")
    else:
        txt.append("极限含其余符号（如 x），一般该级数是 x 的幂级数，收敛半径 R = 1/L（相对该变量）。")
    txt.append(f"注意：判别法得到的是比值/根值极限，收敛半径常借此反推。")
    return {"text": "\n".join(txt)}


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "expr": {"type": "string", "description": "要展开的函数，如 exp(x) 或 1/(1-x) 或 sin(x)"},
        "var": {"type": "string", "description": "展开变量，默认 x"},
        "at": {"type": "string", "description": "展开中心，默认 0（即 Maclaurin 级数）"},
        "order": {"type": "integer", "description": "展开阶数，默认 6"},
    },
    "required": [],
}, category="级数与展开")
def series_taylor(expr="exp(x)", var="x", at=0, order=6):
    """把函数在某点展开成幂级数（Taylor/Maclaurin），返回前若干项。"""
    x = sp.Symbol(str(var))
    f = _sym(expr)
    x0 = _sym(at) if str(at).strip() not in ("", "0") else sp.S(0)
    o = max(1, int(order))
    s = sp.series(f, x, x0, o + 1).removeO()
    kind = "Maclaurin" if float(x0.evalf()) == 0 else f"Taylor（点 {sp.latex(x0)}）"
    return {"text": f"{kind} 展开 y = {sp.latex(f)}（前 {o} 阶，含各阶导）\n= {sp.latex(s)}\n一般写为：\n{sp.pretty(s)}"}


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "eq": {"type": "string", "description": "递推关系，如 f(n) - f(n) + 2*f(n-1) + f(n-2) = 0（用 f 表示未知函数，n 为指标）"},
        "func": {"type": "string", "description": "未知函数名，默认 f"},
        "n": {"type": "string", "description": "指标变量，默认 n"},
        "ic": {"type": "string", "description": "初始条件，如 f(0)=1,f(1)=1；可留空"},
    },
    "required": [],
}, category="级数与展开")
def series_recurrence(eq="f(n) - f(n-1) - f(n-2) = 0", func="f", n="n", ic="f(0)=0,f(1)=1"):
    """求解线性递推关系（sympy rsolve），给出通项表达式。"""
    fc = sp.Function(str(func))
    nn = sp.Symbol(str(n))
    l = str(eq).strip().replace("^", "**")
    if "=" in l:
        lhs, rhs = l.split("=", 1)
        rel = _sym(lhs) - _sym(rhs)
    else:
        rel = _sym(l)
    # 把独立函数符号 f 替换成 f(n)（sympy 解析 f(n) 时已是 Function 实例，此步仅兜底）
    rel = rel.replace(lambda q: isinstance(q, sp.Symbol) and q.name == str(func), lambda q: fc(nn))
    init = {}
    if ic and ic.strip():
        for part in str(ic).replace("，", ",").split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                kk = k.strip()
                idx_s = kk[kk.find("(") + 1: kk.find(")")]
                if idx_s != "" and idx_s.isdigit():
                    init[fc(sp.Integer(int(idx_s)))] = _sym(v)
    ans = sp.rsolve(rel, fc(nn), init if init else None)
    if ans is None:
        return {"text": "未能求出闭式通项（可能递推非线性或需更长初值）。"}
    return {"text": f"递推 {sp.latex(rel)} = 0\n通项 f({n}) = {sp.latex(ans)}\n{sp.pretty(ans)}"}


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "term": {"type": "string", "description": "通项 a_n，如 1/(n*(n+1)) 或 1/(n**2-1)，n 为指标"},
        "n": {"type": "string", "description": "指标变量，默认 n"},
        "lo": {"type": "integer", "description": "求和下限，默认 1"},
        "hi": {"type": "string", "description": "求和上限，默认无穷大 oo；给数字则算有限项和"},
    },
    "required": [],
}, category="级数与展开")
def series_telescoping(term="1/(n*(n+1))", n="n", lo=1, hi=""):
    """裂项相消/部分分式求和：把通项拆成部分分式，并求 1..N（或到无穷）的和。"""
    nn = sp.Symbol(str(n))
    t = _sym(term)
    apart_t = sp.apart(t, nn)
    low = int(lo)
    if str(hi).strip() in ("", "oo", "inf", "无穷"):
        s = sp.summation(t, (nn, low, sp.oo))
        hi_desc = "∞"
    else:
        up = int(float(_sym(hi).evalf()))
        s = sp.summation(t, (nn, low, up))
        hi_desc = str(up)
    txt = [f"通项 a_n = {sp.latex(t)}", f"部分分式分解：{sp.latex(apart_t)}"]
    if sp.simplify(apart_t - t) != 0:
        txt.append("（已化简通项便于裂项）")
    txt.append(f"和 S = Σ a_n（n 从 {low} 到 {hi_desc}）\n= {sp.latex(s)}")
    try:
        v = complex(s.evalf())
        if abs(v.imag) < 1e-12:
            txt.append(f"≈ {v.real:.8g}")
    except Exception:
        pass
    return {"text": "\n".join(txt)}


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "func": {"type": "string", "description": "被展开的周期函数，如 f(t) = t（默认角频率 omega=1）"},
        "var": {"type": "string", "description": "函数变量，默认 t"},
    },
    "required": [],
}, category="级数与展开")
def series_fourier(func="t", var="t"):
    """计算周期函数的 Fourier 级数系数 a0/a_k/b_k（角频率 omega = 1）。"""
    x = sp.Symbol(str(var))
    f = _sym(func)
    n = sp.Symbol("n")
    fs = sp.fourier_series(f, (x, -sp.pi, sp.pi))
    a0 = fs.a0
    aks, bks = [], []
    for k in range(1, 5):
        try:
            aks.append(fs.coeff_a(k))
            bks.append(fs.coeff_b(k))
        except Exception:
            pass
    txt = [f"周期函数 f({var}) = {sp.latex(f)}（在 [-π, π] 上）",
           f"Fourier 级数 = {sp.latex(fs.truncate(5))}",
           f"常数项 a0 ≈ {sp.latex(sp.simplify(a0))}"]
    if aks:
        txt.append("前 4 阶系数 a_k（余弦项）：" + ", ".join(sp.latex(sp.simplify(a)) for a in aks))
    if bks:
        txt.append("前 4 阶系数 b_k（正弦项）：" + ", ".join(sp.latex(sp.simplify(b)) for b in bks))
    return {"text": "\n".join(txt)}


class SeriesPage(ui.BasePage):
    NAME = "级数与展开"
    EMOJI = "\U0001F339"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🌹 级数与展开", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="输入含变量（默认 x）的函数，展开成 Taylor/Maclaurin 级数。",
                     font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["xs"], ui.SPACE["sm"]))
        self.expr_entry = ctk.CTkEntry(body, placeholder_text="如 exp(x) 或 1/(1-x) 或 sin(x)")
        self.expr_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="阶数：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w")
        self.ord_var = ctk.StringVar(value="6")
        ctk.CTkEntry(row, textvariable=self.ord_var, width=90).grid(row=0, column=1, sticky="w")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(body, text="展开 Taylor 级数", height=h, command=self.run_taylor).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ctk.CTkButton(body, text="裂项求和", height=h, fg_color="gray38", command=self.run_telescope).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(ui.SPACE["sm"], 0))
        self.msg("示例：输入 exp(x)，阶数 6，点「展开 Taylor 级数」。")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def run_taylor(self):
        expr = self.expr_entry.get().strip() or "exp(x)"
        try:
            r = series_taylor(expr, order=int(self.ord_var.get() or 6))
            self.msg(r["text"])
        except Exception as e:
            self.msg(f"展开出错：{e}")

    def run_telescope(self):
        expr = self.expr_entry.get().strip() or "1/(n*(n+1))"
        try:
            r = series_telescoping(expr, n="n")
            self.msg(r["text"])
        except Exception as e:
            self.msg(f"裂项求和出错：{e}")


PAGES = [SeriesPage]