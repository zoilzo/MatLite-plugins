# -*- coding: utf-8 -*-
"""数学优化：线性规划（单纯形）、背包（0/1+分数）、指派问题、梯度下降/黄金分割/拉格朗日乘子。
AI 工具 + 页面（教学，展示步骤与结果）。依赖 numpy/scipy/sympy。"""
import re
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

try:
    from scipy import optimize as _opt
except Exception:  # pragma: no cover
    _opt = None

try:
    import sympy as _sp
except Exception:  # pragma: no cover
    _sp = None

PLUGIN = {
    "id": "optimization_pro",
    "name": "数学优化",
    "version": "1.0",
    "author": "zoilzo",
    "description": "线性规划、背包、指派、梯度下降/黄金分割/拉格朗日乘子（工具 + 页面）",
}

_DEF_LP = "min 3x+2y ; 2x+y<=8, x+3y<=9 ; x>=0, y>=0"


def _parse_list(s, of=float):
    """把 '1,2,3' / '1 2 3' / 列表 解析为数值列表。"""
    if s is None or (isinstance(s, str) and not s.strip()):
        return []
    if isinstance(s, (list, tuple)):
        return [of(x) for x in s]
    txt = str(s).replace("，", ",").replace(";", ",")
    return [of(x) for x in re.split(r"[\s,]+", txt) if str(x).strip()]


def _coefs(expr, varset, vindex):
    """把 '2x+y' 解析为相对 varset 的系数向量。"""
    c = [0.0] * len(varset)
    expr = expr.strip()
    if not expr:
        return c
    # 分解带符号的单项
    terms = re.findall(r"[+-]?[^+-]+", expr)
    for term in terms:
        if not term:
            continue
        sign = -1.0 if term.startswith("-") else 1.0
        term = term.lstrip("+- ").strip()
        if not term:
            continue
        var = next((v for v in varset if v in term), None)
        if var is None:
            continue
        num = 1.0
        m = re.match(r"^([0-9.]+)", term)
        if m:
            num = float(m.group(1))
        c[vindex[var]] += sign * num
    return c


def _all_vars(*exprs):
    s = set()
    for e in exprs:
        for ch in e:
            if ch.isalpha():
                s.add(ch)
    return sorted(s)


def _parse_lp(text):
    """把 'max 3x+2y ; 2x+y<=8, x+3y<=9 ; x>=0,y>=0' 解析成标准式。
    返回 (c, A_ub, b_ub, bounds, sense, varset)。内层 A_ub 约束自动把 >= 转为 -行<=-b。"""
    text = (text or _DEF_LP)
    text = text.replace("maximize", "max").replace("minimize", "min")
    parts = [p for p in str(text).replace("；", ";").split(";") if p.strip()]
    if not parts:
        raise ValueError("LP 表达式为空")
    obj = parts[0].strip().lower()
    sense = "max" if obj.startswith("max") else "min"
    obj_expr = obj
    for k in ("max", "min"):
        if obj_expr.startswith(k):
            obj_expr = obj_expr[len(k):]
    obj_expr = obj_expr.replace("=", "")
    rest = " ".join(parts[1:])
    varset = _all_vars(obj_expr + " " + rest)
    if not varset:
        raise ValueError("未识别到变量（用单个字母，如 x,y）")
    vindex = {v: i for i, v in enumerate(varset)}
    c = _coefs(obj_expr, varset, vindex)
    A, b = [], []
    lb = [None] * len(varset)
    ub = [None] * len(varset)
    for con in parts[1:]:
        for piece in re.split(r"[,]", con):
            piece = piece.strip()
            if not piece:
                continue
            # 单变量边界 x>=2 / x<=5
            m = re.match(r"^([a-zA-Z])\s*(>=|<=)\s*(-?[0-9.]+)$", piece)
            if m and m.group(1) in vindex:
                idx = vindex[m.group(1)]
                val = float(m.group(3))
                if m.group(2) == ">=":
                    lb[idx] = max(lb[idx] if lb[idx] is not None else -float("inf"), val)
                else:
                    ub[idx] = min(ub[idx] if ub[idx] is not None else float("inf"), val)
                continue
            # 含多个变量的不等式 2x+y<=8 / x+3y>=9
            m = re.match(r"^(.+?)(<=|>=)(\S+)$", piece)
            if not m:
                continue
            left = m.group(1)
            rel = m.group(2)
            rhs = float(m.group(3))
            row = _coefs(left, varset, vindex)
            if rel == ">=":
                row = [-r for r in row]
                rhs = -rhs
            A.append(row)
            b.append(rhs)
    # 默认非负边界
    for i in range(len(varset)):
        if lb[i] is None:
            lb[i] = 0.0
    return np.array(c, dtype=float), (np.array(A, dtype=float) if A else None),\
        (np.array(b, dtype=float) if b else None), list(zip(lb, ub)), sense, varset


@ai_tools._reg
@ai_tools._tool({"properties": {"lp": {"type": "string"}}}, category="优化")
def optimize_lp(lp=""):
    """线性规划（scipy 单纯形/内点）。lp 形如 'max 3x+2y ; 2x+y<=8, x+3y<=9 ; x>=0,y>=0'。返回最优解与值。"""
    if _opt is None:
        return {"text": "未安装 scipy，无法求解线性规划。"}
    try:
        c, A, b, bounds, sense, varset = _parse_lp(lp)
    except Exception as e:
        return {"text": f"LP 解析出错：{e}"}
    cs = -c if sense == "max" else c
    res = _opt.linprog(cs, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        return {"text": f"无最优解：{res.message}"}
    val = -res.fun if sense == "max" else res.fun
    lines = ["线性规划 " + ("最大化" if sense == "max" else "最小化") + "："]
    for i, v in enumerate(varset):
        lines.append(f"  {v} = {res.x[i]:g}")
    lines.append(f"  最优目标值 = {val:g}")
    lines.append(f"  求解信息：{res.message}")
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"values": {"type": "string"}, "weights": {"type": "string"}, "capacity": {"type": "number"}, "mode": {"type": "string"}}}, category="优化")
def optimize_knapsack(values="60,100,120", weights="10,20,30", capacity=50, mode="01"):
    """背包问题。mode: 01（0/1 背包，动态规划）或 fract（分数背包，贪心）。返回选择与最大价值。"""
    v = _parse_list(values)
    w = _parse_list(weights)
    cap = float(capacity)
    if len(v) != len(w) or not v:
        return {"text": "values 与 weights 需给出等长非空序列。"}
    items = [(str(i + 1), v[i], w[i]) for i in range(len(v))]
    if (mode or "01") == "fract":
        order = sorted(items, key=lambda it: it[1] / it[2], reverse=True)
        rem = cap
        tot = 0.0
        lines = ["分数背包（贪心，容量 {0:g}）：".format(cap)]
        for name, val, wt in order:
            if wt <= rem:
                frac = 1.0
            elif rem > 0:
                frac = rem / wt
            else:
                frac = 0.0
            if frac > 0:
                lines.append("  物品 {0}（价值 {1:g}，重量 {2:g}）：取 {3:g}".format(name, val, wt, frac))
                tot += val * frac
                rem -= wt * frac
            if rem <= 0:
                break
        lines.append("  总价值 = {0:g}".format(tot))
        return {"text": "\n".join(lines)}
    # 0/1 动态规划
    n = len(v)
    Wc = int(cap)
    dp = [[0] * (Wc + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ww = int(w[i - 1])
        vv = int(v[i - 1])
        for c in range(Wc + 1):
            if ww <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - ww] + vv)
            else:
                dp[i][c] = dp[i - 1][c]
    lines = ["0/1 背包（动态规划，容量 {0:g}）：".format(cap)]
    sel = []
    c = Wc
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            sel.append(items[i - 1])
            c -= int(w[i - 1])
    for name, val, wt in sel:
        lines.append("  选物品 {0}（价值 {1:g}，重量 {2:g}）".format(name, val, wt))
    lines.append("  总价值 = {0}".format(dp[n][Wc]))
    lines.append("  装填表最后一行：" + " ".join(str(x) for x in dp[n]))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"cost": {"type": "string"}, "rows": {"type": "string"}, "cols": {"type": "string"}}}, category="优化")
def optimize_assignment(cost="1 2 3 2 1 4 1 1 2", rows="", cols=""):
    """指派问题（最小化总成本，匈牙利法 via scipy）。cost 为行优先矩阵。返回指派与总成本。"""
    if _opt is None:
        return {"text": "未安装 scipy。"}
    vals = _parse_list(cost)
    if not vals:
        return {"text": "cost 为空。"}
    n = int(round(len(vals) ** 0.5))
    if n * n == len(vals):
        m = n
    else:
        try:
            n = int(rows)
            m = int(cols)
        except (TypeError, ValueError):
            return {"text": "cost 不是方阵，请用 rows/cols 指定行数与列数。"}
    if n * m != len(vals):
        return {"text": "cost 元素个数与 matrix 尺寸不符。"}
    C = np.array(vals).reshape(n, m)
    r, c = _opt.linear_sum_assignment(C)
    lines = ["指派问题（最小总成本）："]
    tot = 0.0
    for i, j in zip(r, c):
        lines.append("  工人/行 {0} → 任务/列 {1}  成本 {2:g}".format(i + 1, j + 1, C[i, j]))
        tot += C[i, j]
    lines.append("  总成本 = {0:g}".format(tot))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"mode": {"type": "string"}, "f": {"type": "string"}, "x0": {"type": "string"}}}, category="优化")
def optimize_numeric(mode="gradient", f="x**2 + 3*x + 2", x0=""):
    """数值优化。mode: gradient（梯度下降/BFGS）、golden（一维黄金分割）、lagrange（拉格朗日乘子）。
    f 用 python 表达式。golden 用单变量 x；gradient 默认 2 变量 x,y；lagrange 用 'f=expr, g=expr'。"""
    if mode == "lagrange":
        return _optimize_lagrange(f)
    if _opt is None:
        return {"text": "未安装 scipy，无法做数值优化。"}
    import math
    if mode == "golden":
        fib = lambda t: eval(f, {"x": t, "math": math})
        a, b = -5.0, 5.0
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        for _ in range(100):
            if fib(c) < fib(d):
                b = d
            else:
                a = c
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            if abs(b - a) < 1e-6:
                break
        xm = (a + b) / 2
        lines = ["一维黄金分割最小化 f(x) = {0}：".format(f)]
        lines.append("  收敛区间 ≈ [{0:g}, {1:g}]".format(a, b))
        lines.append("  极小点 x* ≈ {0:g}，f(x*) ≈ {1:g}".format(xm, fib(xm)))
        lines.append("  迭代后区间宽 {0:.3g}".format(abs(b - a)))
        return {"text": "\n".join(lines)}
    # 梯度下降 / BFGS 多元
    x0v = _parse_list(x0) if x0 and x0.strip() else [2.0, 2.0]
    varz = _fvar_names(f, len(x0v))
    local = dict(math=math, np=np)
    local.update(zip(varz, x0v))
    x0a = np.array(x0v, dtype=float)
    res = _opt.minimize(lambda X: eval(f, {**local, **dict(zip(varz, X))}), x0a, method="BFGS")
    if not res.success:
        return {"text": "求解未收敛：{0}".format(res.message)}
    lines = ["梯度下降/BFGS 最小化 f = {0}：".format(f)]
    lines.append("  初始点 = {0}".format(list(x0v)))
    for i, v in enumerate(varz):
        lines.append("    {0} = {1:g}".format(v, res.x[i]))
    lines.append("  最优值 f* = {0:g}".format(res.fun))
    lines.append("  迭代次数 = {0}".format(res.nit))
    return {"text": "\n".join(lines)}


def _fvar_names(f_expr, n):
    """近似提取 f 的变量名（字母）。"""
    varz = sorted(set(re.findall(r"[a-zA-Z]", f_expr)))
    if "x" in varz:
        varz.remove("x")
        varz.insert(0, "x")
    return varz[:n] or ["x"]


def _optimize_lagrange(f):
    """用 sympy 做拉格朗日乘子演示（2 变量 + 1 约束）。f 形如 'f=x+y, g=x^2+y^2-1'。"""
    if _sp is None:
        return {"text": "未安装 sympy，无法做拉格朗日乘子。"}
    parts = [p.strip() for p in str(f).replace("，", ",").replace("；", ";").split(",") if p.strip()]
    if len(parts) < 2:
        return {"text": "请给 'f=expr, g=expr'，如 'f=x+y, g=x^2+y^2-1'。"}
    fs = parts[0].split("=", 1)[-1]
    gs = parts[1].split("=", 1)[-1]
    x, y, lam = _sp.symbols("x y lam")
    try:
        F = _sp.sympify(fs)
        G = _sp.sympify(gs)
    except Exception as e:
        return {"text": "表达式解析出错：{0}".format(e)}
    L = F + lam * G
    lines = ["拉格朗日乘子：优化 f = {0}，约束 g = {1} = 0".format(fs, gs)]
    lines.append("  L(x,y,λ) = {0} + λ·({1})".format(F, G))
    eqx = _sp.diff(L, x)
    eqy = _sp.diff(L, y)
    eqg = _sp.diff(L, lam)
    lines.append("  ∂L/∂x = {0}".format(eqx))
    lines.append("  ∂L/∂y = {0}".format(eqy))
    lines.append("  ∂L/∂λ = {0}".format(eqg))
    # 先由前两式把 x,y 用 λ 表出，再代入第三式求 λ，从而得到驻点
    pts = []
    try:
        sol_xy = _sp.solve([eqx, eqy], (x, y), dict=True)
    except Exception:
        sol_xy = []
    for s in sol_xy:
        e = _sp.simplify(eqg.subs(s))
        for lamv in _sp.solve(e, lam):
            xv = _sp.simplify(s[x].subs(lam, lamv))
            yv = _sp.simplify(s[y].subs(lam, lamv))
            pts.append((xv, yv, _sp.simplify(lamv)))
    if not pts:
        sols = _sp.solve([eqx, eqy, eqg], (x, y, lam), dict=True)
        for s in sols:
            pts.append((_sp.simplify(s[x]), _sp.simplify(s[y]), _sp.simplify(s[lam])))
    if not pts:
        lines.append("  无解析驻点解（可改用梯度下降求数值解）。")
        return {"text": "\n".join(lines)}
    for xv, yv, lam_v in pts:
        val = _sp.simplify(F.subs({x: xv, y: yv}))
        lines.append("  驻点 (x,y) = ({0}, {1})，λ = {2}，f = {3}".format(xv, yv, lam_v, val))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class OptimizationProPage(ui.BasePage):
    NAME = "数学优化"
    EMOJI = "\U0001F4C8"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📈 数学优化", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="线性规划 LP：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.lp_entry = ctk.CTkEntry(body)
        self.lp_entry.insert(0, _DEF_LP)
        self.lp_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ctk.CTkLabel(body, text="背包：价值 / 重量 / 容量", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
        self.kv_entry = ctk.CTkEntry(body)
        self.kv_entry.insert(0, "60,100,120")
        self.kv_entry.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        self.kw_entry = ctk.CTkEntry(body)
        self.kw_entry.insert(0, "10,20,30")
        self.kw_entry.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        self.kc_entry = ctk.CTkEntry(body)
        self.kc_entry.insert(0, "50")
        self.kc_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ctk.CTkLabel(body, text="指派 cost 矩阵（行优先）：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
        self.ass_entry = ctk.CTkEntry(body)
        self.ass_entry.insert(0, "1 2 3 2 1 4 1 1 2")
        self.ass_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ctk.CTkLabel(body, text="数值优化 f（golden 用 x；gradient 用 x,y；lagrange 用 'f=expr,g=expr'）：",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w")
        self.num_entry = ctk.CTkEntry(body)
        self.num_entry.insert(0, "x**2 + 3*x + 2")
        self.num_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, cmd) in enumerate([
            ("求解线性规划", lambda: self.run("lp")),
            ("背包 0/1", lambda: self.run("kp01")),
            ("背包 分数", lambda: self.run("kpf")),
            ("指派问题", lambda: self.run("ass")),
            ("黄金分割", lambda: self.run("golden")),
            ("拉格朗日", lambda: self.run("lagrange")),
        ]):
            btns.grid_columnconfigure(i % 2, weight=1)
            r, c = divmod(i, 2)
            ctk.CTkButton(btns, text=txt, height=h, command=cmd).grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入表达式，点对应按钮计算。")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def run(self, which):
        try:
            if which == "lp":
                r = optimize_lp(self.lp_entry.get())
            elif which == "kp01":
                r = optimize_knapsack(self.kv_entry.get(), self.kw_entry.get(), float(self.kc_entry.get()), "01")
            elif which == "kpf":
                r = optimize_knapsack(self.kv_entry.get(), self.kw_entry.get(), float(self.kc_entry.get()), "fract")
            elif which == "ass":
                r = optimize_assignment(self.ass_entry.get())
            elif which == "golden":
                r = optimize_numeric("golden", self.num_entry.get())
            else:
                r = optimize_numeric("lagrange", self.num_entry.get())
        except Exception as e:
            self.msg(f"出错：{e}")
            return
        self.msg(r.get("text", str(r)))


PAGES = [OptimizationProPage]
