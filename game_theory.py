# -*- coding: utf-8 -*-
"""博弈论与决策：纳什均衡（纯/混合）、零和博弈最大最小、决策树期望值、期望效用、社会福利函数。
AI 工具 + 页面（交叉领域，教学友好）。"""
import numpy as np
from scipy import optimize as _opt
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "game_theory",
    "name": "博弈论",
    "version": "1.0",
    "author": "zoilzo",
    "description": "博弈论与决策：纳什均衡（纯策略/混合策略）、零和博弈最大最小、决策树期望值、期望效用、社会福利（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _nums(v):
    """解析数值序列：接受逗号/空格分隔字符串或列表。"""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    s = str(v).replace("，", ",").replace(";", ",")
    return [float(x) for x in s.split(",") if x.strip()] if str(v).strip() else []


def _pv(expr):
    """解析 '0.5*100+0.5*0' / '0.3*200+0.7*-20' 为 (概率, 收益) 对列表。
    用正则稳健处理负数收益与裸数值（无 '*' 时概率视为 1）。"""
    import re as _re
    s = str(expr).replace("，", ",").replace("×", "*")
    pairs = []
    for m in _re.finditer(r"([-+]?\d*\.?\d+)\s*\*\s*([-+]?\d*\.?\d+)|([-+]?\d*\.?\d+)", s):
        if m.group(1) is not None:
            pairs.append((float(m.group(1)), float(m.group(2))))
        else:
            pairs.append((1.0, float(m.group(3))))
    return pairs


def _mtx(v):
    """解析矩阵：行以 ; 或换行分隔，列以逗号分隔。"""
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    s = str(v).replace("，", ",").replace(";", "\n")
    rows = [r.strip() for r in s.split("\n") if r.strip()]
    if not rows:
        return np.zeros((0, 0))
    return np.asarray([[float(x) for x in r.split(",") if x.strip()] for r in rows], dtype=float)


def _parse_bimatrix(text):
    """解析双矩阵博弈 'A_A1,A12,A21,A22 ; B11,B12,B21,B22' → (A, B)，均为 2x2。"""
    t = str(text or _DEF_BI).replace("，", ",").replace("；", ";")
    parts = [p.strip() for p in t.split(";") if p.strip()]
    if len(parts) < 2:
        raise ValueError("需要用 ';' 分隔玩家A与玩家B的收益，每个 4 个数（行优先 2x2）。")
    a = _nums(parts[0])
    b = _nums(parts[1])
    if len(a) != 4 or len(b) != 4:
        raise ValueError("每位玩家需 4 个收益值（2x2 行优先）。")
    return np.array(a).reshape(2, 2), np.array(b).reshape(2, 2)


# 默认：囚徒困境（双方合作 2,2；背叛占优 → 纳什在 (背叛,背叛)=1,1）
_DEF_BI = "2,0,3,1 ; 2,3,0,1"


# ------------------------------------------------------------
# 工具 1：纳什均衡
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "payoffs": {"type": "string", "description": "双矩阵 'A1..A4 ; B1..B4'（行优先 2x2）"},
        "label_a": {"type": "string", "description": "玩家A的行动名（逗号分隔，2 个）"},
        "label_b": {"type": "string", "description": "玩家B的行动名（逗号分隔，2 个）"},
    },
}, category="博弈论")
def nash_equilibrium(payoffs="", label_a="合作,背叛", label_b="合作,背叛"):
    """纳什均衡：给定 2x2 双矩阵博弈，求纯策略均衡（最优应对扫描）与混合策略均衡。
    均衡 = 没有任何一方可通过单方面改换策略而获益。"""
    A, B = _parse_bimatrix(payoffs)
    la = [x.strip() for x in str(label_a).replace("，", ",").split(",") if x.strip()]
    lb = [x.strip() for x in str(label_b).replace("，", ",").split(",") if x.strip()]
    if len(la) != 2:
        la = ["A1", "A2"]
    if len(lb) != 2:
        lb = ["B1", "B2"]
    # 纯策略 NE：cell(i,j) 满足 A[i][j] 是列 j 最大化，且 B[i][j] 是行 i 最大化
    ne = []
    for i in range(2):
        for j in range(2):
            if abs(A[i, j] - A[:, j].max()) < 1e-12 and abs(B[i, j] - B[i, :].max()) < 1e-12:
                ne.append((i, j))
    lines = [f"双矩阵博弈：A 行=玩家A（{la[0]}/{la[1]}），B 列=玩家B（{lb[0]}/{lb[1]}）", "",
             "       " + "".join("%12s" % b for b in lb)]
    for i in range(2):
        row = "%6s" % la[i]
        for j in range(2):
            row += "%12s" % ("%g,%g" % (A[i, j], B[i, j]))
        lines.append(row)
    lines.append("")
    if ne:
        lines.append("纯策略纳什均衡：")
        for i, j in ne:
            lines.append("  (%s, %s) → 收益 (%g, %g)" % (la[i], lb[j], A[i, j], B[i, j]))
    else:
        lines.append("纯策略纳什均衡：无（需考虑混合策略）。")
    # 混合策略：玩家A 以 p 选第0行，玩家B 以 q 选第0列，使对方无差异
    den_q = A[0, 0] - A[1, 0] - A[0, 1] + A[1, 1]
    den_p = B[0, 0] - B[0, 1] - B[1, 0] + B[1, 1]
    q = (A[1, 1] - A[0, 1]) / den_q if abs(den_q) > 1e-12 else None
    p = (B[1, 1] - B[1, 0]) / den_p if abs(den_p) > 1e-12 else None
    if p is not None and 0 < p < 1 and q is not None and 0 < q < 1:
        lines.append("")
        lines.append("混合策略纳什均衡：")
        lines.append("  玩家A：以 p=%.4g 选 %s，1-p=%.4g 选 %s" % (p, la[0], 1 - p, la[1]))
        lines.append("  玩家B：以 q=%.4g 选 %s，1-q=%.4g 选 %s" % (q, lb[0], 1 - q, lb[1]))
    else:
        lines.append("")
        lines.append("混合策略：该博弈无(0,1)内的非退化混合均衡（存在占优策略或纯均衡）。")
    lines.append("")
    lines.append("➤ 含义：纳什均衡是理性博弈的稳定结局；囚徒困境揭示个体理性导致的集体非最优。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：零和博弈（最大最小）
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "matrix": {"type": "string", "description": "玩家A的收益矩阵（行/列以 ; 逗号分隔）"},
    },
}, category="博弈论")
def zero_sum_game(matrix="3,1 ; 1,3"):
    """零和博弈：玩家A 收矩阵值，玩家B 付同值。用线性规划求最大最小（A 保底）与最小最大（B 压顶），
    两者相等即博弈值（冯诺依曼最小最大定理）。"""
    A = _mtx(matrix)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("矩阵不能为空。")
    m, n = A.shape
    # 行玩家A 最大化 v，满足 v <= (A p)_i 对每行（用应对最优反应）
    # 列玩家B 最小化 v，满足 v >= (p^T A)_j 对每列
    # 行玩家的 LP：max v s.t. A^T x >= v*1, sum(x)=1, x>=0
    from numpy import asarray
    ones_m = np.ones(m)
    # A 玩家：变量 (p[m], v)；目标 max v
    c = np.zeros(m + 1); c[m] = -1.0
    Aub = []; bub = []
    # v - (A^T x)_j <= 0  →  (A^T x)_j >= v  → A^T x - v >= 0
    # 写成 -A^T x + v <= 0
    for j in range(n):
        row = np.zeros(m + 1)
        row[:m] = -A[:, j]
        row[m] = 1.0
        Aub.append(row); bub.append(0.0)
    Aub.append(np.concatenate([np.ones(m), [0.0]])); bub.append(1.0)
    bounds = [(0, None)] * m + [(None, None)]
    res = _opt.linprog(c, A_ub=np.asarray(Aub), b_ub=np.asarray(bub), bounds=bounds, method="highs")
    if not res.success:
        return {"text": "线性规划未收敛：" + res.message}
    p = res.x[:m]; game_val = res.x[m]
    # 对偶/列玩家直接由行玩家最优得博弈值；若 A 是方阵可由对称 LP 得 q
    # 列玩家 LP：min v s.t. (A q)_i <= v；这里用对偶或单独求解
    c2 = np.zeros(n + 1); c2[n] = 1.0
    A2 = []; b2 = []
    for i in range(m):
        row = np.zeros(n + 1)
        row[:n] = A[i, :]
        row[n] = -1.0
        A2.append(row); b2.append(0.0)
    A2.append(np.concatenate([np.ones(n), [0.0]])); b2.append(1.0)
    b2o = [(0, None)] * n + [(None, None)]
    res2 = _opt.linprog(c2, A_ub=np.asarray(A2), b_ub=np.asarray(b2), bounds=b2o, method="highs")
    q = res2.x[:n] if res2.success else None
    val2 = res2.x[n] if res2.success else game_val
    lines = [f"零和博弈：A 收益矩阵 {m}×{n}", "", "" + "".join("%8g" % A[i, j] for j in range(n))]
    for i in range(m):
        lines.append("  " + "".join("%8g" % A[i, j] for j in range(n)))
    lines.append("")
    lines.append("玩家A最大最小策略（保底）：" + "".join("%.4g " % x for x in p) + " → 保底值 v = %.4g" % game_val)
    if q is not None:
        lines.append("玩家B最小最大策略（压顶）：" + "".join("%.4g " % x for x in q) + " → 压顶值 = %.4g" % val2)
    lines.append("")
    lines.append("➤ 最小最大定理：最大值 max_v 与 min_v 若相等，即博弈值 v*，双方有公平的混合策略。")
    lines.append("➤ 应用：对抗性决策、军事/博弈、验证实验和竞技策略的底线评估。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：决策树期望值
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "choice": {"type": "string", "description": "各方案'概率*结果+...'，用 | 分隔；如 '0.5*100+0.5*0 | 0.9*50+0.1*200'"},
        "names": {"type": "string", "description": "方案名（逗号分隔）"},
    },
}, category="博弈论")
def decision_tree_eval(choice="0.5*100+0.5*0 | 0.3*200+0.7*(-20)", names=""):
    """决策树期望值 EMV：对每个备选方案，把各可能结果的概率×收益相加，求期望值并推荐最优方案。"""
    branches = [b.strip() for b in str(choice).replace("，", ",").replace("｜", "|").split("|") if b.strip()]
    if not branches:
        raise ValueError("至少一个方案。")
    name_list = [x.strip() for x in str(names).replace("，", ",").split(",") if x.strip()] if names else []
    if len(name_list) != len(branches):
        name_list = ["方案%d" % (i + 1) for i in range(len(branches))]
    lines = ["决策树期望值（EMV）计算：", ""]
    evs = []
    for bname, br in zip(name_list, branches):
        pairs = _pv(br)
        if not pairs:
            lines.append("%s：未解析到有效结果" % bname)
            evs.append(float("-inf"))
            continue
        ev = sum(p * v for p, v in pairs)
        det = ["%g×%g" % (p, v) for p, v in pairs]
        evs.append(ev)
        lines.append("%s：%s" % (bname, " + ".join(det)))
        lines.append("  期望值 EMV = %.4g" % ev)
    best = int(np.argmax(evs))
    lines.append("")
    lines.append("➤ 推荐：%s（EMV = %.4g 最大）" % (name_list[best], evs[best]))
    lines.append("➤ 含义：EMV 以概率加权平均量化不确定性下的平均收益；风险中性者选 EMV 最大者。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：期望效用
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "gamble": {"type": "string", "description": "赌博'概率*收益+...'，如 '0.5*100+0.5*0'"},
        "gamma": {"type": "number", "description": "风险厌恶系数 γ（CARA/CRRA 幂效用）"},
        "sure": {"type": "string", "description": "无风险确定收益（留空则只算赌博）"},
    },
}, category="博弈论")
def expected_utility(gamble="0.5*100+0.5*60", gamma="0.5", sure=""):
    """期望效用：u(x)=x^(1-γ)/(1-γ)（γ=1 用 ln）。计算赌博的期望效用、确定性等价 CE 与风险溢价，
    并与无风险收益比较，判断风险厌恶方向。"""
    pairs = _pv(gamble)
    if not pairs:
        raise ValueError("赌博表达式未解析到有效结果。")
    gam = float(gamma or 0.5)

    def u(x):
        if x <= 0:
            return float("-inf")
        if abs(gam - 1.0) < 1e-12:
            return float(np.log(x))
        return float(x ** (1 - gam) / (1 - gam))

    pev = 0.0
    eu = 0.0
    det = []
    for p, v in pairs:
        pev += p * v
        eu += p * u(v)
        det.append("%g×%g" % (p, v))
    if eu == float("-inf"):
        return {"text": "赌博包含非正收益，无法用对数/幂效用（本模型要求收益>0）。"}
    # 确定性等价 CE：u(ce) = EU → ce = u^{-1}(eu)
    if abs(gam - 1.0) < 1e-12:
        ce = float(np.exp(eu))
    else:
        ce = float((eu * (1 - gam)) ** (1 / (1 - gam)))
    risk_prem = pev - ce
    lines = ["期望效用：" + " + ".join(det), ""]
    lines.append("期望收益 EV = %.4g" % pev)
    lines.append("期望效用 EU = %.4g" % eu)
    lines.append("确定性等价 CE = %.4g" % ce)
    lines.append("风险溢价 EV - CE = %.4g" % risk_prem)
    s = str(sure).strip()
    if s:
        sv = float(s)
        lines.append("")
        lines.append("无风险收益 S = %g" % sv)
        if sv > ce:
            lines.append("➤ 理性选择：选无风险 S（其效用高于赌博）。")
        elif sv < ce:
            lines.append("➤ 理性选择：选赌博（确定性等价高于无风险）。")
        else:
            lines.append("➤ 两者等价（风险中性）。")
    lines.append("")
    lines.append("➤ 含义：γ 越大越厌恶风险，确定性等价 CE 越低；CE 可把不确定赌局折算成确定的等价金额。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 5：社会福利函数
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "profiles": {"type": "string", "description": "各方案的效用向量，方案用 | 分隔，个体用逗号分隔；如 '8,4,0 | 3,3,3 | 7,1,7'"},
        "names": {"type": "string", "description": "方案名（逗号分隔）"},
    },
}, category="博弈论")
def social_welfare(profiles="8,4,0 | 3,3,3 | 7,1,7", names=""):
    """社会福利函数：对每个方案给出功利主义(总福利)、罗尔斯主义(最小者极大化)与纳什乘积(几何平均)三种评价，
    并指出各准则下的最优方案。"""
    alts = [a.strip() for a in str(profiles).replace("，", ",").replace("｜", "|").split("|") if a.strip()]
    if not alts:
        raise ValueError("至少一个方案。")
    name_list = [x.strip() for x in str(names).replace("，", ",").split(",") if x.strip()] if names else []
    if len(name_list) != len(alts):
        name_list = ["方案%d" % (i + 1) for i in range(len(alts))]
    lines = ["社会福利函数：各方案对个体的效用度量", ""]
    rows = []
    for bname, alt in zip(name_list, alts):
        vals = _nums(alt)
        if not vals:
            raise ValueError("方案 %s 没有效用值。" % bname)
        total = sum(vals)
        mn = min(vals)
        gm = float(np.prod([v for v in vals if v > 0]) ** (1.0 / len(vals))) if all(v > 0 for v in vals) else 0.0
        rows.append((bname, total, mn, gm))
        lines.append("%s：个体效用 %s" % (bname, ", ".join("%g" % v for v in vals)))
        lines.append("  功利主义 Σu = %.4g | 罗尔斯 min = %.4g | 纳什乘积几何均值 = %.4g" % (total, mn, gm))
    it = int(np.argmax([r[1] for r in rows]))
    rl = int(np.argmax([r[2] for r in rows]))
    np_ = int(np.argmax([r[3] for r in rows]))
    lines.append("")
    lines.append("➤ 功利主义最优(总福利最大)：%s" % rows[it][0])
    lines.append("➤ 罗尔斯主义最优(最弱者极大化)：%s" % rows[rl][0])
    lines.append("➤ 纳什乘积最优(几何平均最大)：%s" % rows[np_][0])
    lines.append("")
    lines.append("➤ 含义：功利主义注重总量可能牺牲弱者；罗尔斯保护最不利者；纳什乘积在公平与效率间折中。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "博弈论"
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


class GameTheoryPage(_FormPage):
    NAME = "博弈论"
    EMOJI = "\U0001F40D"
    _TITLE = "🎲 博弈论与决策"
    _TOOLS = {
        "nash_equilibrium": (
            "纳什均衡",
            "2x2 双矩阵博弈：求纯策略与混合策略纳什均衡（含囚徒困境默认例）。",
            [("payoffs", "双矩阵 'A1,A2,A3,A4 ; B1,B2,B3,B4'", "2,0,3,1 ; 2,3,0,1"),
             ("label_a", "玩家A行动名(2个)", "合作,背叛"),
             ("label_b", "玩家B行动名(2个)", "合作,背叛")],
            nash_equilibrium),
        "zero_sum_game": (
            "零和博弈·最大最小",
            "对抗性零和博弈：LP 求 A 最大最小与 B 最小最大，报博弈值与混合策略。",
            [("matrix", "A的收益矩阵(行;列)", "3,-1 ; -1,3")],
            zero_sum_game),
        "decision_tree_eval": (
            "决策树期望值",
            "各备选方案按概率加权求 EMV，推荐期望收益最大者。",
            [("choice", "方案'概率*结果+...'，用 | 分隔", "0.5*100+0.5*0 | 0.4*80+0.6*50"),
             ("names", "方案名(逗号分隔)", "")],
            decision_tree_eval),
        "expected_utility": (
            "期望效用/风险",
            "风险厌恶下的期望效用 EU、确定性等价 CE 与风险溢价，判断选赌局或无风险。",
            [("gamble", "赌博'概率*收益+...'", "0.5*100+0.5*60"),
             ("gamma", "风险厌恶系数 γ", "0.5"),
             ("sure", "无风险确定收益(留空无)", "")],
            expected_utility),
        "social_welfare": (
            "社会福利函数",
            "功利/罗尔斯/纳什乘积三种准则比较不同方案，指出各准则最优。",
            [("profiles", "各方案效用向量(方案|个体)", "8,4,0 | 3,3,3 | 7,1,7"),
             ("names", "方案名(逗号分隔)", "")],
            social_welfare),
    }


PAGES = [GameTheoryPage]