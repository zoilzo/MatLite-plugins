# -*- coding: utf-8 -*-
"""运筹学基础：运输问题(最小费用) / 指派问题(匈牙利法) / 线性规划(单纯形)。

给物流、管理、工业工程、经济专业的同学做数学建模与运筹求解。
（AI 工具 + 页面）
"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "operations_research",
    "name": "运筹学基础",
    "version": "1.0",
    "author": "zoilzo",
    "description": "运输问题(最小费用) / 指派问题(匈牙利法) / 线性规划(单纯形)（工具 + 页面）",
}


def _rows(s):
    s = s.replace("；", ";").replace("\n", ";").replace("，", ",")
    rows = []
    for r in s.split(";"):
        r = r.strip()
        if not r:
            continue
        vals = [float(x) for x in r.replace(",", " ").split() if x.strip()]
        if vals:
            rows.append(vals)
    return np.array(rows, dtype=float)


def _lst(s):
    s = s.replace(",", " ").replace("，", " ").replace(";", " ").replace("；", " ").replace("\n", " ")
    vals = [float(x) for x in s.split() if x.strip()]
    if not vals:
        raise ValueError("输入为空")
    return vals


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"supply": {"type": "string", "default": "30,50,40"}, "demand": {"type": "string", "default": "40,30,30,20"}, "costs": {"type": "string", "default": "8,6,10,9;9,12,13,7;14,9,16,5"}}, "required": ["supply", "demand", "costs"]}, category="运筹学")
def transportation_problem(supply="30,50,40", demand="40,30,30,20", costs="8,6,10,9;9,12,13,7;14,9,16,5"):
    """运输问题：最小总运费。supply 为各产地供应量，demand 为各销地需求量，costs 为单位运价矩阵（行=产地，列=销地）。用线性规划求解。"""
    from scipy.optimize import linprog
    s = np.array(_lst(supply), dtype=float)
    d = np.array(_lst(demand), dtype=float)
    C = _rows(costs)
    m, k = C.shape
    if len(s) != m or len(d) != k:
        return {"text": "supply/demand 长度需与 cost 矩阵行列一致：需 {0} 个产地、{1} 个销地。".format(m, k)}
    if abs(s.sum() - d.sum()) > 1e-6:
        return {"text": "总供应 {0} ≠ 总需求 {1}，需先平衡（增设虚拟产地/销地）。".format(s.sum(), d.sum())}
    nvars = m * k
    c = C.reshape(-1)
    # 等式约束：产地供应和、销地需求和
    A_eq = np.zeros((m + k, nvars))
    for i in range(m):
        for j in range(k):
            A_eq[i, i * k + j] = 1.0
    for j in range(k):
        for i in range(m):
            A_eq[m + j, i * k + j] = 1.0
    b_eq = np.concatenate([s, d])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return {"text": "求解失败：" + str(res.message)}
    lines = ["运输问题最优解（总运费 = {0:.2f}）：".format(res.fun)]
    alloc = res.x.reshape(m, k)
    for i in range(m):
        rows_str = []
        for j in range(k):
            a = alloc[i, j]
            rows_str.append("{0:>6.1f}".format(a))
        lines.append("  产地{0}: [{1}]".format(i + 1, " ".join(rows_str)))
    lines.append("  满足度检查：产地合计 {0}，销地合计 {1}".format(np.round(alloc.sum(axis=1), 4).tolist(), np.round(alloc.sum(axis=0), 4).tolist()))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"costs": {"type": "string", "default": "8,4,2,6;6,9,3,7;5,7,8,9"}}, "required": ["costs"]}, category="运筹学")
def assignment_problem(costs="8,4,2,6;6,9,3,7;5,7,8,9"):
    """指派问题（匈牙利法/最小总成本分配）：costs 为各任务分配给各人的成本矩阵（行列数可不等，自动补0）。返回最优指派与总成本。"""
    from scipy.optimize import linear_sum_assignment
    C = _rows(costs)
    m, k = C.shape
    if m == k:
        costmat = C
        rows, cols = linear_sum_assignment(costmat)
    else:
        # 补零使方阵，再取前 min 行/列
        pad = int(abs(m - k))
        big = max(m, k)
        costmat = np.zeros((big, big))
        costmat[:m, :k] = C
        costmat[m:, :] = 1e9
        costmat[:, k:] = 1e9
        rows, cols = linear_sum_assignment(costmat)
        rows, cols = rows[:min(m, k)], cols[:min(m, k)]
    total = 0.0
    lines = ["指派问题最优解："]
    for i, j in zip(rows, cols):
        if i < m and j < k:
            total += float(C[i, j])
            lines.append("  任务{0} → 人{1}（成本 {2}）".format(j + 1, i + 1, float(C[i, j])))
    lines.append("  最小总成本 = {0:.2f}".format(total))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"c": {"type": "string", "default": "3,2"}, "a_ub": {"type": "string", "default": "1,1;2,1"}, "b_ub": {"type": "string", "default": "4,5"}, "maximize": {"type": "boolean", "default": False}}, "required": ["c", "a_ub", "b_ub"]}, category="运筹学")
def lp_solve(c="3,2", a_ub="1,1;2,1", b_ub="4,5", maximize=False):
    """线性规划 s.t. A x ≤ b, x ≥ 0。c 为目标系数，a_ub 为约束系数矩阵（行=不等式），b_ub 为右端项。maximize=True 时求最大；否则求最小。用单纯形法求解。"""
    from scipy.optimize import linprog
    try:
        obj = np.array(_lst(c), dtype=float)
    except Exception:
        return {"text": "c 应为数字列表，如 3,4。"}
    A = _rows(a_ub)
    b = np.array(_lst(b_ub), dtype=float)
    if obj.shape[0] != A.shape[1]:
        return {"text": "c 长度需等于变量数（A 的列数，当前 {0}）。".format(A.shape[1])}
    if len(b) != A.shape[0]:
        return {"text": "b 的长度需等于 A 的行数。"}
    obj_use = -obj if maximize else obj
    res = linprog(obj_use, A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if not res.success:
        return {"text": "求解失败：" + str(res.message)}
    best = -res.fun if maximize else res.fun
    lines = ["线性规划最优解（{0} z = {1:.4f}）：".format("max" if maximize else "min", best)]
    for i, v in enumerate(res.x):
        lines.append("  x{0} = {1:.4f}".format(i + 1, v))
    # 检查影子约束
    slack = b - A @ res.x
    lines.append("  剩余/松弛量：{0}".format(np.round(slack, 4).tolist()))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class OperationsResearchPage(ui.BasePage):
    NAME = "运筹学基础"
    EMOJI = "\U0001F4E6"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📦 运筹学基础", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="运输（供应/需求/成本）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.supply = ctk.CTkEntry(body, width=140); self.supply.insert(0, "30,50,40")
        self.supply.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.demand = ctk.CTkEntry(body, width=140); self.demand.insert(0, "40,30,30,20")
        self.demand.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.costs = ctk.CTkEntry(body); self.costs.insert(0, "8,6,10,9;9,12,13,7;14,9,16,5"); self.costs.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="指派成本矩阵：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.assign = ctk.CTkEntry(body); self.assign.insert(0, "8,4,2,6;6,9,3,7;5,7,8,9"); self.assign.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="LP：目标 x 系数（c） / 约束A / 约束b", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.c = ctk.CTkEntry(body, width=90); self.c.insert(0, "3,4")
        self.c.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.a = ctk.CTkEntry(body, width=140); self.a.insert(0, "2,3;4,1")
        self.a.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.b = ctk.CTkEntry(body, width=80); self.b.insert(0, "24,5")
        self.b.pack(side="left", padx=(0, ui.SPACE["sm"]))
        ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(0, 0))
        ops = [
            ("运输问题", lambda: self._run(transportation_problem, [self.supply.get(), self.demand.get(), self.costs.get()])),
            ("指派问题", lambda: self._run(assignment_problem, [self.assign.get()])),
            ("线性规划", lambda: self._run(lp_solve, [self.c.get(), self.a.get(), self.b.get()])),
        ]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btns, text=txt, command=fn, width=100).grid(row=0, column=i, padx=4, pady=4)
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


PAGES = [OperationsResearchPage]