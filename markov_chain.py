# -*- coding: utf-8 -*-
"""马尔可夫链：转移矩阵 n 步概率 / 稳态分布 / 吸收链期望步数 / 概率向量迭代。

给运筹、概率统计、数学建模、经济管理同学做随机过程计算。
（AI 工具 + 页面）
"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "markov_chain",
    "name": "马尔可夫链",
    "version": "1.0",
    "author": "zoilzo",
    "description": "转移矩阵 n 步概率 / 稳态分布 / 吸收链期望步数 / 概率向量迭代（工具 + 页面）",
}


def _matrix(s):
    s = s.replace("；", ";").replace("\n", ";").replace("，", ",")
    rows = []
    for r in s.split(";"):
        r = r.strip()
        if not r:
            continue
        vals = [float(x) for x in r.replace(",", " ").split() if x.strip()]
        if vals:
            rows.append(vals)
    if not rows:
        raise ValueError("输入为空")
    return np.array(rows, dtype=float)


def _lst(s):
    s = s.replace(",", " ").replace("，", " ").replace(";", " ").replace("；", " ").replace("\n", " ")
    vals = [float(x) for x in s.split() if x.strip()]
    return vals if vals else raise_err()


def raise_err():
    raise ValueError("输入为空")


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"transition": {"type": "string", "default": "0.7,0.3;0.4,0.6"}, "steps": {"type": "number", "default": 3}}, "required": ["transition"]}, category="马尔可夫链")
def markov_stepping(transition="0.7,0.3;0.4,0.6", steps=3):
    """n 步转移概率矩阵 P^n。transition 为行随机矩阵（行和=1，分号隔行，行内逗号），steps 为步数。"""
    P = _matrix(transition)
    n = int(round(steps))
    if n < 1:
        return {"text": "步数需 ≥ 1。"}
    Pn = np.linalg.matrix_power(P, n)
    lines = ["P^{0}（{1} 步转移概率）：".format(n, n)]
    for r in Pn:
        lines.append("  [ " + ", ".join("%.4f" % v for v in r) + " ]")
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"transition": {"type": "string", "default": "0.7,0.3;0.4,0.6"}}, "required": ["transition"]}, category="马尔可夫链")
def markov_steady(transition="0.8,0.2,0.0;0.4,0.5,0.1;0.0,0.6,0.4"):
    """稳态分布 π（πP=π，Σπ=1）。transition 为行随机矩阵。返回每个状态的长期概率。"""
    P = _matrix(transition)
    rows, cols = P.shape
    if rows != cols:
        return {"text": "转移矩阵需为方阵。"}
    # 求 P^T 的模 = 1 的左特征向量，归一化
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(np.array(vals) - 1.0)))
    pi = np.real(vecs[:, k])
    total = pi.sum()
    if abs(total) < 1e-12:
        return {"text": "无法求得稳态分布（可能非常返）。"}
    pi = pi / total
    pi = np.clip(pi, 0.0, None)
    pi = pi / pi.sum()
    lines = ["稳态分布 π（共 {0} 个状态）：".format(len(pi))]
    for i, v in enumerate(pi):
        lines.append("  状态 {0}：{1:.4f} ({2:.2f}%)".format(i + 1, v, v * 100))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"transition": {"type": "string", "default": "0.7,0.3;0.4,0.6"}, "start": {"type": "string", "default": "1,0"}, "steps": {"type": "number", "default": 4}}, "required": ["transition"]}, category="马尔可夫链")
def markov_iterate(transition="0.7,0.3;0.4,0.6", start="1,0", steps=4):
    """概率向量迭代：p(0)·P^n，看分布如何演化。start 为初始概率向量，steps 为步数。"""
    P = _matrix(transition)
    p0 = np.array(_lst(start), dtype=float)
    if len(p0) != P.shape[1]:
        return {"text": "初始向量长度需等于状态数（{0}）。".format(P.shape[1])}
    p0 = p0 / (p0.sum() or 1.0)
    n = int(round(steps))
    lines = ["概率向量迭代 p(t) = p(0)·P^t："]
    p = p0
    for t in range(n + 1):
        lines.append("  t={0}: [{1}]".format(t, ", ".join("%.4f" % v for v in p)))
        p = p @ P
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"transition": {"type": "string", "default": "0.8,0.2,0.0;0.3,0.6,0.1;0.0,0.0,1.0"}, "absorbing": {"type": "string", "default": "3"}}, "required": ["transition"]}, category="马尔可夫链")
def absorption_expectation(transition="0.8,0.2,0.0;0.3,0.6,0.1;0.0,0.0,1.0", absorbing="3"):
    """吸收链：从各瞬态出发到吸收的平均步数（基本矩阵 N=(I-Q)^-1）。absorbing 为吸收态下标（从 1 计，逗号分隔）。"""
    P = _matrix(transition)
    n = P.shape[0]
    if P.shape[1] != n:
        return {"text": "转移矩阵需为方阵。"}
    abs_idx = [int(v) - 1 for v in _lst(absorbing)]
    abs_set = set(abs_idx)
    if not abs_set:
        return {"text": "请给出吸收态下标。"}
    trans = [i for i in range(n) if i not in abs_set]
    if not trans:
        return {"text": "没有瞬态，无需计算。"}
    Q = P[np.ix_(trans, trans)]
    I = np.eye(len(trans))
    N = np.linalg.pinv(I - Q)  # 基本矩阵
    expected = N.sum(axis=1)
    lines = ["吸收链期望步数（从各瞬态出发，转 = {0}，吸收 = {1}）：".format(len(trans), sorted(a + 1 for a in abs_set))]
    for i, e in enumerate(expected):
        lines.append("  状态 {0}：平均 {1:.4f} 步到达吸收".format(trans[i] + 1, e))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class MarkovChainPage(ui.BasePage):
    NAME = "马尔可夫链"
    EMOJI = "\U0001F300"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🌊 马尔可夫链", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="转移矩阵（行和=1，分号隔行，行内逗号）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.transition = ctk.CTkEntry(body)
        self.transition.insert(0, "0.7,0.3;0.3,0.7")
        self.transition.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="初始向量 / 步数 / 吸收态下标（如 3 或 2,3）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.start = ctk.CTkEntry(body, width=150); self.start.insert(0, "1,0")
        self.start.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.steps = ctk.CTkEntry(body, width=80); self.steps.insert(0, "4")
        self.steps.pack(side="left", padx=(0, ui.SPACE["sm"]))
        self.absorb = ctk.CTkEntry(body, width=100); self.absorb.insert(0, "3")
        self.absorb.pack(side="left")
        ops = [
            ("P^n 步进", lambda: self._run(markov_stepping, [self.transition.get(), self.steps.get()])),
            ("稳态分布", lambda: self._run(markov_steady, ["0.8,0.2,0.0;0.4,0.1,0.5;0.7,0.3,0.0"])),
            ("概率迭代", lambda: self._run(markov_iterate, [self.transition.get(), self.start.get(), self.steps.get()])),
            ("吸收步数", lambda: self._run(absorption_expectation, [self.transition.get(), self.absorb.get()])),
        ]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btns, text=txt, command=fn, width=96).grid(row=0, column=i, padx=4, pady=4)
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


PAGES = [MarkovChainPage]