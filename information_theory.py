# -*- coding: utf-8 -*-
"""信息论：香农熵、二元熵、KL 散度、互信息、交叉熵。AI 工具 + 页面。"""
import customtkinter as ctk
import math
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "information_theory",
    "name": "信息论",
    "version": "1.0",
    "author": "zoilzo",
    "description": "香农熵/二元熵、KL 散度、互信息、交叉熵（工具 + 页面）",
}


def _f(v, d):
    """稳健数值：容忍 None/字符串，失败用默认值。"""
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


def _fmt(x):
    """数值显示。"""
    x = float(x)
    if abs(x) >= 1e6 or (abs(x) < 1e-4 and x != 0):
        return f"{x:.4g}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _vec(s):
    """解析数值向量（逗号或空格分隔）。"""
    return [_f(x, 0.0) for x in str(s).replace(";", ",").replace(" ", ",").split(",") if str(x).strip() != ""]


def _mat(s):
    """解析矩阵（行以分号分隔，行内以逗号/空格分隔）为 numpy 数组。"""
    rows = []
    for r0 in str(s).strip().split(";"):
        parts = [p for p in r0.replace(",", " ").split() if p]
        if parts:
            rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError("矩阵为空")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("矩阵各行长度不一致")
    return np.asarray(rows, dtype=float)


@ai_tools._reg
@ai_tools._tool({"properties": {"probs": {"type": "string"}}, "required": []}, category="信息论")
def shannon_entropy(probs="0.5,0.5"):
    """香农熵：H(X) = -Σ p·log2(p)，单位为比特(bit)。输入概率向量（应归一化）。"""
    p = np.array(_vec(probs))
    if p.size == 0:
        return {"text": "概率向量为空。"}
    p = p / p.sum()
    h = -float(np.sum(p * np.log2(p)))
    return {"text": "香农熵 H = %.4f bit（%d 个事件）" % (h, p.size)}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number"}}, "required": []}, category="信息论")
def binary_entropy(p=0.5):
    """二元熵：H(p, 1-p) = -p·log2(p)-(1-p)·log2(1-p)，单位比特。"""
    p = min(1.0, max(0.0, _f(p, 0.5)))
    q = 1 - p
    h = 0.0
    if p > 0:
        h += -p * math.log2(p)
    if q > 0:
        h += -q * math.log2(q)
    return {"text": "二元熵 H2(p=%.4f) = %.4f bit" % (p, h)}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "string"}, "q": {"type": "string"}}, "required": []}, category="信息论")
def kl_divergence(p="0.5,0.5", q="0.4,0.6"):
    """KL 散度 D(P‖Q) = Σ p·log2(p/q)，单位为比特。用于衡量两个分布差异。"""
    pp = np.array(_vec(p))
    qq = np.array(_vec(q))
    if pp.size != qq.size or pp.size == 0:
        return {"text": "两个概率向量长度不一致。"}
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    total = 0.0
    for pv, qv in zip(pp, qq):
        if pv > 0 and qv > 0:
            total += pv * math.log2(pv / qv)
    return {"text": "KL 散度 D(P‖Q) = %.4f bit" % total}


@ai_tools._reg
@ai_tools._tool({"properties": {"joint": {"type": "string"}}, "required": []}, category="信息论")
def mutual_information(joint="0.1,0.4;0.33,0.07;0.42,0.0"):
    """互信息 I(X;Y)：由联合分布矩阵（行=X，列=Y）计算，单位比特，衡量两变量依赖程度。"""
    j = _mat(joint)
    if j.size == 0:
        return {"text": "联合分布矩阵为空。"}
    j = j / j.sum()
    px = j.sum(axis=1)
    py = j.sum(axis=0)
    mi = 0.0
    for i in range(j.shape[0]):
        for k in range(j.shape[1]):
            if j[i, k] > 0:
                mi += j[i, k] * math.log2(j[i, k] / (px[i] * py[k]))
    return {"text": "互信息 I(X;Y) = %.4f bit（X 有 %d 个状态，Y 有 %d 个状态）" % (mi, j.shape[0], j.shape[1])}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "string"}, "q": {"type": "string"}}, "required": []}, category="信息论")
def cross_entropy(p="0.5,0.5", q="0.4,0.6"):
    """交叉熵 H(P,Q) = -Σ p·log2(q)：用分布 Q 编码分布 P 的平均代价，单位比特。"""
    pp = np.array(_vec(p))
    qq = np.array(_vec(q))
    if pp.size != qq.size or pp.size == 0:
        return {"text": "两个概率向量长度不一致。"}
    pp = pp / pp.sum()
    qq = qq / qq.sum()
    total = 0.0
    for pv, qv in zip(pp, qq):
        if pv > 0 and qv > 0:
            total += -pv * math.log2(qv)
    return {"text": "交叉熵 H(P,Q) = %.4f bit" % total}


class InfoTheoryPage(ui.BasePage):
    NAME = "信息论"
    EMOJI = "\U0001F5C4\U0000FE0F"  # 🗄️

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🗄️ 信息论", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="概率用逗号分隔；互信息联合分布用分号分、行内逗号，如 0.1,0.4;0.4,0.1",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.p_ = self._box(body, "分布 P / 联合矩阵")
        self.p_.insert("1.0", "0.5,0.5")
        self.q_ = self._box(body, "分布 Q")
        self.q_.insert("1.0", "0.4,0.6")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("香农熵", self.do_ent), ("二元熵", self.do_bin), ("KL 散度", self.do_kl),
                          ("互信息", self.do_mi), ("交叉熵", self.do_ce)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def _box(self, frame, title):
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        t = ctk.CTkTextbox(frame, height=70, font=ui.mono(ui.FONT["body"]))
        t.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        return t

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _p(self):
        return self.p_.get("1.0", "end").strip()

    def _q(self):
        return self.q_.get("1.0", "end").strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_ent(self):
        self._run(shannon_entropy, self._p())

    def do_bin(self):
        self._run(binary_entropy, 0.5)

    def do_kl(self):
        self._run(kl_divergence, self._p(), self._q())

    def do_mi(self):
        self._run(mutual_information, "0.1,0.4;0.4,0.1")

    def do_ce(self):
        self._run(cross_entropy, self._p(), self._q())


PAGES = [InfoTheoryPage]
