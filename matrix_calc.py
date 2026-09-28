# -*- coding: utf-8 -*-
"""矩阵运算：加减乘、转置、行列式、逆、解方程组、特征值。AI 工具 + 页面。"""
import customtkinter as ctk
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "matrix_calc",
    "name": "矩阵运算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "矩阵加减乘/转置/行列式/逆/解方程组/特征值（工具 + 页面）",
}


def _parse_mat(s):
    """把矩阵字符串解析成 2 维 numpy 数组。
    格式：行用分号分隔，行内数字用空格或逗号分隔；也接受 Python 嵌套列表/JSON 字符串。
    例："1 2; 3 4" 或 "[[1,2],[3,4]]"。"""
    s = str(s).strip()
    if not s:
        raise ValueError("矩阵不能为空")
    if s.startswith("["):
        try:
            import json
            obj = json.loads(s)
        except Exception:
            import ast
            obj = ast.literal_eval(s)
        arr = np.asarray(obj, dtype=float)
    else:
        rows = []
        for r0 in s.split(";"):
            r = r0.strip().replace(",", " ").replace("[", "").replace("]", "").replace("（", "").replace("）", "")
            parts = [p for p in r.split() if p]
            if not parts:
                continue
            rows.append([float(p) for p in parts])
        if not rows:
            raise ValueError("矩阵格式无法解析")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("矩阵各行长度不一致（须为矩形矩阵）")
        arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2:
        raise ValueError("矩阵须为 2 维（多行多列）")
    return arr


def _fmt_arr(a):
    """把 numpy 数组格式化成易读文本。"""
    a = np.asarray(a)
    if a.ndim == 0:
        return f"{a.item():.6g}"
    if a.ndim == 1:
        return "[" + ", ".join(f"{v:.6g}" for v in a) + "]"
    return "\n".join("[" + ", ".join(f"{v:.6g}" for v in row) + "]" for row in a)


def _fmt_eig(vals):
    out = []
    for v in vals:
        if abs(v.imag) < 1e-9:
            out.append(f"{v.real:.6g}")
        else:
            out.append(f"{v.real:.4g}{v.imag:+.4g}i")
    return "[" + ", ".join(out) + "]"


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}, "B": {"type": "string"}}, "required": ["A", "B"]}, category="矩阵")
def mat_add(A, B):
    """两个矩阵相加。A、B 用 "行; 行" 表示（行内数字用空格或逗号分隔），如 "1 2; 3 4"。"""
    r = np.asarray(_parse_mat(A)) + np.asarray(_parse_mat(B))
    return {"text": _fmt_arr(r)}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}, "B": {"type": "string"}}, "required": ["A", "B"]}, category="矩阵")
def mat_mul(A, B):
    """两个矩阵相乘（A·B，行乘列）。格式同矩阵加法。"""
    r = np.asarray(_parse_mat(A)) @ np.asarray(_parse_mat(B))
    return {"text": _fmt_arr(r)}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}}, "required": ["A"]}, category="矩阵")
def mat_det(A):
    """矩阵的行列式（须为方阵）。"""
    a = np.asarray(_parse_mat(A))
    return {"text": f"det = {np.linalg.det(a):.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}}, "required": ["A"]}, category="矩阵")
def mat_inv(A):
    """矩阵的逆矩阵（可逆时）。"""
    a = np.asarray(_parse_mat(A))
    if abs(np.linalg.det(a)) < 1e-12:
        return {"text": "该矩阵不可逆（行列式为 0）"}
    return {"text": _fmt_arr(np.linalg.inv(a))}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}, "b": {"type": "string"}}, "required": ["A", "b"]}, category="矩阵")
def mat_solve(A, b):
    """解线性方程组 A·x = b。b 用 "行; 行" 表示（列向量可为单行/单列）。"""
    a = np.asarray(_parse_mat(A))
    bb = np.asarray(_parse_mat(b))
    x = np.linalg.solve(a, bb)
    return {"text": _fmt_arr(x)}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}}, "required": ["A"]}, category="矩阵")
def mat_transpose(A):
    """矩阵的转置。"""
    return {"text": _fmt_arr(np.asarray(_parse_mat(A)).T)}


@ai_tools._reg
@ai_tools._tool({"properties": {"A": {"type": "string"}}, "required": ["A"]}, category="矩阵")
def mat_props(A):
    """矩阵性质：秩、行列式、迹、特征值（方阵）。"""
    a = np.asarray(_parse_mat(A))
    m, n = a.shape
    s = f"形状 {m}×{n}, 秩 = {np.linalg.matrix_rank(a)}"
    if m == n:
        s += f", det = {np.linalg.det(a):.6g}, 迹 = {np.trace(a):.6g}"
        s += f"\n特征值: {_fmt_eig(np.linalg.eigvals(a))}"
    else:
        s += "（非方阵，无行列式/迹/特征值）"
    return {"text": s}


class MatrixPage(ui.BasePage):
    NAME = "矩阵运算"
    EMOJI = "\U0001F9EE"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🧮 矩阵运算", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="矩阵格式：行用分号分隔，行内数字用空格或逗号，如 1 2; 3 4",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.ta = self._box(body, "矩阵 A")
        self.tb = self._box(body, "矩阵 B（/向量 b）")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("加法", self.do_add), ("乘法", self.do_mul), ("行列式", self.do_det),
                          ("逆矩阵", self.do_inv), ("转置", self.do_transp), ("解 A·x=b", self.do_solve), ("性质", self.do_props)):
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

    def _a(self):
        return self.ta.get("1.0", "end").strip()

    def _b(self):
        return self.tb.get("1.0", "end").strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_add(self):
        self._run(mat_add, self._a(), self._b())

    def do_mul(self):
        self._run(mat_mul, self._a(), self._b())

    def do_det(self):
        self._run(mat_det, self._a())

    def do_inv(self):
        self._run(mat_inv, self._a())

    def do_transp(self):
        self._run(mat_transpose, self._a())

    def do_solve(self):
        self._run(mat_solve, self._a(), self._b())

    def do_props(self):
        self._run(mat_props, self._a())


PAGES = [MatrixPage]
