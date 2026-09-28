# -*- coding: utf-8 -*-
"""向量几何：点积/叉积/模长/夹角/投影。AI 工具 + 页面。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "vector_geo",
    "name": "向量几何",
    "version": "1.0",
    "author": "zoilzo",
    "description": "向量点积/叉积/模长/夹角/投影（工具 + 页面）",
}


def _v(s):
    return [float(x) for x in str(s).replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",") if x.strip()]


def _show(v):
    return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="向量")
def vec_dot(a, b):
    """向量点积 a·b（分量以逗号分隔，如 1,2,3）。"""
    A, B = _v(a), _v(b)
    if len(A) != len(B):
        return {"text": f"两向量维数不同（{len(A)} vs {len(B)}）"}
    d = sum(x * y for x, y in zip(A, B))
    return {"text": f"{_show(A)}·{_show(B)} = {d:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="向量")
def vec_cross(a, b):
    """三维向量叉积 a×b。"""
    A, B = _v(a), _v(b)
    if len(A) != 3 or len(B) != 3:
        return {"text": "叉积要求两向量都是 3 维（如 1,2,3）"}
    c = [A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]]
    return {"text": f"{_show(A)}×{_show(B)} = {_show(c)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}}, "required": ["a"]}, category="向量")
def vec_norm(a):
    """向量的模长（长度）。"""
    A = _v(a)
    n = math.sqrt(sum(x * x for x in A))
    return {"text": f"|{_show(A)}| = {n:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="向量")
def vec_angle(a, b):
    """两向量夹角（度）。"""
    A, B = _v(a), _v(b)
    na = math.sqrt(sum(x * x for x in A))
    nb = math.sqrt(sum(x * x for x in B))
    if na == 0 or nb == 0:
        return {"text": "零向量无夹角"}
    d = sum(x * y for x, y in zip(A, B))
    cosv = max(-1.0, min(1.0, d / (na * nb)))
    ang = math.degrees(math.acos(cosv))
    return {"text": f"{_show(A)} 与 {_show(B)} 夹角 = {ang:.6g}°（余弦 = {cosv:.6g}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="向量")
def vec_proj(a, b):
    """a 在 b 上的投影向量：proj_b(a) = (a·b / |b|²) b。"""
    A, B = _v(a), _v(b)
    nb2 = sum(x * x for x in B)
    if nb2 == 0:
        return {"text": "b 是零向量，无法投影"}
    d = sum(x * y for x, y in zip(A, B))
    k = d / nb2
    p = [k * x for x in B]
    return {"text": f"{_show(A)} 在 {_show(B)} 上的投影 = {_show(p)}（系数 {k:.6g}）"}


class VectorPage(ui.BasePage):
    NAME = "向量几何"
    EMOJI = "\U0001F9D0"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="向量几何", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("向量 a（分量逗号分隔）", "a", "1,2,3"), ("向量 b（分量逗号分隔）", "b", "2,-1,4")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("点积", self.do_dot), ("叉积", self.do_cross), ("模长 a", self.do_norm), ("夹角", self.do_angle), ("投影", self.do_proj)):
            ctk.CTkButton(body, text=text, height=h, fg_color=(None if text == "点积" else "gray40"), command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入两个向量，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def do_dot(self):
        self.msg(vec_dot(self.a.get(), self.b.get())["text"])

    def do_cross(self):
        self.msg(vec_cross(self.a.get(), self.b.get())["text"])

    def do_norm(self):
        self.msg(vec_norm(self.a.get())["text"])

    def do_angle(self):
        self.msg(vec_angle(self.a.get(), self.b.get())["text"])

    def do_proj(self):
        self.msg(vec_proj(self.a.get(), self.b.get())["text"])


PAGES = [VectorPage]
