# -*- coding: utf-8 -*-
"""分数运算：加减乘除、约分、通分比较。AI 工具 + 页面。"""
import customtkinter as ctk
from fractions import Fraction
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "fraction_kit",
    "name": "分数运算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "分数加减乘除、约分、大小比较（工具 + 页面）",
}


def _frac(s):
    s = str(s).strip()
    if "/" in s:
        return Fraction(s)
    return Fraction(int(s), 1)


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}, "op": {"type": "string"}}, "required": ["a", "b", "op"]}, category="分数")
def frac_op(a, b, op):
    """分数四则运算。a/b 用 "分子/分母" 表示，op 为 add/sub/mul/div。"""
    A, B = _frac(a), _frac(b)
    op = op.lower()
    if op in ("add", "+"):
        R, name = A + B, "+"
    elif op in ("sub", "-"):
        R, name = A - B, "-"
    elif op in ("mul", "*", "×"):
        R, name = A * B, "×"
    elif op in ("div", "/", "÷"):
        if B == 0:
            return {"text": "除数不能为 0"}
        R, name = A / B, "÷"
    else:
        return {"text": "op 只支持 add/sub/mul/div"}
    return {"text": f"{a} {name} {b} = {R}（约分后）； 小数 = {float(R):.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}}, "required": ["a"]}, category="分数")
def frac_simplify(a):
    """把分数量化成最简形式。"""
    A = _frac(a)
    return {"text": f"{a} = {A}（最简）； 小数 = {float(A):.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="分数")
def frac_compare(a, b):
    """比较两个分数的大小。"""
    A, B = _frac(a), _frac(b)
    if A == B:
        s = f"{a} = {b}（相等）"
    elif A > B:
        s = f"{a} > {b}"
    else:
        s = f"{a} < {b}"
    return {"text": s}


class FractionPage(ui.BasePage):
    NAME = "分数运算"
    EMOJI = "\U0001F9EE"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🧮 分数运算", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("分数 a（分子/分母）", "a", "1/2"), ("分数 b（分子/分母）", "b", "1/3")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("加法", lambda: self.do("add")), ("减法", lambda: self.do("sub")), ("乘法", lambda: self.do("mul")), ("除法", lambda: self.do("div")), ("约分", self.simplify), ("比较大小", self.compare)):
            ctk.CTkButton(body, text=text, height=h, fg_color=(None if text == "加法" else "gray40"), command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入两个分数，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal"); self.out.delete("1.0", "end"); self.out.insert("1.0", str(s)); self.out.configure(state="disabled")

    def do(self, op):
        try:
            self.msg(frac_op(self.a.get(), self.b.get(), op)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def simplify(self):
        try:
            self.msg(frac_simplify(self.a.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def compare(self):
        try:
            self.msg(frac_compare(self.a.get(), self.b.get())["text"])
        except Exception as e:
            self.msg(f"出错：{e}")


PAGES = [FractionPage]