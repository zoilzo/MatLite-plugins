# -*- coding: utf-8 -*-
"""集合运算：并/交/差/对称差/笛卡尔积 + 基数。AI 工具 + 页面。"""
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "sets_ops",
    "name": "集合运算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "集合并/交/差/对称差/笛卡尔积/基数（工具 + 页面）",
}


def _parse(s):
    """把逗号/空格分隔的字符串解析成元素列表（整数优先）。"""
    raw = [x.strip() for x in str(s).replace("，", ",").split(",") if x.strip()]
    out = []
    for x in raw:
        try:
            out.append(int(x))
        except ValueError:
            out.append(x)
    return sorted(set(out))


def _show(a):
    return "{" + ", ".join(str(x) for x in sorted(a, key=lambda z: (isinstance(z, str), z))) + "}"


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="集合")
def set_union(a, b):
    """集合并集 A∪B。"""
    A, B = set(_parse(a)), set(_parse(b))
    return {"text": f"A∪B = {_show(A | B)}（基数 {len(A | B)}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="集合")
def set_intersect(a, b):
    """集合交集 A∩B。"""
    A, B = set(_parse(a)), set(_parse(b))
    return {"text": f"A∩B = {_show(A & B)}（基数 {len(A & B)}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="集合")
def set_diff(a, b):
    """集合差集 A-B（属于 A 不属于 B）。"""
    A, B = set(_parse(a)), set(_parse(b))
    return {"text": f"A-B = {_show(A - B)}（基数 {len(A - B)}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="集合")
def set_sym_diff(a, b):
    """集合对称差 A△B = (A-B)∪(B-A)。"""
    A, B = set(_parse(a)), set(_parse(b))
    return {"text": f"A△B = {_show(A ^ B)}（基数 {len(A ^ B)}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a", "b"]}, category="集合")
def set_cartesian(a, b):
    """集合笛卡尔积 A×B（所有有序对）。"""
    A, B = _parse(a), _parse(b)
    pairs = [(x, y) for x in A for y in B]
    return {"text": f"A×B = {{" + ", ".join(f"({x},{y})" for x, y in pairs) + "}}（基数 {len(pairs)}）"}


class SetsPage(ui.BasePage):
    NAME = "集合运算"
    EMOJI = "\U0001F4CA"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📊 集合运算", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("集合 A（逗号分隔）", "a", "1,2,3,4"), ("集合 B（逗号分隔）", "b", "3,4,5,6")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("并集", self.do_union), ("交集", self.do_inter), ("差集 A-B", self.do_diff), ("对称差", self.do_sym), ("笛卡尔积", self.do_cart)):
            ctk.CTkButton(body, text=text, height=h, fg_color=(None if text == "并集" else "gray40"), command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入两个集合，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal"); self.out.delete("1.0", "end"); self.out.insert("1.0", str(s)); self.out.configure(state="disabled")

    def do_union(self):
        self.msg(set_union(self.a.get(), self.b.get())["text"])

    def do_inter(self):
        self.msg(set_intersect(self.a.get(), self.b.get())["text"])

    def do_diff(self):
        self.msg(set_diff(self.a.get(), self.b.get())["text"])

    def do_sym(self):
        self.msg(set_sym_diff(self.a.get(), self.b.get())["text"])

    def do_cart(self):
        self.msg(set_cartesian(self.a.get(), self.b.get())["text"])


PAGES = [SetsPage]