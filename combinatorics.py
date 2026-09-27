# -*- coding: utf-8 -*-
"""组合数学：排列 / 组合 / 阶乘。插件页 + AI 工具。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "combinatorics",
    "name": "组合数学",
    "version": "1.0",
    "author": "zoilzo",
    "description": "排列/组合/阶乘：支持页面与 AI 助手调用",
}

@ai_tools._reg  # 注意：_reg 在上、_tool 在下
@ai_tools._tool({"properties": {"n": {"type": "number"}, "r": {"type": "number"}}, "required": ["n", "r"]}, category="组合数学")
def npr(n, r):
    """计算排列数 P(n, r) = n!/(n-r)!"""
    return {"text": f"P({int(n)}, {int(r)}) = {math.perm(int(n), int(r))}"}

@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}, "r": {"type": "number"}}, "required": ["n", "r"]}, category="组合数学")
def ncr(n, r):
    """计算组合数 C(n, r) = n!/(r!(n-r)!)"""
    return {"text": f"C({int(n)}, {int(r)}) = {math.comb(int(n), int(r))}"}

@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}}, "required": ["n"]}, category="组合数学")
def factorial_n(n):
    """计算阶乘 n!"""
    return {"text": f"{int(n)}! = {math.factorial(int(n))}"}

class CombinatoricsPage(ui.BasePage):
    NAME = "组合数学"
    EMOJI = "\U0001F3B2"
    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🎲 组合数学", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(
            anchor="w", pady=(ui.SPACE["xs"], ui.SPACE["sm"]))
        for lab, attr, default in (("n =", "n", "10"), ("r =", "r", "3")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(row, text="排列 P(n,r)", height=h, command=self.do_perm).grid(
            row=0, column=0, padx=(0, ui.SPACE["xs"]), sticky="ew")
        ctk.CTkButton(row, text="组合 C(n,r)", height=h, fg_color="gray40", command=self.do_comb).grid(
            row=0, column=1, padx=(ui.SPACE["xs"], 0), sticky="ew")
        ctk.CTkButton(body, text="阶乘 n!", height=h, fg_color="gray40", command=self.do_fact).pack(
            fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("点按钮计算")
    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")
    def do_perm(self):
        try:
            n, r = int(self.n.get()), int(self.r.get())
            self.msg(f"P({n}, {r}) = {math.perm(n, r)}")
        except Exception as e:
            self.msg(f"出错：{e}")
    def do_comb(self):
        try:
            n, r = int(self.n.get()), int(self.r.get())
            self.msg(f"C({n}, {r}) = {math.comb(n, r)}")
        except Exception as e:
            self.msg(f"出错：{e}")
    def do_fact(self):
        try:
            n = int(self.n.get())
            self.msg(f"{n}! = {math.factorial(n)}")
        except Exception as e:
            self.msg(f"出错：{e}")

PAGES = [CombinatoricsPage]
