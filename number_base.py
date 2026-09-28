# -*- coding: utf-8 -*-
"""进制转换与位运算：十/二/八/十六进制互转 + 按位与或异或。AI 工具 + 页面。"""
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "number_base",
    "name": "进制与位运算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "十进制/二进制/八进制/十六进制互转，按位与或异或（工具 + 页面）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}, "base": {"type": "number"}}, "required": ["n", "base"]}, category="进制")
def dec_to_base(n, base):
    """把十进制整数 n 转成 base 进制字符串（base 为 2/8/16）。"""
    n, base = int(n), int(base)
    if base not in (2, 8, 16):
        return {"text": "base 只支持 2/8/16"}
    return {"text": f"十进制 {n} = {format(n, {2: 'b', 8: 'o', 16: 'x'}[base])}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"s": {"type": "string"}, "base": {"type": "number"}}, "required": ["s", "base"]}, category="进制")
def base_to_dec(s, base):
    """把 base 进制字符串转成十进制整数（base 为 2/8/16）。"""
    s = str(s).strip().lower()
    if s[:2] in ("0b", "0o", "0x"):
        s = s[2:]
    base = int(base)
    if base not in (2, 8, 16):
        return {"text": "base 只支持 2/8/16"}
    return {"text": f"{s}(base {base}) = {int(s, base)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "number"}, "b": {"type": "number"}, "op": {"type": "string"}}, "required": ["a", "b", "op"]}, category="进制")
def bit_op(a, b, op):
    """两个整数按位运算：and/or/xor，给出结果与二进制。"""
    a, b = int(a), int(b)
    op = str(op).lower()
    if op in ("and", "&"):
        r, name = a & b, "按位与(AND)"
    elif op in ("or", "|"):
        r, name = a | b, "按位或(OR)"
    elif op in ("xor", "^"):
        r, name = a ^ b, "按位异或(XOR)"
    else:
        return {"text": "op 只支持 and/or/xor"}
    return {"text": f"{name}: {a} {op} {b} = {r}; 二进制: {bin(a)} {op} {bin(b)} = {bin(r)}"}


class NumberBasePage(ui.BasePage):
    NAME = "进制与位运算"
    EMOJI = "\U0001F522"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="进制与位运算", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("数值 n 或 base 进制串", "s", "255"), ("基数 base (2/8/16)", "base", "16"), ("整数 a", "a", "12"), ("整数 b", "b", "10")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("转进制", self.do_to_base), ("转十进制", self.do_to_dec), ("按位与", self.do_and), ("按位或", self.do_or), ("按位异或", self.do_xor)):
            ctk.CTkButton(body, text=text, height=h, fg_color=(None if text == "转进制" else "gray40"), command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入数值，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def do_to_base(self):
        self.msg(dec_to_base(self.s.get(), int(self.base.get()))["text"])

    def do_to_dec(self):
        self.msg(base_to_dec(self.s.get(), int(self.base.get()))["text"])

    def do_and(self):
        self.msg(bit_op(self.a.get(), self.b.get(), "and")["text"])

    def do_or(self):
        self.msg(bit_op(self.a.get(), self.b.get(), "or")["text"])

    def do_xor(self):
        self.msg(bit_op(self.a.get(), self.b.get(), "xor")["text"])


PAGES = [NumberBasePage]
