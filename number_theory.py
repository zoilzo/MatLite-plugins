# -*- coding: utf-8 -*-
"""数论：最大公约数/最小公倍数、质因数分解、素数判断、完全数。AI 工具。"""
from math import gcd as _gcd
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "number_theory",
    "name": "数论基础",
    "version": "1.0",
    "author": "zoilzo",
    "description": "最大公约数/最小公倍数、质因数分解、素数判断、完全数（AI 工具）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}, category="数论")
def gcd_lcm(a, b):
    """求两数的最大公约数(GCD) 与最小公倍数(LCM)。"""
    a, b = int(a), int(b)
    g = _gcd(a, b)
    l = 0 if (g == 0 or a == 0 or b == 0) else abs(a * b) // g
    return {"text": f"gcd({a},{b})={g}； lcm({a},{b})={l}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}}, "required": ["n"]}, category="数论")
def is_prime(n):
    """判断 n 是否为素数（n>=2）"""
    n = int(n)
    if n < 2:
        return {"text": f"{n} 不是素数（素数必须 >= 2）"}
    if n in (2, 3):
        return {"text": f"{n} 是素数"}
    if n % 2 == 0 or n % 3 == 0:
        return {"text": f"{n} 不是素数"}
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return {"text": f"{n} 不是素数"}
        i += 6
    return {"text": f"{n} 是素数"}


@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}}, "required": ["n"]}, category="数论")
def prime_factors(n):
    """给出 n 的质因数分解（底数 + 指数）。"""
    n = int(n)
    if n <= 0:
        return {"text": "请输入正整数"}
    x = n
    parts = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            e = 0
            while x % d == 0:
                x //= d
                e += 1
            parts.append((d, e))
        d += 1 if d == 2 else 2
    if x > 1:
        parts.append((x, 1))
    if not parts:
        return {"text": f"{n} = 1（无质因数）"}
    expr = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in parts)
    return {"text": f"{n} = {expr}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"n": {"type": "number"}}, "required": ["n"]}, category="数论")
def perfect_number(n):
    """判断 n 是否为完全数（等于其真因子之和，如 6=1+2+3）。"""
    n = int(n)
    if n <= 0:
        return {"text": "请输入正整数"}
    s = 1 if n > 1 else 0
    i = 2
    while i * i <= n:
        if n % i == 0:
            s += i
            if i != n // i:
                s += n // i
        i += 1
    ok = (s == n)
    return {"text": f"{n} 的因子和={s}； {'是' if ok else '不是'}完全数"}


class NumberTheoryPage(ui.BasePage):
    NAME = "数论基础"
    EMOJI = "\U0001F522"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🔢 数论基础", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(
            anchor="w", pady=(ui.SPACE["xs"], ui.SPACE["sm"]))
        for lab, attr, default in (("a =", "a", "12"), ("b =", "b", "18"), ("n =", "n", "120")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(row, text="最大公约/最小公倍", height=h, command=self.do_gcd).grid(
            row=0, column=0, padx=(0, ui.SPACE["xs"]), sticky="ew")
        ctk.CTkButton(row, text="素数判断", height=h, fg_color="gray40", command=self.do_prime).grid(
            row=0, column=1, padx=(ui.SPACE["xs"], 0), sticky="ew")
        ctk.CTkButton(body, text="质因数分解", height=h, fg_color="gray40", command=self.do_factor).pack(
            fill="x", pady=(0, ui.SPACE["sm"]))
        ctk.CTkButton(body, text="完全数判断", height=h, fg_color="gray40", command=self.do_perfect).pack(
            fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入数值，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def do_gcd(self):
        try:
            self.msg(gcd_lcm(int(self.a.get()), int(self.b.get()))["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_prime(self):
        try:
            self.msg(is_prime(int(self.n.get()))["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_factor(self):
        try:
            self.msg(prime_factors(int(self.n.get()))["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_perfect(self):
        try:
            self.msg(perfect_number(int(self.n.get()))["text"])
        except Exception as e:
            self.msg(f"出错：{e}")


PAGES = [NumberTheoryPage]
