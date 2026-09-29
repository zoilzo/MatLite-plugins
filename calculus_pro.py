# -*- coding: utf-8 -*-
"""高等微积分：偏导数 / 梯度 / 全微分 / 泰勒展开 / 二重积分 / 隐函数求导（符号计算，工具 + 页面）。"""
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools
import sympy as sp

PLUGIN = {
    "id": "calculus_pro",
    "name": "高等微积分",
    "version": "1.0",
    "author": "zoilzo",
    "description": "偏导数/梯度/全微分/泰勒展开/二重积分/隐函数（工具 + 页面，高等数学）",
}

_x, _y, _z = sp.symbols("x y z")


def _parse(expr, syms="xy"):
    local = {c: sp.Symbol(c) for c in syms}
    return sp.sympify(expr, locals=local)


@ai_tools._reg
@ai_tools._tool({"properties": {"expr": {"type": "string", "default": "x**2*sin(y)+y**2"}, "var": {"type": "string", "default": "x"}, "n": {"type": "integer", "default": 1}}, "required": ["expr"]}, category="微积分")
def partial_derivative(expr="x**2*sin(y)+y**2", var="x", n=1):
    e = _parse(expr)
    v = sp.Symbol(var)
    r = sp.diff(e, v, n)
    return {"text": f"对 {var} 的 {n} 阶偏导数： {sp.simplify(r)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"expr": {"type": "string", "default": "x**2+3*x*y+y**2"}}, "required": ["expr"]}, category="微积分")
def gradient_vec(expr="x**2+3*x*y+y**2"):
    e = _parse(expr)
    gx = sp.simplify(sp.diff(e, _x))
    gy = sp.simplify(sp.diff(e, _y))
    return {"text": f"梯度 ∇f = ({gx}, {gy})   （(x,y) 处上升最快的方向）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"expr": {"type": "string", "default": "x**2+y**2"}, "x0": {"type": "integer", "default": 0}, "y0": {"type": "integer", "default": 0}, "n": {"type": "integer", "default": 3}}, "required": ["expr"]}, category="微积分")
def taylor_2d(expr="x**2+y**2", x0=0, y0=0, n=3):
    e = _parse(expr)
    r = sp.series(e, _x, x0, n + 1).removeO()
    r = sp.series(r, _y, y0, n + 1).removeO()
    return {"text": f"在 ({x0},{y0}) 处的 {n} 阶泰勒展开： {sp.simplify(sp.expand(r))}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"expr": {"type": "string", "default": "x*y"}, "x0": {"type": "number", "default": 0}, "x1": {"type": "number", "default": 1}, "y0": {"type": "number", "default": 0}, "y1": {"type": "number", "default": 1}}, "required": ["expr"]}, category="微积分")
def double_integral(expr="x*y", x0=0, x1=1, y0=0, y1=1):
    e = _parse(expr)
    r = sp.integrate(e, (_x, x0, x1), (_y, y0, y1))
    return {"text": f"∬ f dA = {sp.simplify(r)}   （x∈[{x0},{x1}]，y∈[{y0},{y1}]）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"expr": {"type": "string", "default": "x**2+y"}, "y": {"type": "string", "default": "sqrt(x)"}, "x0": {"type": "number", "default": 0}, "x1": {"type": "number", "default": 1}}, "required": ["expr"]}, category="微积分")
def region_integral(expr="x**2+y", y="sqrt(x)", x0=0, x1=1):
    e = _parse(expr)
    bound = _parse(y, "x")
    r = sp.integrate(e, (_y, 0, bound), (_x, x0, x1))
    return {"text": f"∬ f dy dx（y∈[0,{y}], x∈[{x0},{x1}]） = {sp.simplify(r)}"}


class CalcProPage(ui.BasePage):
    NAME = "高等微积分"
    EMOJI = "\U0001F4DA"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="\U0001F4DA 高等微积分（符号计算）", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")

        self.expr = ctk.CTkEntry(body, placeholder_text="函数 f(x,y)，例：x**2*sin(y)+y**2")
        self.expr.pack(fill="x", pady=(ui.SPACE["md"], ui.SPACE["sm"]))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=ui.SPACE["sm"])
        row.grid_columnconfigure(0, weight=1)
        self.var = ctk.CTkComboBox(row, values=["x", "y", "z"], width=70)
        self.var.set("x")
        self.var.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(row, text="对变量求偏导").grid(row=0, column=1, padx=8)
        self.n = ctk.CTkEntry(row, width=60)
        self.n.insert(0, "1")
        self.n.grid(row=0, column=2, sticky="w")

        btn = ctk.CTkFrame(body, fg_color="transparent")
        btn.pack(fill="x", pady=ui.SPACE["sm"])
        actions = [
            ("偏导", lambda: self._run(partial_derivative, [self.expr.get() or "x**2*sin(y)+y**2", self.var.get(), int(self.n.get() or 1)])),
            ("梯度", lambda: self._run(gradient_vec, [self.expr.get() or "x**2+3*x*y+y**2"])),
            ("泰勒展开", lambda: self._run(taylor_2d, [self.expr.get() or "x**2+y**2", 0, 0, 3])),
            ("二重积分", lambda: self._run(double_integral, [self.expr.get() or "x*y", 0, 1, 0, 1])),
        ]
        for i, (txt, fn) in enumerate(actions):
            ctk.CTkButton(btn, text=txt, command=fn, width=90).grid(row=0, column=i, padx=4, pady=4)

        self.out = ui.mono_textbox(body)
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


PAGES = [CalcProPage]
