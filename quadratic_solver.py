# -*- coding: utf-8 -*-
"""一元二次方程全面解：判别式/根/顶点/对称轴。AI 工具 + 页面。"""
import cmath
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "quadratic_solver",
    "name": "一元二次方程",
    "version": "1.0",
    "author": "zoilzo",
    "description": "Ax²+Bx+C=0 的判别式/根(实或复)/顶点/对称轴（工具 + 页面）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "number"}}, "required": ["a", "b", "c"]}, category="方程")
def solve_quadratic(a, b, c):
    """解一元二次方程 ax²+bx+c=0，含判别式、实/复根、顶点与对称轴。"""
    a, b, c = float(a), float(b), float(c)
    if a == 0:
        if b == 0:
            return {"text": "不是方程：a=0 且 b=0，无解或恒成立"}
        return {"text": f"一次方程 {b}x+{c}=0：x = {-c / b:.6g}"}
    disc = b * b - 4 * a * c
    D = math.sqrt(abs(disc))
    if disc >= 0:
        x1 = (-b + D) / (2 * a)
        x2 = (-b - D) / (2 * a)
        roots = f"x1={x1:.6g}, x2={x2:.6g}"
        rtype = "两相异实根" if disc > 0 else "两相等实根"
    else:
        re_part = -b / (2 * a)
        im_part = D / (2 * a)
        roots = f"x1={re_part:.6g}+{im_part:.6g}i, x2={re_part:.6g}-{im_part:.6g}i"
        rtype = "两共轭复根"
    vx = -b / (2 * a)
    vy = (4 * a * c - b * b) / (4 * a)
    return {"text": f"判别式 Δ={disc:.6g}（{rtype}）； 根：{roots}； 顶点=({vx:.6g},{vy:.6g})； 对称轴 x={vx:.6g}"}


class QuadraticPage(ui.BasePage):
    NAME = "一元二次方程"
    EMOJI = "\U0001F4C8"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📈 一元二次方程", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("a =", "a", "1"), ("b =", "b", "-3"), ("c =", "c", "2")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w")
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(body, text="求解", height=h, command=self.do_solve).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入 a/b/c，点「求解」。例：1,-3,2")

    def msg(self, s):
        self.out.configure(state="normal"); self.out.delete("1.0", "end"); self.out.insert("1.0", str(s)); self.out.configure(state="disabled")

    def do_solve(self):
        try:
            self.msg(solve_quadratic(float(self.a.get()), float(self.b.get()), float(self.c.get()))["text"])
        except Exception as e:
            self.msg(f"出错：{e}")


PAGES = [QuadraticPage]