# -*- coding: utf-8 -*-
"""三角函数：角度制正弦/余弦/正切、直角三角形求解。AI 工具 + 页面。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "trig_kit",
    "name": "三角函数",
    "version": "1.0",
    "author": "zoilzo",
    "description": "角度制 sin/cos/tan，直角三角形求边与角（工具 + 页面）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"deg": {"type": "number"}, "func": {"type": "string"}}, "required": ["deg", "func"]}, category="三角函数")
def trig_value(deg, func):
    """求角度 deg（度）的正弦/余弦/正切。func: sin/cos/tan。"""
    d = float(deg)
    r = math.radians(d)
    f = func.lower()
    if f in ("sin", "正弦"):
        return {"text": f"sin({d}°) = {math.sin(r):.6g}"}
    if f in ("cos", "余弦"):
        return {"text": f"cos({d}°) = {math.cos(r):.6g}"}
    if f in ("tan", "正切"):
        if abs(math.cos(r)) < 1e-12:
            return {"text": f"tan({d}°) 无定义（cos(90°)=0）"}
        return {"text": f"tan({d}°) = {math.tan(r):.6g}"}
    return {"text": "func 只支持 sin/cos/tan"}


@ai_tools._reg
@ai_tools._tool({"properties": {"opp": {"type": "number"}, "adj": {"type": "number"}}, "required": ["opp", "adj"]}, category="三角函数")
def solve_right(opp, adj):
    """直角三角形：已知两条直角边 opp/adj，求斜边与两个锐角（度）。"""
    opp, adj = float(opp), float(adj)
    hyp = math.hypot(opp, adj)
    if hyp == 0:
        return {"text": "两条边不能都为 0"}
    ang_a = math.degrees(math.atan2(opp, adj))
    ang_b = 90 - ang_a
    return {"text": f"直角边 opp={opp}, adj={adj}：斜边={hyp:.6g}； 对边对角={ang_a:.6g}°，邻边对角={ang_b:.6g}°"}


@ai_tools._reg
@ai_tools._tool({"properties": {"deg": {"type": "number"}}, "required": ["deg"]}, category="三角函数")
def deg_to_rad(deg):
    """角度转弧度。"""
    return {"text": f"{deg}° = {math.radians(float(deg)):.6g} 弧度"}


@ai_tools._reg
@ai_tools._tool({"properties": {"rad": {"type": "number"}}, "required": ["rad"]}, category="三角函数")
def rad_to_deg(rad):
    """弧度转角度。"""
    return {"text": f"{rad} 弧度 = {math.degrees(float(rad)):.6g}°"}


class TrigPage(ui.BasePage):
    NAME = "三角函数"
    EMOJI = "\U0001F4CF"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📏 三角函数", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        for lab, attr, default in (("角度（度）", "deg", "30"), ("函数 sin/cos/tan", "func", "sin"), ("直角边 对边", "opp", "3"), ("直角边 邻边", "adj", "4")):
            ctk.CTkLabel(body, text=lab, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
            e = ctk.CTkEntry(body)
            e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
            e.insert(0, default)
            setattr(self, attr, e)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("求三角值", self.do_trig), ("解直角三角形", self.do_right)):
            ctk.CTkButton(body, text=text, height=h, fg_color=(None if text == "求三角值" else "gray40"), command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入数值，点按钮计算")

    def msg(self, s):
        self.out.configure(state="normal"); self.out.delete("1.0", "end"); self.out.insert("1.0", str(s)); self.out.configure(state="disabled")

    def do_trig(self):
        self.msg(trig_value(self.deg.get(), self.func.get())["text"])

    def do_right(self):
        self.msg(solve_right(float(self.opp.get()), float(self.adj.get()))["text"])


PAGES = [TrigPage]