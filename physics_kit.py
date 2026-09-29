# -*- coding: utf-8 -*-
"""物理公式计算：抛体运动、匀变速运动、串并联电阻、功与能。AI 工具 + 页面（教学友好）。"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "physics_kit",
    "name": "物理公式",
    "version": "1.0",
    "author": "zoilzo",
    "description": "物理公式计算：抛体运动（射程/最大高度/飞行时间）、匀变速运动、串并联电阻、功与能（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _n(v, d=0.0):
    try:
        return float(str(v).replace("，", "").strip())
    except Exception:
        return d


def _deg(v):
    return float(np.radians(_n(v, 0.0)))


# ------------------------------------------------------------
# 工具 1：抛体运动（斜抛）
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "v0": {"type": "number", "description": "初速度大小 v0（m/s）"},
        "angle": {"type": "number", "description": "抛射仰角 θ（度）"},
        "g": {"type": "number", "description": "重力加速度 g，默认 9.8（m/s²）"},
        "h0": {"type": "number", "description": "抛出点离地高度 h0（m），默认 0"},
    },
}, category="物理公式")
def projectile_motion(v0="20", angle="45", g="9.8", h0="0"):
    """斜抛运动：求初速分量、射程、最大高度、飞行时间。仰角 90° 为竖直上抛。"""
    v = _n(v0, 20)
    th = _deg(angle)
    g = _n(g, 9.8)
    h0 = _n(h0, 0)
    if v <= 0 or g <= 0:
        raise ValueError("初速度与重力加速度必须为正。")
    vx = v * np.cos(th)
    vy = v * np.sin(th)
    t_top = vy / g
    h_max = h0 + (vy ** 2) / (2 * g)
    # 落地时间：h0 + vy*t - 0.5 g t^2 = 0
    if h0 == 0:
        t_flight = 2 * vy / g if vy > 0 else 0.0
    else:
        disc = vy ** 2 + 2 * g * h0
        t_flight = (vy + np.sqrt(disc)) / g if disc >= 0 else float("nan")
    rng = vx * t_flight
    txt = (f"初速度分解：vx = {vx:.4g} m/s，vy = {vy:.4g} m/s\n"
           f"到达最高点时间 = {t_top:.4g} s\n"
           f"最大高度 = {h_max:.4g} m\n")
    if t_flight and not np.isnan(t_flight):
        txt += f"飞行时间 = {t_flight:.4g} s\n射程（水平距离）= {rng:.4g} m\n"
    else:
        txt += "该参数下无法落地（坑深过浅），请检查。\n"
    txt += f"\n提示：45° 仰角射程最大；g 默认取 {g:g} m/s²。"
    return {"text": txt}


# ------------------------------------------------------------
# 工具 2：匀变速直线运动
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "v0": {"type": "number", "description": "初速度 v0（m/s）"},
        "a": {"type": "number", "description": "加速度 a（m/s²）"},
        "t": {"type": "number", "description": "时间 t（s）"},
    },
}, category="物理公式")
def kinematics_calc(v0="0", a="9.8", t="2"):
    """匀变速直线运动：v=v0+at、s=v0·t+½·a·t²，并给出末速度与位移。"""
    v0 = _n(v0, 0)
    a = _n(a, 9.8)
    t = _n(t, 2)
    v = v0 + a * t
    s = v0 * t + 0.5 * a * t ** 2
    v_chk = np.sqrt(max(v0 ** 2 + 2 * a * s, 0))  # v² = v0² + 2as
    return {"text": f"v0 = {v0:g} m/s，a = {a:g} m/s²，t = {t:g} s\n\n"
            f"末速度 v = v0 + a·t = {v:.4g} m/s\n"
            f"位移 s = v0·t + ½·a·t² = {s:.4g} m\n"
            f"验证 v² = v0² + 2as → v = {v_chk:.4g} m/s\n\n公式：v=v0+at；s=v0t+½at²；v²=v0²+2as"}


# ------------------------------------------------------------
# 工具 3：串并联电阻
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "mode": {"type": "string", "description": "串联 series 或 并联 parallel"},
        "r": {"type": "string", "description": "电阻（欧姆），逗号分隔，如 '10,20,30'"},
        "voltage": {"type": "number", "description": "电源电压 U（V），默认 12"},
    },
}, category="物理公式")
def resistor_circuit(mode="series", r="10,20,30", voltage="12"):
    """串/并联电阻电路：求总电阻、总电流、各电阻上的电压（串联）或电流（并联）与总功率。"""
    rs = [x for x in (str(r).replace("，", ",").split(",")) if x.strip()]
    rv = [_n(x, 1) for x in rs]
    if not rv:
        raise ValueError("请至少提供一个电阻值。")
    if any(x <= 0 for x in rv):
        raise ValueError("电阻必须为正。")
    u = _n(voltage, 12)
    m = str(mode or "series").strip().lower()
    if m in ("series", "串联", "s"):
        total = sum(rv)
        cur = u / total
        lines = [f"串联等效电阻 R = ΣRi = {(' + '.join(f'{x:g}' for x in rv))} = {total:.4g} Ω",
                 f"总电流 I = U / R = {u:g} / {total:.4g} = {cur:.4g} A", "", "各电阻电压（分压）："]
        for x in rv:
            lines.append(f"  (U=IR) R={x:g} Ω → U = {cur * x:.4g} V")
        power = u * cur
        lines.append(f"\n总功率 P = U·I = {power:.4g} W")
        return {"text": "\n".join(lines)}
    if m in ("parallel", "并联", "p"):
        # 1/R = Σ 1/Ri
        inv = sum(1.0 / x for x in rv)
        total = 1.0 / inv
        cur = u / total
        lines = [f"并联等效电阻 1/R = Σ 1/Ri = {(' + '.join(f'1/{x:g}' for x in rv))} = {inv:.4g} Ω⁻¹",
                 f"等效电阻 R = {total:.4g} Ω",
                 f"总电流 I = U / R = {u:g} / {total:.4g} = {cur:.4g} A", "", "各电阻支路电流（分流）："]
        for x in rv:
            lines.append(f"  (I=U/R) R={x:g} Ω → I = {u / x:.4g} A")
        power = u * cur
        lines.append(f"\n总功率 P = U·I = {power:.4g} W")
        return {"text": "\n".join(lines)}
    raise ValueError("mode 取值：series（串联）或 parallel（并联）。")


# ------------------------------------------------------------
# 工具 4：功与能
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "mass": {"type": "number", "description": "质量 m（kg）"},
        "v": {"type": "number", "description": "速度 v（m/s）"},
        "force": {"type": "number", "description": "恒力 F（N）"},
        "distance": {"type": "number", "description": "沿力方向的位移 d（m）"},
        "angle": {"type": "number", "description": "力与位移夹角（度），默认 0"},
    },
}, category="物理公式")
def work_energy_calc(mass="2", v="10", force="5", distance="20", angle="0"):
    """功与动能：W=F·d·cosθ，动能 Ek=½mv²，并给出净功（若有）与动能定理验证。"""
    m = _n(mass, 2)
    v = _n(v, 10)
    f = _n(force, 5)
    d = _n(distance, 20)
    ang = np.radians(_n(angle, 0))
    work = f * d * np.cos(ang)
    ek = 0.5 * m * v ** 2
    txt = f"功 W = F·d·cosθ = {f:g} × {d:g} × cos({_n(angle, 0):g}°) = {work:.4g} J\n"
    txt += f"动能 Ek = ½·m·v² = ½ × {m:g} × {v:g}² = {ek:.4g} J\n"
    w_net = work  # 仅该力做功时净功取 work
    txt += f"\n动能定理（ΔEk = 净功）：当该力为唯一合力时，ΔEk = {w_net:.4g} J\n"
    txt += f"若 v=0 从静止启动，则 v = √(2·W/m) = {np.sqrt(max(2 * w_net / m, 0)) if m > 0 else 0:.4g} m/s"
    return {"text": txt}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "物理公式"
    _TOOLS = {}

    def __init__(self, master):
        super().__init__(master, layout=False)
        self._fields = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text=self._TITLE, font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="功能：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w", padx=(0, ui.SPACE["sm"]))
        self._names = [v[0] for v in self._TOOLS.values()]
        self._key_of = {v[0]: k for k, v in self._TOOLS.items()}
        self.mode_var = ctk.StringVar(value=self._names[0])
        ctk.CTkOptionMenu(row, values=self._names, variable=self.mode_var,
                          command=lambda _: self._rebuild()).grid(row=0, column=1, sticky="w")
        self.desc = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=ui.FONT["body"]), text_color="gray60",
                                 wraplength=560, justify="left")
        self.desc.pack(anchor="w", pady=(ui.SPACE["xs"], 0))
        self.field_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.field_frame.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(body, text="计算", height=h, fg_color=ui.body(), command=self._run).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self._rebuild()

    def _rebuild(self):
        for w in self.field_frame.winfo_children():
            w.destroy()
        self._fields = {}
        key = self._key_of[self.mode_var.get()]
        label, desc, fields, func = self._TOOLS[key]
        self.desc.configure(text=desc)
        self._current = func
        for r, (fkey, flabel, fdefault) in enumerate(fields):
            ctk.CTkLabel(self.field_frame, text=flabel, font=ctk.CTkFont(size=ui.FONT["body"])).grid(
                row=r, column=0, sticky="w", pady=(0, ui.SPACE["xs"]))
            e = ctk.CTkEntry(self.field_frame)
            e.insert(0, str(fdefault))
            e.grid(row=r, column=1, sticky="ew", pady=(0, ui.SPACE["xs"]))
            self.field_frame.grid_columnconfigure(1, weight=1)
            self._fields[fkey] = e

    def _run(self):
        vals = {k: e.get().strip() for k, e in self._fields.items()}
        out = self.out
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.configure(state="disabled")
        try:
            res = self._current(**vals)
            text = res.get("text", str(res)) if isinstance(res, dict) else str(res)
        except Exception as ex:
            text = f"计算出错：{ex}"
        out.configure(state="normal")
        out.insert("1.0", text)
        out.configure(state="disabled")


class PhysicsPage(_FormPage):
    NAME = "物理公式"
    EMOJI = "\U0001F3F3"
    _TITLE = "🔭 物理公式计算"
    _TOOLS = {
        "projectile_motion": (
            "抛体运动",
            "斜抛运动：分解初速度，求射程、最大高度、飞行时间。45° 射程最大。",
            [("v0", "初速度 v0 (m/s)", "20"),
             ("angle", "抛射仰角 θ（度）", "45"),
             ("g", "重力加速度 g (m/s²)", "9.8"),
             ("h0", "抛出点离地高度 (m)", "0")],
            projectile_motion),
        "kinematics_calc": (
            "匀变速运动",
            "匀变速直线运动：v=v0+at、s=v0·t+½·a·t²、v²=v0²+2as。",
            [("v0", "初速度 v0 (m/s)", "0"),
             ("a", "加速度 a (m/s²)", "9.8"),
             ("t", "时间 t (s)", "2")],
            kinematics_calc),
        "resistor_circuit": (
            "串并联电阻",
            "串/并联电阻电路：求等效电阻、总电流、分压/分流与总功率。",
            [("mode", "串联 series / 并联 parallel", "series"),
             ("r", "各电阻（Ω，逗号分隔）", "10,20,30"),
             ("voltage", "电源电压 U (V)", "12")],
            resistor_circuit),
        "work_energy_calc": (
            "功与能",
            "功 W=F·d·cosθ、动能 Ek=½·m·v²，并用动能定理验证。",
            [("mass", "质量 m (kg)", "2"),
             ("v", "速度 v (m/s)", "10"),
             ("force", "恒力 F (N)", "5"),
             ("distance", "沿力方向位移 d (m)", "20"),
             ("angle", "力与位移夹角（度）", "0")],
            work_energy_calc),
    }


PAGES = [PhysicsPage]