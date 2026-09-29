# -*- coding: utf-8 -*-
"""空间解析几何：三维点/线/面/球的距离、夹角、方程与交点。AI 工具 + 页面（进阶教学）。"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "geometry_3d",
    "name": "空间几何",
    "version": "1.0",
    "author": "zoilzo",
    "description": "空间解析几何：异面/相交直线距离与交点、三点定平面、点到平面距离、平面夹角（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算（numpy，不碰 GUI）
# ------------------------------------------------------------
def _vec(v, n=3):
    """把 'x,y,z' 或 [x,y,z] 解析成 n 维 numpy 向量。"""
    if isinstance(v, (list, tuple)):
        a = np.array([float(x) for x in v[:n]], dtype=float)
    else:
        s = str(v).replace("，", ",").replace(" ", ",")
        a = np.array([float(x) for x in s.split(",")[:n]], dtype=float)
    if a.size < n:
        raise ValueError(f"向量需要 {n} 个分量（用逗号或空格分隔），收到：{v}")
    return a


def _fmt(a):
    """向量 -> 可读文本。"""
    return "(" + ", ".join(f"{float(x):.4g}" for x in np.atleast_1d(a)) + ")"


def _n(v, d=0.0):
    try:
        return float(str(v).replace("，", "").strip())
    except Exception:
        return d


# ------------------------------------------------------------
# 工具 1：空间两直线的距离 / 交点
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "p1": {"type": "string", "description": "直线 L1 过的点 (x,y,z)，如 '0,0,0'"},
        "d1": {"type": "string", "description": "直线 L1 的方向向量，如 '1,1,0'"},
        "p2": {"type": "string", "description": "直线 L2 过的点，如 '1,0,0'"},
        "d2": {"type": "string", "description": "直线 L2 的方向向量，如 '1,0,0'"},
    },
}, category="空间解析几何")
def line_line_3d(p1="0,0,0", d1="1,1,0", p2="1,0,0", d2="1,0,0"):
    """求三维空间中两条直线的距离与最近点；若相交则给出交点。直线由「过点+方向向量」给出。"""
    a, v1 = _vec(p1), _vec(d1)
    b, v2 = _vec(p2), _vec(d2)
    if np.linalg.norm(v1) < 1e-12 or np.linalg.norm(v2) < 1e-12:
        raise ValueError("方向向量不能为零向量。")
    e = b - a
    nrm = np.linalg.norm(np.cross(v1, v2))
    lines = f"L1: 过点 {_fmt(a)} 方向 {_fmt(v1)}\nL2: 过点 {_fmt(b)} 方向 {_fmt(v2)}"
    if nrm < 1e-9:
        # 平行
        d = np.linalg.norm(np.cross(e, v1)) / np.linalg.norm(v1)
        return {"text": f"{lines}\n\n两直线平行，最短距离 d = {d:.5g}"}
    # 最近点：解 2x2 方程组
    A = np.array([[v1 @ v1, -(v1 @ v2)], [v1 @ v2, -(v2 @ v2)]])
    r = np.array([v1 @ e, v2 @ e])
    t, s = np.linalg.solve(A, r)
    cp1, cp2 = a + t * v1, b + s * v2
    dist = np.linalg.norm(cp1 - cp2)
    txt = f"{lines}\n\n最近点 P1' = {_fmt(cp1)}（t = {t:.5g}）\n最近点 P2' = {_fmt(cp2)}（s = {s:.5g}）\n最短距离 d = {dist:.5g}"
    if dist < 1e-9:
        txt += "\n\n➤ 两直线共面且相交，交点即为 " + _fmt(cp1)
    return {"text": txt}


# ------------------------------------------------------------
# 工具 2：三点确定平面
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "p1": {"type": "string", "description": "平面上的点 A (x,y,z)"},
        "p2": {"type": "string", "description": "平面上的点 B (x,y,z)"},
        "p3": {"type": "string", "description": "平面上的点 C (x,y,z)"},
    },
}, category="空间解析几何")
def plane_from_points(p1="1,0,0", p2="0,1,0", p3="0,0,1"):
    """由三个不共线的点确定平面方程，返回法向量、一般式与点法式。"""
    a, b, c = _vec(p1), _vec(p2), _vec(p3)
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    if np.linalg.norm(n) < 1e-12:
        raise ValueError("三点共线，无法确定唯一平面。")
    n = n / np.linalg.norm(n)
    n0, n1, n2 = n
    d = -float(n @ a)
    pts = f"A {_fmt(a)}\nB {_fmt(b)}\nC {_fmt(c)}"
    return {"text": f"{pts}\n\n法向量 n = {_fmt(n)}\n\n点法式：\n  {n0:.5g}(x-{a[0]:.5g}) + {n1:.5g}(y-{a[1]:.5g}) + {n2:.5g}(z-{a[2]:.5g}) = 0\n\n一般式 (Ax+By+Cz+D=0)：\n  {n0:.5g}x + {n1:.5g}y + {n2:.5g}z + {d:.5g} = 0"}


# ------------------------------------------------------------
# 工具 3：点到平面的距离
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "point": {"type": "string", "description": "目标点 P (x,y,z)"},
        "normal": {"type": "string", "description": "平面法向量 (A,B,C)"},
        "through": {"type": "string", "description": "平面上一点 (x0,y0,z0)"},
    },
}, category="空间解析几何")
def point_plane_distance(point="1,2,3", normal="1,1,1", through="0,0,0"):
    """求点到平面的距离与垂足（投影点）。平面由「法向量 + 过一点」给出。"""
    p, n = _vec(point), _vec(normal)
    q = _vec(through)
    ln = np.linalg.norm(n)
    if ln < 1e-12:
        raise ValueError("法向量不能为零向量。")
    n = n / ln
    signed = float(n @ (p - q))
    foot = p - signed * n
    return {"text": f"点 P {_fmt(p)}\n平面：过 {_fmt(q)}，法向量 {_fmt(n)}\n\n有向距离 = {signed:.5g}\n距离 |d| = {abs(signed):.5g}\n垂足（投影）H = {_fmt(foot)}\n\n点 P 到平面距离公式：d = |A·（P-P0)| / |n|"}


# ------------------------------------------------------------
# 工具 4：两平面（或两方向向量）的夹角
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "n1": {"type": "string", "description": "第一个平面法向量 (A,B,C)"},
        "n2": {"type": "string", "description": "第二个平面法向量 (A,B,C)"},
    },
}, category="空间解析几何")
def angle_planes_3d(n1="1,0,0", n2="0,1,0"):
    """求两个平面的二面角。两平面的夹角取其法向量夹角（取锐角）。也可用于两直线方向向量夹角。"""
    a, b = _vec(n1), _vec(n2)
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if la < 1e-12 or lb < 1e-12:
        raise ValueError("方向/法向量不能为零。")
    cosv = float(np.clip((a @ b) / (la * lb), -1.0, 1.0))
    rad = float(np.arccos(cosv))
    deg = float(np.degrees(rad))
    if deg > 90:
        deg = 180 - deg
        rad = float(np.radians(deg))
    return {"text": f"u = {_fmt(a)}\nv = {_fmt(b)}\n\ncos θ = {cosv:.5g}\n夹角 θ = {deg:.4g}° （{rad:.5g} 弧度）\n\n说明：两平面夹角取其法向量夹角（锐角）；给直线方向向量同样适用。"}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "空间几何"
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


class Geometry3dPage(_FormPage):
    NAME = "空间几何"
    EMOJI = "\U0001F4CF"
    _TITLE = "📐 空间解析几何"
    _TOOLS = {
        "line_line_3d": (
            "异面/相交直线",
            "输入两条直线（过点 + 方向向量），求它们的最近点、最短距离；若相交则给出交点。",
            [("p1", "直线 L1 过点 (x,y,z)", "0,0,0"),
             ("d1", "直线 L1 方向向量", "1,1,0"),
             ("p2", "直线 L2 过点 (x,y,z)", "1,0,0"),
             ("d2", "直线 L2 方向向量", "1,0,0")],
            line_line_3d),
        "plane_from_points": (
            "三点定平面",
            "输入不共线的三个点，得到平面的法向量、点法式与一般式方程。",
            [("p1", "点 A (x,y,z)", "1,0,0"),
             ("p2", "点 B (x,y,z)", "0,1,0"),
             ("p3", "点 C (x,y,z)", "0,0,1")],
            plane_from_points),
        "point_plane_distance": (
            "点到平面距离",
            "输入目标点与平面（法向量 + 过一点），求距离与垂足。",
            [("point", "目标点 P (x,y,z)", "1,2,3"),
             ("normal", "平面法向量 (A,B,C)", "1,1,1"),
             ("through", "平面上一点 (x0,y0,z0)", "0,0,0")],
            point_plane_distance),
        "angle_planes_3d": (
            "平面夹角",
            "输入两个平面法向量，求二面角（也适用于两直线方向向量夹角）。",
            [("n1", "法向量 1 (A,B,C)", "1,0,0"),
             ("n2", "法向量 2 (A,B,C)", "0,1,0")],
            angle_planes_3d),
    }


PAGES = [Geometry3dPage]