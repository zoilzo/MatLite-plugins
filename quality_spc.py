# -*- coding: utf-8 -*-
"""质量管理统计（SPC）：X̄-R 控制图界限 / 过程能力 Cp·Cpk / 缺陷率 DPU·DPMO / Pareto 分析。

给工科、质量管理、生产管理的同学做过程控制与能力分析。
（AI 工具 + 页面）
"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "quality_spc",
    "name": "质量管理统计",
    "version": "1.0",
    "author": "zoilzo",
    "description": "X̄-R 控制图界限 / 过程能力 Cp·Cpk / 缺陷率 DPU·DPMO / Pareto（工具 + 页面）",
}

# 控制图常数（子组大小 n=2~10，标准 AIAG/Montgomery 值）
# 控制图常数 A2/D3/D4 采用标准查表值（避免数值积分误差）
from functools import lru_cache as _lru
from scipy.stats import norm as _norm
import math as _math

# 控制图常数（子组大小 n=2..10，标准 AIAG/Montgomery 值）
_RC = {2: (1.88, 0.0, 3.267), 3: (1.023, 0.0, 2.574), 4: (0.729, 0.0, 2.282), 5: (0.577, 0.0, 2.114), 6: (0.483, 0.0, 2.004), 7: (0.419, 0.076, 1.924), 8: (0.373, 0.136, 1.864), 9: (0.337, 0.184, 1.816), 10: (0.308, 0.223, 1.777)}
def _rc(n):
    return _RC[n]



def _rows(s):
    s = s.replace("；", ";").replace("\n", ";").replace("，", ",")
    out = []
    for row in s.split(";"):
        row = row.strip()
        if not row:
            continue
        vals = [float(x) for x in row.replace(",", " ").split() if x.strip()]
        if vals:
            out.append(vals)
    if not out:
        raise ValueError("输入为空")
    return out


def _list_of(s):
    s = s.replace(",", " ").replace("，", " ").replace(";", " ").replace("；", " ").replace("\n", " ")
    vals = [float(x) for x in s.split() if x.strip()]
    if not vals:
        raise ValueError("输入为空")
    return np.array(vals, dtype=float)


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"data": {"type": "string", "default": "3.9,4.2,4.1;4.0,4.3,3.8;4.0,4.1,4.2;3.8,4.2,4.0;4.1,3.9,4.4"}}, "required": ["data"]}, category="质量管理")
def xbar_r_limits(data="3.9,4.2,4.1;4.0,4.3,3.8;4.0,4.1,4.2;3.8,4.2,4.0;4.1,3.9,4.4"):
    """X̄-R 控制图界限。data 为子组矩阵：每个子组一行（分号隔开），行内为观测值。返回 X̄ 与 R 图的 UCL/CL/LCL。"""
    rows = _rows(data)
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        return {"text": "各子组观测值个数需一致（n=%d）。" % n}
    if n not in tuple(_RC):
        return {"text": "支持子组大小 n=2~10，当前 n=%d。" % n}
    Xmean = [float(np.mean(r)) for r in rows]
    Rrange = [float(np.max(r) - np.min(r)) for r in rows]
    xbar = float(np.mean(Xmean))
    rbar = float(np.mean(Rrange))
    a2, d3, d4 = _rc(n)
    x_ucl = xbar + a2 * rbar
    x_lcl = xbar - a2 * rbar
    r_ucl = d4 * rbar
    r_lcl = d3 * rbar
    lines = ["X̄-R 控制图（子组 n=%d，共 %d 组）" % (n, len(rows))]
    lines.append("  X̄ 图：CL = %.4f，UCL = %.4f，LCL = %.4f" % (xbar, x_ucl, x_lcl))
    lines.append("  R  图：CL = %.4f，UCL = %.4f，LCL = %.4f" % (rbar, r_ucl, r_lcl))
    out_x = [i + 1 for i, v in enumerate(Xmean) if not (x_lcl <= v <= x_ucl)]
    out_r = [i + 1 for i, v in enumerate(Rrange) if not (r_lcl <= v <= r_ucl)]
    lines.append("  X̄ 越界子组：%s" % (", ".join(map(str, out_x)) if out_x else "无"))
    lines.append("  R  越界子组：%s" % (", ".join(map(str, out_r)) if out_r else "无"))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"data": {"type": "string", "default": "4.0,4.2,3.9,4.1,4.3,3.8,4.0,4.1"}, "usl": {"type": "number", "default": 4.6}, "lsl": {"type": "number", "default": 3.4}}, "required": ["data", "usl", "lsl"]}, category="质量管理")
def capability_cp_cpk(data="4.0,4.2,3.9,4.1,4.3,3.8,4.0,4.1", usl=4.6, lsl=3.4):
    """过程能力：Cp、Cpk、Cpu、Cpl。data 为测量值列表，usl/lsl 为上/下规格限。Cpk≥1.33 能力充足，<1 能力不足。"""
    x = _list_of(data)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    usl, lsl = float(usl), float(lsl)
    if sigma <= 0:
        return {"text": "数据标准差为 0，无法计算能力指数。"}
    cp = (usl - lsl) / (6 * sigma)
    cpu = (usl - mu) / (3 * sigma)
    cpl = (mu - lsl) / (3 * sigma)
    cpk = min(cpu, cpl)
    if cpk >= 1.33:
        verdict = "能力充足"
    elif cpk >= 1.0:
        verdict = "能力尚可"
    else:
        verdict = "能力不足，需改进"
    return {"text": "均值 μ = %.4f，标准差 σ = %.4f\nCp = %.4f   Cpk = %.4f\nCpu = %.4f   Cpl = %.4f\n判断：%s（Cpk≥1.33 充足；1.0~1.33 尚可；<1 不足）" % (mu, sigma, cp, cpk, cpu, cpl, verdict)}


@ai_tools._reg
@ai_tools._tool({"properties": {"defects": {"type": "number", "default": 12}, "units": {"type": "number", "default": 500}, "opportunities": {"type": "number", "default": 4}}, "required": ["defects", "units"]}, category="质量管理")
def defect_metrics(defects=12, units=500, opportunities=4):
    """缺陷率指标：DPU、DPO、DPMO、产率与 SIGMA 水平（近似）。defects 总缺陷数，units 单位数，opportunities 每单位潜在缺陷机会。"""
    d = float(defects)
    u = float(units)
    opp = float(opportunities) if opportunities else 1
    if u <= 0 or opp <= 0:
        return {"text": "units 与 opportunities 需为正数。"}
    dpu = d / u
    dpo = d / (u * opp)
    dpmo = dpo * 1e6
    yield_rate = 1 - dpo
    lines = ["缺陷率分析（缺陷 %g / 单位 %g / 机会/单位 %g）：" % (d, u, opp)]
    lines.append("  DPU（每单位缺陷）= %.4f" % dpu)
    lines.append("  DPO（每机会缺陷）= %.6f" % dpo)
    lines.append("  DPMO（每百万机会缺陷）= %.1f" % dpmo)
    lines.append("  良率（一次合格）≈ %.4f%%" % (yield_rate * 100))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"data": {"type": "string", "default": "外观缺陷:42;尺寸偏差:25;划伤:16;材料不良:8;其他:6"}}, "required": ["data"]}, category="质量管理")
def pareto_analysis(data="外观缺陷:42;尺寸偏差:25;划伤:16;材料不良:8;其他:6"):
    """Pareto 分析：按缺陷类别计数排序，计算占比与累计占比，找出占 80% 的主要类别。data 为 类别:数量，用分号隔开。"""
    items = []
    for part in data.replace("；", ";").replace("，", ",").split(";"):
        part = part.strip()
        if not part:
            continue
        if ":" not in part and "：" not in part:
            continue
        # 支持半角/全角冒号
        for sep in (":", "："):
            if sep in part:
                name, cnt = part.split(sep, 1)
                break
        else:
            continue
        try:
            items.append((name.strip(), int(cnt)))
        except ValueError:
            continue
    if not items:
        return {"text": "请按 类别:数量 输入，如 外观:42。"}
    items.sort(key=lambda t: t[1], reverse=True)
    total = sum(c for _, c in items)
    if total <= 0:
        return {"text": "总数需 > 0。"}
    lines = ["Pareto 分析（共 %d 项，总 %d）：" % (len(items), total)]
    cum = 0
    for i, (name, cnt) in enumerate(items):
        pct = cnt / total * 100
        cum += pct
        lines.append("  %-3d  %-14s %5d  %5.1f%%  累计 %5.1f%%" % (i + 1, name, cnt, pct, cum))
    # 找累计到 80%
    acc = 0
    focus = []
    for name, cnt in items:
        acc += cnt
        focus.append(name)
        if acc / total >= 0.8:
            break
    lines.append("")
    lines.append("占 80%% 的主要类别（优先改进）：%s" % (", ".join(focus)))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class QualitySpcPage(ui.BasePage):
    NAME = "质量管理统计"
    EMOJI = "\U0001F4C8"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📈 质量管理统计", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="X̄-R 子组矩阵（每组一行分号隔开，组内用逗号）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.subgroups = ctk.CTkEntry(body)
        self.subgroups.insert(0, "3.9,4.2,4.1;4.0,4.3,3.8;4.0,4.1,4.2;3.8,4.2,4.0;4.1,3.9,4.4")
        self.subgroups.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="测量值 + 规格限（USL / LSL）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.data = ctk.CTkEntry(body)
        self.data.insert(0, "4.0,4.2,3.9,4.1,4.3,3.8,4.0,4.1")
        self.data.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        self.usl = ctk.CTkEntry(row, width=140); self.usl.insert(0, "4.6")
        self.usl.grid(row=0, column=0, padx=(0, ui.SPACE["sm"]))
        self.lsl = ctk.CTkEntry(row, width=140); self.lsl.insert(0, "3.4")
        self.lsl.grid(row=0, column=1)
        ctk.CTkLabel(body, text="缺陷（数）/单位/机会 与 Pareto（类别:数量，分号隔开）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.pareto = ctk.CTkEntry(body)
        self.pareto.insert(0, "外观缺陷:42;尺寸偏差:25;划伤:16;材料不良:8;其他:6")
        self.pareto.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ops = [
            ("X̄-R 控制图", lambda: self._run(xbar_r_limits, [self.subgroups.get()])),
            ("Cp/Cpk", lambda: self._run(capability_cp_cpk, [self.data.get(), self.usl.get(), self.lsl.get()])),
            ("缺陷率", lambda: self._run(defect_metrics, [12, 500, 4])),
            ("Pareto", lambda: self._run(pareto_analysis, [self.pareto.get()])),
        ]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btns, text=txt, command=fn, width=104).grid(row=0, column=i, padx=4, pady=4)
        self.out = ui.mono_textbox(body, ui.FONT["body"])
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


PAGES = [QualitySpcPage]