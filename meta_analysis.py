# -*- coding: utf-8 -*-
"""元分析：固定/随机效应合并效应、Q 异质性、I2、tau2、加权均值与森林图数据（工具 + 页面）。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools
from scipy.stats import chi2

PLUGIN = {
    "id": "meta_analysis",
    "name": "元分析",
    "version": "1.0",
    "author": "zoilzo",
    "description": "固定/随机效应合并 / 异质性 I2 / Q / tau2（工具 + 页面，循证医学）",
}


def _studies(s):
    """解析研究：每行 "y,se" 效应量y与标准误se（换行分隔）。"""
    rows = [r.strip().replace(" ", ",").split(",") for r in s.replace("；", "\n").split("\n") if r.strip()]
    ys = [float(r[0]) for r in rows]
    ses = [float(r[1]) for r in rows]
    return ys, ses


@ai_tools._reg
@ai_tools._tool({"properties": {"studies": {"type": "string", "default": "0.2,0.1\n0.5,0.2\n0.3,0.15"}}, "required": ["studies"]}, category="元分析")
def fixed_effect(studies="0.2,0.1\n0.5,0.2\n0.3,0.15"):
    """固定效应模型合并效应量（逆方差加权）。"""
    ys, ses = _studies(studies)
    w = [1.0 / (s * s) for s in ses]
    wsum = sum(w)
    pooled = sum(w[i] * ys[i] for i in range(len(ys))) / wsum
    se = math.sqrt(1.0 / wsum)
    lo, hi = pooled - 1.96 * se, pooled + 1.96 * se
    return {"text": f"固定效应合并效应={pooled:.3f}，95%CI [{lo:.3f},{hi:.3f}]，SE={se:.3f}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"studies": {"type": "string", "default": "0.2,0.1\n0.5,0.2\n0.48,0.15"}}, "required": ["studies"]}, category="元分析")
def heterogeneity(studies="0.2,0.1\n0.5,0.2\n0.48,0.12"):
    """异质性检验：Cochran Q、I2、tau2。"""
    ys, ses = _studies(studies)
    w = [1.0 / (s * s) for s in ses]
    wsum = sum(w)
    pooled = sum(w[i] * ys[i] for i in range(len(ys))) / wsum
    q = sum(w[i] * (ys[i] - pooled) ** 2 for i in range(len(ys)))
    k = len(ys)
    p = 1 - chi2.cdf(q, k - 1)
    i2 = max(0.0, (q - (k - 1)) / q) * 100 if q else 0.0
    c = wsum - (sum(x * x for x in w) / wsum)
    tau2 = (q - (k - 1)) / c if c else 0.0
    level = "低" if i2 < 25 else ("中" if i2 < 75 else "高")
    return {"text": f"Q={q:.2f}（p={p:.4f}），I2={i2:.1f}%（{level}异质性），tau2={tau2:.4f}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"studies": {"type": "string", "default": "0.2,0.1\n0.5,0.25\n0.78,0.2"}}, "required": ["studies"]}, category="元分析")
def random_effect(studies="0.2,0.1\n0.5,0.2\n0.48,0.12"):
    """随机效应模型合并效应量（DerSimonian-Laird，用 tau2 调整权重）。"""
    ys, ses = _studies(studies)
    w = [1.0 / (s * s) for s in ses]
    wsum = sum(w)
    pooled = sum(w[i] * ys[i] for i in range(len(ys))) / wsum
    q = sum(w[i] * (ys[i] - pooled) ** 2 for i in range(len(ys)))
    k = len(ys)
    c = wsum - (sum(x * x for x in w) / wsum)
    tau2 = max(0.0, (q - (k - 1)) / c) if c else 0.0
    wr = [1.0 / (s * s * tau2 + 1.0) for s in ses]
    wsumr = sum(wr)
    pooled_r = sum(wr[i] * ys[i] for i in range(len(ys))) / wsumr
    se_r = math.sqrt(1.0 / wsumr)
    lo, hi = pooled_r - 1.96 * se_r, pooled_r + 1.96 * se_r
    return {"text": f"随机效应合并效应={pooled_r:.3f}，95%CI [{lo:.3f},{hi:.3f}]，SE={se_r:.3f}（tau2={tau2:.4f}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"studies": {"type": "string", "default": "0.2,0.1\n0.5,0.14\n0.6,0.1"}, "label": {"type": "string", "default": "研究1\n研究2\n研究3"}}, "required": ["studies"]}, category="元分析")
def forest_data(studies="0.2,0.1\n0.5,0.2\n0.3,0.15", label="研究1\n研究2\n研究3"):
    """森林图的制图数据：每行研究 效应量、CI 与权重。"""
    ys, ses = _studies(studies)
    labs = [x.strip() for x in label.split("\n") if x.strip()] or [f"研究{i+1}" for i in range(len(ys))]
    w = [1.0 / (s * s) for s in ses]
    wsum = sum(w)
    rows = []
    for i in range(len(ys)):
        lo, hi = ys[i] - 1.96 * ses[i], ys[i] + 1.96 * ses[i]
        rows.append(f"{labs[i]}: 效应={ys[i]:.3f} [{lo:.3f},{hi:.3f}] 权重={w[i]/wsum*100:.1f}%")
    return {"text": "森林图数据：\n" + "\n".join(rows)}


class MetaPage(ui.BasePage):
    NAME = "元分析"
    EMOJI = "\U0001F50D"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="\U0001F50D 元分析（Meta 分析）", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="每行一项研究：效应量,标准误（换行分隔），例：\n0.2,0.1\n0.5,0.14", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(0, 4))
        self.studies = ctk.CTkEntry(body)
        self.studies.insert(0, "0.2,0.1\n0.5,0.14\n0.3,0.15")
        self.studies.pack(fill="x")
        btn = ctk.CTkFrame(body, fg_color="transparent")
        btn.pack(fill="x", pady=ui.SPACE["sm"])
        ops = [
            ("固定效应", lambda: self._run(fixed_effect, [self.studies.get()])),
            ("异质性", lambda: self._run(heterogeneity, [self.studies.get()])),
            ("随机效应", lambda: self._run(random_effect, [self.studies.get()])),
            ("森林图数据", lambda: self._run(forest_data, [self.studies.get()])),
        ]
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btn, text=txt, command=fn, width=90).grid(row=0, column=i, padx=4, pady=4)
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


PAGES = [MetaPage]
