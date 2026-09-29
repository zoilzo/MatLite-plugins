# -*- coding: utf-8 -*-
"""诊断试验评价：混淆矩阵 / 敏感度/特异度/PPV/NPV / Youden / LR / ROC-AUC / Cohen kappa（工具 + 页面）。"""
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "diag_metrics",
    "name": "诊断试验评价",
    "version": "1.0",
    "author": "zoilzo",
    "description": "混淆矩阵/敏感度/特异度/Youden/ROC-AUC/kappa（工具 + 页面，临床与分类评价）",
}


def _cm(rows):
    """解析 2x2 混淆矩阵 (TP FP; FN TN)。"""
    vals = [float(x) for x in rows.replace(" ", ",").replace("；", ";").replace(";", ",").split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("需要 4 个数字：TP,FP,FN,TN")
    tp, fp, fn, tn = vals
    return tp, fp, fn, tn


@ai_tools._reg
@ai_tools._tool({"properties": {"cm": {"type": "string", "default": "80,20;10,90"}}, "required": ["cm"]}, category="诊断评价")
def conf_metrics(cm="80,20;10,90"):
    """由混淆矩阵计算敏感度/特异度/PPV/NPV/准确率/Youden。输入 TP,FP;FN,TN。"""
    tp, fp, fn, tn = _cm(cm)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0
    youden = sens + spec - 1
    return {"text": f"敏感度(Sens)={sens:.3f} 特异度(Spec)={spec:.3f} 阳性预测值(PPV)={ppv:.3f} 阴性预测值(NPV)={npv:.3f} 准确率={acc:.3f} Youden={youden:.3f}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"cm": {"type": "string", "default": "80,20;10,90"}}, "required": ["cm"]}, category="诊断评价")
def likelihood_ratios(cm="80,20;10,90"):
    """阳性似然比 LR+ 与阴性似然比 LR-。"""
    tp, fp, fn, tn = _cm(cm)
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    lr_p = sens / (1 - spec) if spec < 1 else float("inf")
    lr_n = (1 - sens) / spec if spec > 0 else float("inf")
    return {"text": f"LR+={lr_p:.3f}（越大越支持患病），LR-={lr_n:.3f}（越小越支持不患病）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"scores": {"type": "string", "default": "0.1,0.4,0.5,0.6,0.3"}, "labels": {"type": "string", "default": "0,1,1,1,0"}}, "required": []}, category="诊断评价")
def roc_auc(scores="0.1,0.4,0.5,0.6,0.3", labels="0,1,1,1,0"):
    """由连续预测值算 AUC（梯形法）并找 Youden 最优阈值。"""
    import numpy as np
    from scipy.stats import rankdata
    sc = np.array([float(x) for x in scores.replace("，", ",").split(",") if x.strip()], dtype=float)
    lab = np.array([int(x) for x in labels.replace("，", ",").split(",") if x.strip()])
    order = np.argsort(sc)
    sc, lab = sc[order], lab[order]
    r = rankdata(sc, method="average")
    pos = lab == 1
    auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()) if pos.any() and (~pos).any() else float("nan")
    # 找 Youden 最优阈值
    best_t, best_j = None, -1
    for t in sc:
        pred = sc >= t
        tp = ((pred) & (lab == 1)).sum(); fn = ((~pred) & (lab == 1)).sum()
        fp = ((pred) & (lab == 0)).sum(); tn = ((~pred) & (lab == 0)).sum()
        j = tp / (tp + fn + 1e-9) - fp / (fp + tn + 1e-9)
        if j > best_j:
            best_j, best_t = j, t
    return {"text": f"AUC={auc:.4f}，Youden 最优阈值为 {best_t:.3f}（Youden J={best_j:.3f}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"cm": {"type": "string", "default": "80,20;10,90"}}, "required": ["cm"]}, category="诊断评价")
def kappa_score(cm="80,20;10,90"):
    """Cohen kappa 一致性系数。"""
    tp, fp, fn, tn = _cm(cm)
    n = tp + fp + fn + tn
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"text": f"Cohen kappa={kappa:.3f}（>0.8 几乎一致，0.6-0.8 较一致，<0.4 较差）"}


class DiagPage(ui.BasePage):
    NAME = "诊断试验评价"
    EMOJI = "\U0001F52D"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="\U0001F52D 诊断试验评价", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="混淆矩阵 TP,FP;FN,TN：（例如 80,20;10,90）", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(0, 4))
        self.cm = ctk.CTkEntry(body); self.cm.insert(0, "80,20;10,90"); self.cm.pack(fill="x")
        btn = ctk.CTkFrame(body, fg_color="transparent")
        btn.pack(fill="x", pady=ui.SPACE["sm"])
        ops = [
            ("混淆指标", lambda: self._run(conf_metrics, [self.cm.get()])),
            ("似然比", lambda: self._run(likelihood_ratios, [self.cm.get()])),
            ("ROC-AUC", lambda: self._run(roc_auc, ["0.1,0.4,0.5,0.6,0.3", "0,1,1,1,0"])),
            ("Kappa", lambda: self._run(kappa_score, [self.cm.get()])),
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


PAGES = [DiagPage]
