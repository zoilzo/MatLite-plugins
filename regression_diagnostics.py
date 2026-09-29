# -*- coding: utf-8 -*-
"""回归诊断：多重共线性 VIF / Durbin-Watson / Breusch-Pagan 异方差 / 残差正态性 / Cook 距离。

给做多元线性回归、论文数据分析的同学，检验模型是否真的"合格"。
输入统一用"自变量的每一行用分号隔开、行内用逗号或空格隔开"。
（AI 工具 + 页面）
"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "regression_diagnostics",
    "name": "回归诊断",
    "version": "1.0",
    "author": "zoilzo",
    "description": "多重共线性 VIF / Durbin-Watson / Breusch-Pagan / 残差正态性 / Cook 距离（工具 + 页面）",
}


# ---------------- 解析辅助 ----------------
def _rows(s):
    """把字符串解析成二维 float 数组：行以 ; 或 ；或换行分隔，行内以 , 空格分隔。"""
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
    return np.array(out, dtype=float)


def _list_of(s):
    """把字符串解析成一维 float 数组（逗号/空格/分号分隔）。"""
    s = s.replace(",", " ").replace("，", " ").replace(";", " ").replace("；", " ").replace("\n", " ")
    vals = [float(x) for x in s.split() if x.strip()]
    if not vals:
        raise ValueError("输入为空")
    return np.array(vals, dtype=float)


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"predictors": {"type": "string", "default": "1,2,3;2,4,6;3,6,9;4,8,2;5,9,0"}}, "required": ["predictors"]}, category="回归诊断")
def vif_metrics(predictors="1,2,3;2,4,6;3,6,9;4,8,2;5,9,0"):
    """多重共线性：计算每个自变量的方差膨胀因子 VIF。predictors 为自变量矩阵（不含因变量），行隔开、行内逗号分隔。VIF>10 需警惕共线性。"""
    X = _rows(predictors)
    n, p = X.shape
    if p < 2:
        return {"text": "至少要 2 个自变量才能算 VIF。"}
    Xc = np.column_stack([np.ones(n), X])  # 含常数项
    vifs = []
    for j in range(p):
        y_j = X[:, j]
        others = np.delete(np.column_stack([np.ones(n), X]), j + 1, axis=1)
        beta, *_ = np.linalg.lstsq(others, y_j, rcond=None)
        pred = others @ beta
        r2 = 1 - np.sum((y_j - pred) ** 2) / np.sum((y_j - np.mean(y_j)) ** 2)
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vifs.append(vif)
    lines = ["多重共线性（VIF，>10 需警惕，>5 需关注）："]
    for j, v in enumerate(vifs):
        flag = " ⚠ 高" if v >= 10 else (" 注意" if v >= 5 else " 正常")
        lines.append("  自变量{j}：VIF = {v:.2f}{f}".format(j=j + 1, v=v, f=flag))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"residuals": {"type": "string", "default": "1.2,-0.5,0.4,1.8,-1.1,0.7,-0.2,1.5"}}, "required": ["residuals"]}, category="回归诊断")
def dw_test(residuals="1.2,-0.5,0.4,1.8,-1.1,0.7,-0.2,1.5"):
    """Durbin-Watson 自相关检验。residuals 为残差序列。DW≈2 无自相关；越接近 0 越正自相关，越接近 4 越负自相关。"""
    e = _list_of(residuals)
    n = len(e)
    if n < 3:
        return {"text": "残差至少需 3 个。"}
    dw = np.sum((e[1:] - e[:-1]) ** 2) / np.sum(e ** 2)
    if 1.5 <= dw <= 2.5:
        verdict = "无显著自相关（OK）"
    elif dw < 1.5:
        verdict = "可能存在正自相关（DW 偏小）"
    else:
        verdict = "可能存在负自相关（DW 偏大）"
    return {"text": "Durbin-Watson = {0:.3f}\n判断：{1}".format(dw, verdict)}


@ai_tools._reg
@ai_tools._tool({"properties": {"predictors": {"type": "string", "default": "1,2,3;2,4,6;3,6,9;4,8,11;5,9,14"}, "y": {"type": "string", "default": "4,8,13,16,20"}}, "required": ["predictors", "y"]}, category="回归诊断")
def bp_test(predictors="1,2,3;2,4,6;3,6,9;4,8,11;5,9,14", y="4,8,13,16,20"):
    """Breusch-Pagan 异方差检验。predictors 为自变量矩阵（行隔开、行内逗号），y 为因变量。p<0.05 提示存在异方差。"""
    X = _rows(predictors)
    y = _list_of(y)
    n, p = X.shape
    if len(y) != n:
        return {"text": "predictors 行数与 y 长度不一致。"}
    Xc = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    resid2 = resid ** 2
    beta2, *_ = np.linalg.lstsq(Xc, resid2, rcond=None)
    r2 = 1 - np.sum((resid2 - Xc @ beta2) ** 2) / np.sum((resid2 - np.mean(resid2)) ** 2)
    lm = n * r2
    from scipy.stats import chi2
    pval = float(chi2.sf(lm, p))
    verdict = "存在异方差（p<0.05）" if pval < 0.05 else "未见显著异方差（p≥0.05）"
    return {"text": "Breusch-Pagan：LM = {0:.3f}，p = {1:.4f}\n判断：{2}".format(lm, pval, verdict)}


@ai_tools._reg
@ai_tools._tool({"properties": {"residuals": {"type": "string", "default": "1.2,-0.5,0.4,1.8,-1.1,0.7,-0.2,1.5"}}, "required": ["residuals"]}, category="回归诊断")
def resid_normality(residuals="1.2,-0.5,0.4,1.8,-1.1,0.7,-0.2,1.5"):
    """残差正态性检验：Shapiro-Wilk 与 Jarque-Bera。p≥0.05 接受残差接近正态。"""
    from scipy import stats as st
    e = _list_of(residuals)
    n = len(e)
    lines = []
    if n >= 3 and n <= 5000:
        w, p_sh = st.shapiro(e)
        lines.append("Shapiro-Wilk：W = {0:.4f}，p = {1:.4f} → {2}".format(w, p_sh, "近似正态" if p_sh >= 0.05 else "偏离正态"))
    else:
        lines.append("Shapiro-Wilk 需 3~5000 个样本，当前 {0} 个，跳过。".format(n))
    jb, p_jb = st.jarque_bera(e)
    lines.append("Jarque-Bera：JB = {0:.4f}，p = {1:.4f} → {2}".format(jb, p_jb, "近似正态" if p_jb >= 0.05 else "偏离正态"))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"predictors": {"type": "string", "default": "1,2,3;2,4,6;3,6,4;4,8,3;5,9,14;6,10,2"}, "y": {"type": "string", "default": "5,9,13,16,20,22"}}, "required": ["predictors", "y"]}, category="回归诊断")
def cooks_distance(predictors="1,2,3;2,4,6;3,6,4;4,8,3;5,9,14;6,10,2", y="5,9,13,16,20,22"):
    """Cook 距离识别强影响点/异常值。predictors 为自变量矩阵，y 为因变量。Cook>4/n 视为强影响点。"""
    X = _rows(predictors)
    y = _list_of(y)
    n, p = X.shape
    if len(y) != n:
        return {"text": "predictors 行数与 y 长度不一致。"}
    Xc = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)
    hat = np.diag(Xc @ XtX_inv @ Xc.T)
    beta = XtX_inv @ Xc.T @ y
    resid = y - Xc @ beta
    s2 = np.sum(resid ** 2) / (n - p - 1)
    cook = (resid ** 2 / (p + 1) / s2) * (hat / (1 - hat) ** 2)
    thr = 4.0 / n
    lines = ["Cook 距离（> {0:.3f} = 4/n 视为强影响点）：".format(thr)]
    for i, c in enumerate(cook):
        flag = " ⚠ 强影响点" if c > thr else ""
        lines.append("  #{0}：{1:.3f}{2}".format(i + 1, c, flag))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class RegressionDiagPage(ui.BasePage):
    NAME = "回归诊断"
    EMOJI = "\U0001F9EE"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🧾 回归诊断", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="自变量矩阵（每行用分号 ; 隔开，行内用逗号）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.predictors = ctk.CTkEntry(body)
        self.predictors.insert(0, "1,2,3;2,4,6;3,6,9;4,8,11;5,9,14")
        self.predictors.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="因变量 y（逗号分隔）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.y = ctk.CTkEntry(body)
        self.y.insert(0, "4,8,13,16,20")
        self.y.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="残差（逗号分隔）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.residuals = ctk.CTkEntry(body)
        self.residuals.insert(0, "1.2,-0.5,0.4,1.8,-1.1,0.7")
        self.residuals.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ops = [
            ("VIF 共线性", lambda: self._run(vif_metrics, [self.predictors.get()])),
            ("Durbin-Watson", lambda: self._run(dw_test, [self.residuals.get()])),
            ("Breusch-Pagan", lambda: self._run(bp_test, [self.predictors.get(), self.y.get()])),
            ("残差正态性", lambda: self._run(resid_normality, [self.residuals.get()])),
            ("Cook 距离", lambda: self._run(cooks_distance, [self.predictors.get(), self.y.get()])),
        ]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btns, text=txt, command=fn, width=96).grid(row=0, column=i, padx=4, pady=4)
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


PAGES = [RegressionDiagPage]