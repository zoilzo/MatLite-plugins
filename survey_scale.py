# -*- coding: utf-8 -*-
"""问卷/量表信效度：Cronbach α / 分半信度(Spearman-Brown) / KMO+Bartlett / 题项分析 / 反向题计分。

给做问卷调查、毕业设计、量表研究的同学，验证问卷"靠不靠谱"。
输入统一：每位受访者一行（分号 ; 隔开），行内各题得分用逗号分隔。
（AI 工具 + 页面）
"""
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "survey_scale",
    "name": "问卷信效度",
    "version": "1.0",
    "author": "zoilzo",
    "description": "Cronbach α / 分半信度 / KMO+Bartlett / 题项分析 / 反向题计分（工具 + 页面）",
}


def _rows(s):
    """把字符串解析成二维 float 数组（每位受访者一行，分号隔开，行内逗号分隔）。"""
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
    s = s.replace(",", " ").replace("，", " ").replace(";", " ").replace("；", " ").replace("\n", " ")
    vals = [float(x) for x in s.split() if x.strip()]
    if not vals:
        raise ValueError("输入为空")
    return np.array(vals, dtype=float)


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"items": {"type": "string", "default": "4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"}}, "required": ["items"]}, category="问卷量表")
def cronbach_alpha(items="4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"):
    """Cronbach α 信度系数。items 为量表矩阵：每位受访者一行（分号隔开），行内各题得分逗号分隔。α≥0.8 很好，0.7~0.8 可接受，<0.6 需修订。"""
    X = _rows(items)
    n, k = X.shape
    if k < 2:
        return {"text": "至少需要 2 个题项。"}
    item_var = np.var(X, axis=0, ddof=1)
    total = X.sum(axis=1)
    total_var = np.var(total, ddof=1)
    if total_var <= 0:
        return {"text": "总分方差为 0，无法计算 α。"}
    alpha = (k / (k - 1.0)) * (1 - item_var.sum() / total_var)
    if 0.8 <= alpha:
        verdict = "信度很好（α≥0.8）"
    elif alpha >= 0.7:
        verdict = "信度可接受（0.7≤α<0.8）"
    else:
        verdict = "信度偏低（α<0.7），建议修订题项"
    return {"text": "Cronbach α = {0:.4f}（{1} 个受访者 × {2} 题）\n判断：{3}".format(alpha, n, k, verdict)}


@ai_tools._reg
@ai_tools._tool({"properties": {"items": {"type": "string", "default": "4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"}}, "required": ["items"]}, category="问卷量表")
def split_half_reliability(items="4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"):
    """分半信度：按奇偶分两半，Spearman-Brown 校正。评价量表内部一致性。"""
    X = _rows(items)
    n, k = X.shape
    if k < 2:
        return {"text": "至少需要 2 个题项。"}
    half1 = X[:, :(k + 1) // 2].sum(axis=1)
    half2 = X[:, (k + 1) // 2:].sum(axis=1)
    r = np.corrcoef(half1, half2)[0, 1]
    if r <= -1:  # 避免除零
        return {"text": "两半完全负相关，无法校正。"}
    sb = 2 * r / (1 + r) if abs(1 + r) > 1e-9 else float("inf")
    return {"text": "两半相关 r = {0:.4f}\nSpearman-Brown 分半信度 = {1:.4f}".format(r, sb)}


@ai_tools._reg
@ai_tools._tool({"properties": {"items": {"type": "string", "default": "4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,4,4,4;3,3,4,3;4,5,5,4;5,5,4,5"}}, "required": ["items"]}, category="问卷量表")
def kmo_bartlett(items="4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,4,4,4;3,3,4,3;4,5,5,4;5,5,4,5"):
    """KMO 与 Bartlett 球形检验，判断量表适合因子分析。KMO>0.8 很适合，<0.6 不适合；Bartlett p<0.05 可做因子分析。"""
    from scipy.stats import chi2
    from scipy.linalg import pinv
    X = _rows(items)
    n, k = X.shape
    if k < 2:
        return {"text": "至少需要 2 个题项。"}
    R = np.corrcoef(X.T)
    det = np.linalg.det(R)
    if det <= 0:
        return {"text": "相关矩阵非正定（题项间共线过强或样本过少），Bartlett 无法计算。"}
    chi2_val = -(n - 1 - (2 * k + 5) / 6.0) * np.log(det)
    df = k * (k - 1) / 2.0
    p_bar = float(chi2.sf(chi2_val, df))
    # KMO
    Rinv = pinv(R)
    d = np.sqrt(np.einsum("ii->i", Rinv))
    P = -Rinv / np.outer(d, d)
    iu = np.triu_indices(k, 1)
    r2 = (R[iu] ** 2).sum()
    p2 = (P[iu] ** 2).sum()
    kmo = r2 / (r2 + p2) if (r2 + p2) > 0 else 0.0
    if kmo >= 0.8:
        kv = "很适合做因子分析"
    elif kmo >= 0.7:
        kv = "适合做因子分析"
    elif kmo >= 0.6:
        kv = "勉强适合"
    else:
        kv = "不适合做因子分析"
    return {"text": "KMO = {0:.3f}（{1}）\nBartlett：χ² = {2:.2f}，df = {3:.0f}，p = {4:.4f}\n判断：p{5} 0.05 → {6}".format(kmo, kv, chi2_val, df, p_bar, "<" if p_bar < 0.05 else "≥", "适合因子分析" if p_bar < 0.05 else "不适合因子分析")}


@ai_tools._reg
@ai_tools._tool({"properties": {"items": {"type": "string", "default": "4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"}}, "required": ["items"]}, category="问卷量表")
def item_analysis(items="4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3"):
    """题项分析：每题均值/标准差、校正后题总相关、删题后 Cronbach α。用于精选/剔除坏题。"""
    X = _rows(items)
    n, k = X.shape
    if k < 3:
        return {"text": "至少需要 3 个题项。"}
    total = X.sum(axis=1)
    total_var = np.var(total, ddof=1)
    item_var = np.var(X, axis=0, ddof=1)
    alpha_all = (k / (k - 1.0)) * (1 - item_var.sum() / total_var)
    lines = ["题项分析（n={0}）\n原始 Cronbach α = {1:.4f}".format(n, alpha_all)]
    lines.append("  题号   均值    标准差  校正题总相关   删后α")
    for j in range(k):
        col = X[:, j]
        total_minus = total - col
        corr = float(np.corrcoef(col, total_minus)[0, 1])
        # 删去第 j 题后的 α
        item_var_drop = np.delete(item_var, j)
        alpha_drop = (k - 1) / (k - 2.0) * (1 - item_var_drop.sum() / np.var(total_minus, ddof=1))
        lines.append("  {0:>4}  {1:6.2f}  {2:6.2f}   {3:8.3f}   {4:.4f}".format(j + 1, np.mean(col), np.std(col, ddof=1), corr, alpha_drop))
    lines.append("提示：校正题总相关<0.3、删后α明显升高 的题可考虑剔除。")
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"scores": {"type": "string", "default": "5,4,3,2,1"}, "max_score": {"type": "number", "default": 5}}, "required": ["scores", "max_score"]}, category="问卷量表")
def reverse_scale(scores="5,4,3,2,1", max_score=5):
    """反向题计分：Likert 量表反向题 = 最大值+1 - 原分。scores 为原始得分列表，max_score 为量表最大值(如5)。"""
    s = _list_of(scores)
    ms = float(max_score)
    rev = ms + 1 - s
    return {"text": "原分：{0}\n反向计分：{1}".format(", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in s), ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in rev))}


# ============================================================
# 页面
# ============================================================
class SurveyScalePage(ui.BasePage):
    NAME = "问卷信效度"
    EMOJI = "\U0001F4CA"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📊 问卷信效度", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="量表矩阵：每位受访者一行（分号 ; 隔开），行内各题得分逗号分隔：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.items = ctk.CTkEntry(body)
        self.items.insert(0, "4,3,3,4;3,4,4,5;5,4,5,4;2,3,2,3;4,5,4,5;3,3,4,3")
        self.items.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="反向题原分（逗号分隔，可选）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.reverse = ctk.CTkEntry(body)
        self.reverse.insert(0, "5,4,3,2,1")
        self.reverse.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ops = [
            ("Cronbach α", lambda: self._run(cronbach_alpha, [self.items.get()])),
            ("分半信度", lambda: self._run(split_half_reliability, [self.items.get()])),
            ("KMO+Bartlett", lambda: self._run(kmo_bartlett, [self.items.get()])),
            ("题项分析", lambda: self._run(item_analysis, [self.items.get()])),
            ("反向题计分", lambda: self._run(reverse_scale, [self.reverse.get(), 5])),
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


PAGES = [SurveyScalePage]