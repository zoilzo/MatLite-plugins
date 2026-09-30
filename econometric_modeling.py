# -*- coding: utf-8 -*-
"""经济计量建模：ADF 单位根检验、协整(Engle-Granger)、差分/平稳化、Chow 结构断点、Granger 因果。
AI 工具 + 页面（交叉领域，教学友好）。依赖 statsmodels。"""
import numpy as np
import pandas as pd
from scipy import stats as st
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "econometric_modeling",
    "name": "经济计量",
    "version": "1.0",
    "author": "zoilzo",
    "description": "经济计量建模：ADF 单位根检验、Engle-Granger 协整、差分/平稳化、Chow 结构断点、Granger 因果检验（工具 + 页面）",
}


try:
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
    _SM = True
except Exception:  # pragma: no cover
    _SM = False


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _nums(v):
    """解析一维序列：接受逗号/空格分隔字符串或列表。"""
    if isinstance(v, (list, tuple)):
        return np.asarray([float(x) for x in v], dtype=float)
    s = str(v).replace("，", ",").replace(";", ",").replace("\n", ",")
    return np.asarray([float(x) for x in s.split(",") if x.strip()], dtype=float)


def _rs(seed=0, n=60):
    """默认随机游走序列（非平稳）：用固定种子保证可复现。"""
    rng = np.random.RandomState(seed)
    return np.cumsum(rng.randn(n))


def _two_series(seed=1, n=80):
    """默认为两个协整序列（共享随机趋势），用于协整/Granger 演示。"""
    rng = np.random.RandomState(seed)
    e = rng.randn(n)
    x = np.cumsum(rng.randn(n))
    y = 0.8 * x + e  # 协整关系 y = 0.8x + 平稳误差
    return x, y


# ------------------------------------------------------------
# 工具 1：ADF 单位根检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "series": {"type": "string", "description": "时间序列（逗号分隔）；留空用默认随机游走"},
        "trend": {"type": "string", "description": "回归形式：c 常数 / ct 常数+趋势 / n 无"},
        "maxlag": {"type": "number", "description": "最大滞后阶数（留空自动选）"},
    },
}, category="经济计量")
def adf_test(series="", trend="c", maxlag=""):
    """ADF 增广迪基-富勒单位根检验：检验序列是否平稳。p<0.05 拒绝单位根→平稳；否则是非平稳(需差分)。"""
    if not _SM:
        return {"text": "未安装 statsmodels，无法做 ADF 检验。"}
    x = _rs() if not str(series).strip() else _nums(series)
    if x.size < 8:
        raise ValueError("序列过短，至少 8 个观测。")
    tt = str(trend or "c").strip().lower()
    reg = "c"
    if tt in ("ct", "c+t", "constant+trend"):
        reg = "ct"
    elif tt in ("n", "nc", "none"):
        reg = "n"
    maxl = None
    if str(maxlag).strip():
        maxl = int(maxlag)
    res = adfuller(x, autolag="AIC", regression=reg, maxlag=maxl)
    stat_, p_, lags, nobs, crit_, icbest = res
    # 判断
    if p_ < 0.05:
        verdict = "拒绝单位根（p<0.05）→ 平稳（I(0)）"
    else:
        verdict = "不能拒绝单位根（p≥0.05）→ 非平稳，需差分或协整建模"
    crit_str = " ".join("%s=%g" % (k, v) for k, v in crit_.items())
    lines = ["ADF 单位根检验：", "",
             "ADF 统计量 = %.4g" % stat_,
             "p 值 = %.4g" % p_,
             "滞后阶数 = %d，观测 = %d" % (lags, nobs), "",
             "临界值：" + crit_str, "",
             "➤ 判定：" + verdict]
    if p_ >= 0.05:
        lines.append("➤ 建议：若非平稳，先做一次差分再检验，或用协整/误差修正模型。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：Engle-Granger 协整检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "series1": {"type": "string", "description": "序列 y（逗号分隔）；留空用默认协整对"},
        "series2": {"type": "string", "description": "序列 x（逗号分隔）"},
        "trend": {"type": "string", "description": "协整回归形式：c 常数 / ct 常数+趋势 / n 无"},
    },
}, category="经济计量")
def cointegration_test(series1="", series2="", trend="c"):
    """Engle-Granger 两变量协整检验：检验两个非平稳序列是否存在长期稳定的线性关系。
    p<0.05 拒绝无协整 → 存在协整(长期均衡)；否则需误差修正模型。"""
    if not _SM:
        return {"text": "未安装 statsmodels，无法做协整检验。"}
    if not str(series1).strip() or not str(series2).strip():
        x, y = _two_series()
        notch = "（默认演示：y = 0.8x + 平稳误差）"
    else:
        y = _nums(series1)
        x = _nums(series2)
        notch = ""
    if x.size != y.size or x.size < 10:
        raise ValueError("两序列需等长且长度≥10。")
    reg_map = {"c": "c", "ct": "ct", "n": "n"}
    reg = reg_map.get(str(trend or "c").lower(), "c")
    res = coint(y, x, trend=reg)
    tobs, p_, crit_ = res
    # critical often comes as a 3-element array (1%, 5%, 10%) — normalize
    crit_arr = np.asarray(crit_, dtype=float).ravel()
    labels = ["1%", "5%", "10%"]
    crit_str = "  ".join("%s=%g" % (labels[i], crit_arr[i]) for i in range(min(len(labels), crit_arr.size)))
    # 估计长期系数 beta（OLS: y = a + b x + e）
    Xd = np.column_stack([np.ones(x.size), x]) if reg != "n" else x.reshape(-1, 1)
    beta_, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    if reg != "n":
        a_, b_ = beta_[0], beta_[1]
    else:
        a_, b_ = 0.0, beta_[0]
    verdict = "存在协整（p<0.05）→ y 与 x 有长期均衡关系" if p_ < 0.05 else "不能拒绝无协整（p≥0.05）→ 需差分处理"
    lines = ["Engle-Granger 协整检验%s：" % notch, "",
             "协整回归：y = %.4g + %.4g·x" % (a_, b_), "",
             "检验统计量 = %.4g" % tobs,
             "p 值 = %.4g" % p_, "",
             "临界值：" + crit_str, "",
             "➤ 判定：" + verdict]
    if p_ < 0.05:
        lines.append("➤ 建议：可用误差修正模型(ECM)刻画短期偏离与长期均衡的调整。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：差分 / 平稳化
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "series": {"type": "string", "description": "时间序列（逗号分隔）；留空用默认随机游走"},
        "order": {"type": "number", "description": "差分阶数 d（1 或 2）"},
    },
}, category="经济计量")
def difference_transform(series="", order="1"):
    """差分平稳化：对非平稳序列做 $d$ 阶差分，输出差分后序列并再度 ADF 检验其平稳性。
    I(d) 表示需做 d 次差分才平稳。"""
    if not _SM:
        return {"text": "未安装 statsmodels，无法做差分检验。"}
    x = _rs() if not str(series).strip() else _nums(series)
    if x.size < 10:
        raise ValueError("序列过短，至少 10 个观测。")
    d = int(order or 1)
    if d < 1 or d > 2:
        d = 1
    y = x
    steps = []
    for k in range(d):
        y = np.diff(y)
        steps.append(y)
    # 差分后 ADF
    res = adfuller(y, autolag="AIC", regression="c")
    stat_, p_, lags, nobs, crit_, _ = res
    verdict = "平稳（p<0.05）" if p_ < 0.05 else "仍非平稳（需再差分或建模）"
    lines = ["差分平稳化（d = %d）：" % d, "",
             "原序列长度 = %d，%d 阶差分后长度 = %d" % (x.size, d, y.size), "",
             "差分后 ADF：统计量 = %.4g，p = %.4g" % (stat_, p_), "",
             "差分后序列（前 12 项）：%s" % np.array2string(y[:12], precision=4, separator=', '), "",
             "➤ 判定：" + verdict]
    if p_ >= 0.05 and d == 1:
        lines.append("➤ 建议：可试 2 阶差分，或选协整/误差修正模型保留长期信息。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：Chow 结构断点检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "y": {"type": "string", "description": "因变量序列（逗号分隔）"},
        "x": {"type": "string", "description": "自变量序列（逗号分隔）"},
        "break_point": {"type": "number", "description": "断点位置（从1起算的观测序号）"},
    },
}, category="经济计量")
def chow_test_break(y="", x="", break_point=""):
    """Chow 结构断点检验：把样本在某点分成两段，检验回归系数是否发生结构性突变。
    p<0.05 拒绝无断点 → 存在结构断点（模型发生改变）。"""
    if not str(y).strip() or not str(x).strip():
        t = np.arange(1, 41)
        x = t + 2.0 * np.sin(t / 4)
        y = 1.5 + 0.8 * t + 0.3 * np.random.RandomState(0).randn(40)
        y = np.where(t > 20, y + 4.0, y)  # 第 20 点后截距上移 → 断点
    else:
        y = _nums(y)
        x = _nums(x)
    if x.size != y.size or x.size < 8:
        raise ValueError("两序列需等长且长度≥10。")
    n = x.size
    if str(break_point).strip():
        b = int(break_point)
    else:
        b = n // 2
    b = max(2, min(b, n - 2))
    k = 1  # 斜率 + 截距
    Xc = np.column_stack([np.ones(n), x])
    X1 = np.column_stack([np.ones(b), x[:b]])
    X2 = np.column_stack([np.ones(n - b), x[b:]])

    def _rss(Xm, yy):
        beta, *_ = np.linalg.lstsq(Xm, yy, rcond=None)
        return float(np.sum((yy - Xm @ beta) ** 2))

    rss_full = _rss(Xc, y)
    rss1 = _rss(X1, y[:b])
    rss2 = _rss(X2, y[b:])
    rss_u = rss1 + rss2
    kk = 2  # 每段参数数（截距+斜率）
    dof_num = kk
    dof_den = n - 2 * kk
    if dof_den <= 0 or rss_full <= 0:
        raise ValueError("样本过短或退化，无法做 Chow 检验。")
    F = ((rss_full - rss_u) / dof_num) / (rss_u / dof_den)
    pval = float(st.f.sf(F, dof_num, dof_den))
    verdict = "存在结构断点（p<0.05）" if pval < 0.05 else "未检测到显著结构断点（p≥0.05）"
    lines = ["Chow 结构断点检验（断点 t = %d）：" % b, "",
             "全样本 RSS = %.4g，分段 RSS = %.4g" % (rss_full, rss_u), "",
             "F 统计量 = %.4g" % F,
             "p 值 = %.4g" % pval, "",
             "➤ 判定：" + verdict]
    if pval < 0.05:
        lines.append("➤ 建议：分段建模或用虚拟变量/交互项刻画断点，避免漏掉结构变动。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 5：Granger 因果检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "series1": {"type": "string", "description": "序列 y（因，逗号分隔）；留空用默认演示"},
        "series2": {"type": "string", "description": "序列 x（因，逗号分隔）"},
        "maxlag": {"type": "number", "description": "最大滞后阶数"},
    },
}, category="经济计量")
def granger_causality(series1="", series2="", maxlag="2"):
    """Granger 因果关系检验：检验序列 x 的历史值是否能预测/改善对 y 的预测。
    p<0.05 认为 x 是 y 的 Granger 原因（统计上的先行关系，非必然因果）。"""
    if not _SM:
        return {"text": "未安装 statsmodels，无法做 Granger 检验。"}
    if not str(series1).strip() or not str(series2).strip():
        y_n = 60
        rng = np.random.RandomState(2)
        x = np.cumsum(rng.randn(y_n))
        # y 受 x 的滞后影响 → 存在 Granger 因果
        y = np.zeros(y_n)
        for i in range(1, y_n):
            y[i] = 0.6 * x[i - 1] + rng.randn()
        notch = "（默认演示：y 受 x 滞后影响）"
    else:
        y = _nums(series1)
        x = _nums(series2)
        notch = ""
    if x.size != y.size or x.size < 10:
        raise ValueError("两序列需等长且长度≥10。")
    ml = int(maxlag or 2)
    df = pd.DataFrame({"x": x, "y": y})
    lines = ["Granger 因果检验%s：" % notch, ""]
    try:
        tmp = grangercausalitytests(df[["y", "x"]], maxlag=ml)
    except Exception as ex:
        return {"text": "Granger 检验失败：%r" % ex}
    for lag, result in tmp.items():
        # result：ssr_ftest = (F, p值, 分子df, 分母df)
        ssr = result[0]["ssr_ftest"]
        f_stat = float(ssr[0])
        p_value = float(ssr[1])
        verdict = "x 是 y 的 Granger 原因（p<0.05）" if p_value < 0.05 else "x 不是 y 的 Granger 原因（p≥0.05）"
        lines.append("滞后 %d：F = %.4g，p = %.4g → %s" % (lag, f_stat, p_value, verdict))
    lines.append("")
    lines.append("➤ 含义：Granger 因果是预测意义上的先行关系，不能直接断言真实因果关系。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "经济计量"
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


class EconometricsPage(_FormPage):
    NAME = "经济计量"
    EMOJI = "\U0001F4C9"
    _TITLE = "📉 经济计量建模"
    _TOOLS = {
        "adf_test": (
            "ADF 单位根检验",
            "检验序列是否平稳：p<0.05 平稳，否则非平稳需差分或协整。",
            [("series", "时间序列(逗号分隔，留空=默认随机游走)", ""),
             ("trend", "回归形式 c/ct/n", "c"),
             ("maxlag", "最大滞后(留空自动)", "")],
            adf_test),
        "cointegration_test": (
            "Engle-Granger 协整",
            "检验两个非平稳序列是否有长期稳定线性关系(长期均衡)。",
            [("series1", "序列 y(逗号分隔，留空=默认演示)", ""),
             ("series2", "序列 x(逗号分隔)", ""),
             ("trend", "协整回归形式 c/ct/n", "c")],
            cointegration_test),
        "difference_transform": (
            "差分 / 平稳化",
            "对非平稳序列做 d 阶差分，并再次 ADF 检验平稳性。",
            [("series", "时间序列(逗号分隔，留空=默认随机游走)", ""),
             ("order", "差分阶数 d", "1")],
            difference_transform),
        "chow_test_break": (
            "Chow 结构断点",
            "在某点把样本分段，检验回归系数是否发生结构突变。",
            [("y", "因变量 y(逗号分隔，留空=默认演示)", ""),
             ("x", "自变量 x(逗号分隔)", ""),
             ("break_point", "断点位置(观测序号)", "")],
            chow_test_break),
        "granger_causality": (
            "Granger 因果检验",
            "检验 x 的历史值能否改善对 y 的预测(统计上的先行关系)。",
            [("series1", "序列 y(因，逗号分隔，留空=默认演示)", ""),
             ("series2", "序列 x(因，逗号分隔)", ""),
             ("maxlag", "最大滞后阶数", "2")],
            granger_causality),
    }


PAGES = [EconometricsPage]