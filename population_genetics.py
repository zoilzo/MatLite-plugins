# -*- coding: utf-8 -*-
"""群体遗传学：Hardy-Weinberg 平衡检验、等位基因频率、近交系数、基因型分布、遗传漂变(Wright-Fisher)。
AI 工具 + 页面（交叉领域：把概率论/统计用到遗传学）。依赖 numpy/scipy。"""
import numpy as np
from scipy import stats as st
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "population_genetics",
    "name": "群体遗传学",
    "version": "1.0",
    "author": "zoilzo",
    "description": "群体遗传学：Hardy-Weinberg 平衡卡方检验、等位基因/基因型频率、近交系数、基因型分布、Wright-Fisher 遗传漂变（工具 + 页面）",
}


# ------------------------------------------------------------
# 通用辅助（纯计算，无 GUI）
# ------------------------------------------------------------
def _num(v, default):
    """稳健数值解析：容忍空串、中文逗号、空格。解析失败回退默认值。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("，", ",").replace(" ", "").replace("%", "")
    if not s or s in ("-", "无", "空"):
        return default
    try:
        return float(s)
    except Exception:
        return default


def _counts(genotypes):
    """解析基因型计数 AA,Aa,aa：接受 'AA,Aa,aa' 或 '50,60,30'；留空用默认演示(接近 HWE)。"""
    s = str(genotypes or "").strip()
    if not s:
        return 89, 110, 31  # 演示：N=230, p≈0.63, 接近 HWE
    s = s.replace("，", ",").replace(";", ",").replace("\t", ",").replace(" ", ",")
    parts = [p for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("请提供 3 个计数：AA, Aa, aa（逗号分隔）。")
    return int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))


def _freqs(AA, Aa, aa):
    """由基因型计数算等位基因频率 p(显性A)、q(隐性a) 与样本量 N。"""
    N = AA + Aa + aa
    if N <= 0:
        raise ValueError("样本量为 0。")
    p = (2 * AA + Aa) / (2 * N)
    q = 1.0 - p
    return p, q, N


# ------------------------------------------------------------
# 工具 1：Hardy-Weinberg 平衡检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "genotypes": {"type": "string", "description": "基因型计数 AA,Aa,aa（逗号分隔）；留空用默认演示"},
    },
}, category="群体遗传学")
def hardy_weinberg_test(genotypes=""):
    """Hardy-Weinberg 平衡卡方拟合优度检验：由基因型计数估计等位基因频率，与期望基因型分布对比。
    p<0.05 偏离 HWE（可能存在选择、近交、群体分层或基因分型错误）；p≥0.05 认为接近平衡。"""
    AA, Aa, aa = _counts(genotypes)
    p, q, N = _freqs(AA, Aa, aa)
    exp = np.array([N * p * p, 2 * N * p * q, N * q * q])
    obs = np.array([AA, Aa, aa], dtype=float)
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-9)))
    pval = float(st.chi2.sf(chi2, 1))  # HWE 自由度 = 3 - 1(总数) - 1(估计 p) = 1
    verdict = "偏离 Hardy-Weinberg 平衡（p<0.05）" if pval < 0.05 else "接近 Hardy-Weinberg 平衡（p≥0.05）"
    lines = ["Hardy-Weinberg 平衡检验：", "",
             "观测基因型：AA = %d，Aa = %d，aa = %d（N = %d）" % (AA, Aa, aa, N),
             "等位基因频率：p(A) = %.4f，q(a) = %.4f" % (p, q), "",
             "期望基因型：AA = %.1f，Aa = %.1f，aa = %.1f" % (exp[0], exp[1], exp[2]), "",
             "卡方 = %.4f，p = %.4f（自由度 1）" % (chi2, pval), "",
             "➤ 判定：" + verdict]
    if pval < 0.05:
        lines.append("➤ 建议：排查群体分层/近交/选择压力或基因型判定错误。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：等位基因频率
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "genotypes": {"type": "string", "description": "基因型计数 AA,Aa,aa（逗号分隔）；留空用默认演示"},
    },
}, category="群体遗传学")
def allele_frequency(genotypes=""):
    """等位基因频率与基因型分布：由基因型计数计算 p/q，并给出 Hardy-Weinberg 期望比例与杂合度。
    用于遗传结构描述、育种与群体分层分析。"""
    AA, Aa, aa = _counts(genotypes)
    p, q, N = _freqs(AA, Aa, aa)
    obs_het = Aa / N
    exp_het = 2 * p * q
    lines = ["等位基因频率：", "",
             "基因型：AA = %d，Aa = %d，aa = %d（N = %d）" % (AA, Aa, aa, N), "",
             "p(A) = %.4f，q(a) = %.4f" % (p, q), "",
             "基因型期望比例：AA = p² = %.4f，Aa = 2pq = %.4f，aa = q² = %.4f" % (p * p, 2 * p * q, q * q), "",
             "观测杂合度 = %.4f，期望杂合度(2pq) = %.4f" % (obs_het, exp_het)]
    effAA = 2 * N * p * q  # 有效杂合子数估值
    lines.append("➤ 随机交配且无选择时，杂合子比例约为 2pq；若观测偏低，可能存在近交/分层。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：近交系数
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "genotypes": {"type": "string", "description": "基因型计数 AA,Aa,aa（逗号分隔）；留空用默认演示"},
    },
}, category="群体遗传学")
def inbreeding_coefficient(genotypes=""):
    """近交系数 F = 1 - 观测杂合度 / 期望杂合度(2pq)：F>0 表示杂合子缺失(近交/分层)；F<0 表示杂合子过剩。
    常用于遗传学与植物育种评估群体受近交/选择影响的程度。"""
    AA, Aa, aa = _counts(genotypes)
    p, q, N = _freqs(AA, Aa, aa)
    obs_het = Aa / N
    exp_het = 2 * p * q
    F = 1 - (obs_het / exp_het) if exp_het > 0 else 0.0
    if abs(F) < 0.03:
        verdict = "接近随机交配（F≈0）"
    elif F > 0:
        verdict = "杂合子缺失 → 存在近交/群体分层（F>0）"
    else:
        verdict = "杂合子过剩 → 可能存在杂合优势/负选择（F<0）"
    lines = ["近交系数 F：", "",
             "N = %d，p = %.4f，q = %.4f" % (N, p, q),
             "观测杂合度 = %.4f，期望杂合度 = %.4f" % (obs_het, exp_het), "",
             "F = %.4f" % F, "",
             "➤ 判定：" + verdict]
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：基因型分布
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "p": {"type": "number", "description": "等位基因 A 的频率（0~1，默认 0.5）"},
        "n": {"type": "number", "description": "样本个体数（默认 200）"},
    },
}, category="群体遗传学")
def genotype_distribution(p=0.5, n=200):
    """Hardy-Weinberg 基因型分布：给定等位基因频率 p 与样本量，给出期望的 AA/Aa/aa 基因型个体数。
    是群体遗传学中假设随机交配时的基准分布。"""
    pv = _num(p, 0.5)
    pv = min(max(pv, 0.0001), 0.9999)
    N = int(_num(n, 200))
    N = max(N, 10)
    q = 1.0 - pv
    aa = N * q * q
    a_ = N * 2 * pv * q
    AA = N * pv * pv
    lines = ["Hardy-Weinberg 基因型分布：", "",
             "p(A) = %.4f，q(a) = %.4f，样本量 N = %d" % (pv, q, N), "",
             "期望个体数：AA = %.1f，Aa = %.1f，aa = %.1f" % (AA, a_, aa), "",
             "期望比例：AA = %.4f，Aa = %.4f，aa = %.4f" % (pv * pv, 2 * pv * q, q * q)]
    het = 2 * pv * q
    lines.append("➤ 随机交配时杂合子占比 2pq 最大（p=q=0.5 时达 0.5）；偏离则提示非随机交配。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 5：Wright-Fisher 遗传漂变
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "p": {"type": "number", "description": "初始等位基因 A 频率（0~1，默认 0.5）"},
        "population": {"type": "number", "description": "群体个体数 N（默认 100）"},
        "generations": {"type": "number", "description": "模拟代数（默认 40）"},
        "trials": {"type": "number", "description": "独立实验次数（默认 3）"},
    },
}, category="群体遗传学")
def genetic_drift(p=0.5, population=100, generations=40, trials=3):
    """Wright-Fisher 遗传漂变模拟：理想群体(无选择/无突变)中，等位基因频率随世代随机漂移直至固定或消失。
    演示群体大小对多态性维持的影响：N 越小漂变越快。"""
    p0 = _num(p, 0.5)
    p0 = min(max(p0, 0.001), 0.999)
    N = int(_num(population, 100))
    N = max(N, 10)
    G = int(_num(generations, 40))
    G = max(G, 1)
    T = int(_num(trials, 3))
    T = max(T, 1)
    rng = np.random.RandomState(2025)
    finals = []
    lost = 0
    fixed = 0
    lines = ["Wright-Fisher 遗传漂变模拟：", "",
             "初始 p = %.3f，群体个体数 N = %d，代数 = %d，独立实验 = %d" % (p0, N, G, T), ""]
    for t in range(T):
        p = p0
        traj = [p]
        for _ in range(G):
            cnt = rng.binomial(2 * N, p)
            p = cnt / (2.0 * N)
            traj.append(p)
        last = traj[-1]
        finals.append(last)
        if last >= 0.999:
            fixed += 1
        elif last <= 0.001:
            lost += 1
        lines.append("实验 %d：末代 p = %.3f" % (t + 1, last))
    arr = np.array(finals)
    lines.append("")
    lines.append("末代频率：平均 = %.3f，标准差 = %.3f" % (arr.mean(), arr.std()))
    lines.append("固定 %d 次，消失 %d 次，仍多态 %d 次（共 %d 次实验）" % (fixed, lost, T - fixed - lost, T))
    lines.append("➤ 群体越小(N 小)、代数越多，漂变越易使等位基因固定或消失；大群体多态性维持更久。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class GenoFormPage(ui.BasePage):
    _TITLE = "群体遗传学"
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
            text = "计算出错：%s" % ex
        out.configure(state="normal")
        out.insert("1.0", text)
        out.configure(state="disabled")


class PopulationGeneticsPage(GenoFormPage):
    NAME = "群体遗传学"
    EMOJI = "🧬"
    _TITLE = "🧬 群体遗传学"
    _TOOLS = {
        "hardy_weinberg_test": (
            "H-W 平衡检验",
            "由基因型计数做卡方检验，判断是否偏离 Hardy-Weinberg 平衡。",
            [("genotypes", "AA,Aa,aa(逗号分隔，留空=默认演示)", "")],
            hardy_weinberg_test),
        "allele_frequency": (
            "等位基因频率",
            "计算 p/q 与基因型期望比例、杂合度，用于遗传结构描述。",
            [("genotypes", "AA,Aa,aa(逗号分隔)", "")],
            allele_frequency),
        "inbreeding_coefficient": (
            "近交系数 F",
            "F = 1 - 观测杂合度/期望杂合度；反映杂合子缺失或过剩。",
            [("genotypes", "AA,Aa,aa(逗号分隔)", "")],
            inbreeding_coefficient),
        "genotype_distribution": (
            "基因型分布",
            "给定 p 与样本量，输出 Hardy-Weinberg 期望的基因型个体数。",
            [("p", "等位基因 A 频率(0~1)", "0.5"),
             ("n", "样本个体数", "200")],
            genotype_distribution),
        "genetic_drift": (
            "遗传漂变模拟",
            "Wright-Fisher 模型模拟频率随机漂移，演示群体大小的影响。",
            [("p", "初始频率 p", "0.5"),
             ("population", "个体数 N", "100"),
             ("generations", "代数", "40"),
             ("trials", "实验次数", "3")],
            genetic_drift),
    }


PAGES = [PopulationGeneticsPage]