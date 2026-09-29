# -*- coding: utf-8 -*-
"""流行病学建模：SIR/SEIR 仓室模型、基本再生数 R0、群体免疫阈值、Logistic 种群增长、Hardy-Weinberg 平衡。
AI 工具 + 页面（交叉领域，教学友好）。"""
import numpy as np
from scipy.integrate import solve_ivp
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "epidemiology_pro",
    "name": "流行病学",
    "version": "1.0",
    "author": "zoilzo",
    "description": "流行病学建模：SIR/SEIR 传染模型、基本再生数 R0、群体免疫阈值、Logistic 种群增长、Hardy-Weinberg 平衡（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _n(v, d=0.0):
    try:
        return float(str(v).replace("，", "").strip())
    except Exception:
        return d


def _nums(v):
    """解析个体/数量序列：接受逗号/空格分隔字符串或列表。"""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    s = str(v).replace("，", ",").replace(";", ",").replace("\n", ",")
    return [float(x) for x in s.split(",") if x.strip()] if str(v).strip() else []


def _sir_rhs(t, y, beta, gamma, N):
    """SIR 仓室模型右侧：return d[S,I,R]/dt。N=S+I+R 保持常数。"""
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def _seir_rhs(t, y, beta, sigma, gamma, N):
    """SEIR 仓室模型右侧：加入潜伏期 E。"""
    S, E, I, R = y
    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I
    return [dS, dE, dI, dR]


# ------------------------------------------------------------
# 工具 1：SIR 模型模拟
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "population": {"type": "number", "description": "总人口 N"},
        "beta": {"type": "number", "description": "传染率 β（每个易感者每天接触并感染的期望）"},
        "gamma": {"type": "number", "description": "恢复率 γ（1/γ 为平均传染期天数）"},
        "exposed": {"type": "number", "description": "初始感染者 I0"},
        "days": {"type": "number", "description": "模拟天数"},
    },
}, category="流行病学")
def sir_curve(population="10000", beta="0.40", gamma="0.10", exposed="10", days="120"):
    """SIR 传染病模型：S(易感)-I(感染)-R(康复) 仓室。输出 I 峰值、达峰时间、最终流行规模与 R0。"""
    N = _n(population, 10000)
    beta = _n(beta, 0.40)
    gamma = _n(gamma, 0.10)
    I0 = _n(exposed, 10)
    T = _n(days, 120)
    if N <= 0 or beta <= 0 or gamma <= 0:
        raise ValueError("N / beta / gamma 必须为正。")
    if I0 <= 0 or I0 >= N:
        raise ValueError("初始感染者 I0 应大于 0 且小于总人数 N。")
    S0 = N - I0
    y0 = [S0, I0, 0.0]
    sol = solve_ivp(_sir_rhs, [0, T], y0, args=(beta, gamma, N), dense_output=True, max_step=1.0)
    t = sol.t
    S, I, R = sol.y
    # 峰值
    peak_i = float(np.max(I))
    peak_t = float(t[int(np.argmax(I))])
    # 最终流行规模（取期末，够收敛）
    final_I = float(I[-1])
    final_S = float(S[-1])
    recovered = N - final_S
    r0 = beta / gamma
    # 抽样 6 个时间点
    idxs = np.unique(np.linspace(0, len(t) - 1, 6).astype(int))
    lines = [f"SIR 模型：N = {N:g}，β = {beta:g}/天，γ = {gamma:g}/天，初始 I0 = {I0:g}",
             f"基本再生数 R0 = β/γ = {beta:g}/{gamma:g} = {r0:.3g}", ""]
    lines.append(f"{'天数':>6} {'易感S':>12} {'感染I':>12} {'康复R':>12}")
    for i in idxs:
        lines.append(f"{t[i]:>6.0f} {S[i]:>12.0f} {I[i]:>12.0f} {R[i]:>12.0f}")
    lines.append("")
    lines.append(f"感染峰值 = {peak_i:.0f} 人（第 {peak_t:.0f} 天）")
    lines.append(f"期末仍在传染 = {final_I:.0f} 人，累计康复 = {recovered:.0f} 人")
    lines.append(f"总流行规模 ≈ {recovered:.0f} / {N:g}（{recovered / N * 100:.3g}%）")
    lines.append("")
    lines.append("➤ 含义：R0 > 1 疫情会爆发并自限；R0 < 1 则疫情自行消退。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：SEIR 模型模拟
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "population": {"type": "number", "description": "总人口 N"},
        "beta": {"type": "number", "description": "传染率 β"},
        "sigma": {"type": "number", "description": "潜伏转感染率 σ（1/σ 为平均潜伏期天数）"},
        "gamma": {"type": "number", "description": "恢复率 γ"},
        "exposed": {"type": "number", "description": "初始感染者 I0"},
        "incubated": {"type": "number", "description": "初始潜伏者 E0"},
        "days": {"type": "number", "description": "模拟天数"},
    },
}, category="流行病学")
def seir_curve(population="10000", beta="0.40", sigma="0.20", gamma="0.10", exposed="5", incubated="5", days="150"):
    """SEIR 传染病模型：在 SIR 基础上加入潜伏仓室 E（感染后先潜伏后发病）。输出 I 峰值与流行规模。"""
    N = _n(population, 10000)
    beta = _n(beta, 0.40)
    sigma = _n(sigma, 0.15)
    gamma = _n(gamma, 0.10)
    I0 = _n(exposed, 5)
    E0 = _n(incubated, 5)
    T = _n(days, 150)
    if N <= 0 or beta <= 0 or sigma <= 0 or gamma <= 0:
        raise ValueError("N / beta / sigma / gamma 必须为正。")
    if (I0 + E0) <= 0 or (I0 + E0) >= N:
        raise ValueError("初始潜伏+感染者应大于 0 且小于总人数 N。")
    S0 = N - I0 - E0
    y0 = [S0, E0, I0, 0.0]
    sol = solve_ivp(_seir_rhs, [0, T], y0, args=(beta, sigma, gamma, N), dense_output=True, max_step=1.0)
    t = sol.t
    S, E, I, R = sol.y
    peak_i = float(np.max(I))
    peak_t = float(t[int(np.argmax(I))])
    recovered = N - float(S[-1])
    r0 = beta / gamma
    idxs = np.unique(np.linspace(0, len(t) - 1, 6).astype(int))
    lines = [f"SEIR 模型：N = {N:g}，β = {beta:g}/天，1/σ = {1 / sigma:.3g} 天潜伏，1/γ = {1 / gamma:.3g} 天传染",
             f"基本再生数 R0 = β/γ = {r0:.3g}", ""]
    lines.append(f"{'天数':>6} {'易感S':>12} {'潜伏E':>12} {'感染I':>12} {'康复R':>12}")
    for i in idxs:
        lines.append(f"{t[i]:>6.0f} {S[i]:>12.0f} {E[i]:>12.0f} {I[i]:>12.0f} {R[i]:>12.0f}")
    lines.append("")
    lines.append(f"感染峰值 = {peak_i:.0f} 人（第 {peak_t:.0f} 天）")
    lines.append(f"累计被感染（潜伏+感染→康复）≈ {recovered:.0f} / {N:g}（{recovered / N * 100:.3g}%）")
    lines.append("")
    lines.append("➤ 含义：潜伏期（1/σ）越长，疫情起步越慢；R0 > 1 的传染病需干预才会收敛。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：基本再生数 / 群体免疫
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "beta": {"type": "number", "description": "传染率 β（也可留空，用生长率 r + 传染期反算）"},
        "gamma": {"type": "number", "description": "恢复率 γ（有传染期时可直接算 R0）"},
        "growth_rate": {"type": "string", "description": "疫情指数增长率 r（1/天）；留空则用 β/γ 直接算"},
        "infectious_period": {"type": "string", "description": "平均传染期（天）；留空则用 1/γ"},
    },
}, category="流行病学")
def r0_herd_immunity(beta="0.4", gamma="0.1", growth_rate="0.15", infectious_period=""):
    """基本再生数 R0：疫情早期一名感染者平均能传染的人数。R0>1 疫情扩散，R0<1 消退。
    再由 R0 求群体免疫阈值 h=1-1/R0 与所需疫苗接种率。"""
    r = _n(growth_rate, 0.0)
    period = _n(infectious_period, 0.0)
    b = _n(beta, 0.0)
    g = _n(gamma, 0.0)
    method = ""
    if period > 0:
        # R0 = 1 + r * 传染期
        r0 = 1.0 + r * period if r > 0 else None
        method = "R0 = 1 + r×D = 1 + %g×%g" % (r, period)
    elif b > 0 and g > 0:
        r0 = b / g
        method = "R0 = β/γ = %g/%g" % (b, g)
    else:
        return {"text": "请提供两种途径之一：① beta + gamma（R0=β/γ）；② growth_rate + infectious_period（R0=1+r·D）。"}
    if r0 is None:
        return {"text": "生长率 r 需为正才能用「R0 = 1 + r·D」估算。"}
    herd = 1.0 - 1.0 / r0 if r0 > 1 else None
    lines = [f"基本再生数 R0 = {r0:.3g}   〔{method}〕", ""]
    if r0 > 1:
        lines.append(f"群体免疫阈值 h = 1 - 1/R0 = 1 - 1/{r0:.3g} = {herd * 100:.3g}%"
                     f"（约需 {herd * 100:.3g}% 人群获得免疫，疫情自然下降）")
        lines.append(f"所需基础疫苗覆盖率 ≈ {herd * 100:.3g}%（若疫苗有效率 100%）")
    elif r0 < 1:
        lines.append(f"R0 < 1：疫情无法维持传播，会自行消退（无需强制群体免疫）。")
    else:
        lines.append("R0 = 1：疫情界值，需要轻微干预即可控制。")
    lines.append("")
    lines.append("➤ 流行病学意义：R0 是衡量传染病传播力的核心指标；麻疹≈12-18，新冠原始株≈2.5-3，流感≈1.3-1.8。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：Logistic 种群增长
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "carrying": {"type": "number", "description": "环境容纳量 K"},
        "growth_rate": {"type": "number", "description": "固有增长率 r"},
        "initial": {"type": "number", "description": "初始种群数 P0"},
        "time": {"type": "number", "description": "考察时间点 t"},
    },
}, category="流行病学")
def logistic_growth(carrying="1000", growth_rate="0.30", initial="50", time="20"):
    """Logistic 种群增长 P(t)=K/(1+((K-P0)/P0)·e^(-r·t))：受资源限制的 S 形增长。
    输出 t 时刻规模、增速最快时间及接近 K 的程度。"""
    K = _n(carrying, 1000)
    r = _n(growth_rate, 0.3)
    P0 = _n(initial, 50)
    t = _n(time, 2)
    if K <= 0 or r <= 0 or P0 <= 0:
        raise ValueError("K / r / P0 必须为正。")
    if P0 >= K:
        raise ValueError("初始种群 P0 应小于环境容纳量 K。")
    P = K / (1.0 + ((K - P0) / P0) * np.exp(-r * t))
    # 增速最快点（拐点）= K/2，对应时间
    t_infl = np.log((K - P0) / P0) / r
    pct = P / K * 100
    lines = [f"Logistic 增长：K = {K:g}，r = {r:g}/单位时间，P0 = {P0:g}", "",
             f"P({t:g}) = K / (1 + ((K-P0)/P0)·e^(-r·t)) = {P:.3g}", ""]
    lines.append(f"当前占环境容量 = {pct:.3g}%")
    lines.append(f"增速峰值（拐点）出现在 t = {t_infl:.3g}，届时种群 = K/2 = {K / 2:.3g}")
    lines.append(f"当 t = {t_infl:.3g} 时接近 K 的程度：取决于 t 相对 t_infl 的位置，t 愈大愈接近 K。")
    lines.append("")
    lines.append("➤ 生物学意义：初期按 e^(r·t) 指数增长，接近 K 时增速放缓，最终趋向环境容纳量 K。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 5：Hardy-Weinberg 平衡检验
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "counts": {"type": "string", "description": "三种基因型个体数，逗号分隔：AA,Aa,aa"},
    },
}, category="流行病学")
def hardy_weinberg(counts="490,420,90"):
    """Hardy-Weinberg 平衡：根等位基因频率，反推基因型期望频率，并用卡方拟合优度检验群体是否处于平衡。"""
    c = _nums(counts)
    if len(c) != 3:
        raise ValueError("请输入 3 个基因型计数：AA,Aa,aa。")
    AA, Aa, aa = c
    if AA < 0 or Aa < 0 or aa < 0:
        raise ValueError("基因型计数不能为负。")
    N = AA + Aa + aa
    if N <= 0:
        raise ValueError("总个体数必须为正。")
    # 等位基因频率 p = (2*AA + Aa)/(2N), q = 1-p
    p = (2 * AA + Aa) / (2 * N)
    q = (2 * aa + Aa) / (2 * N)
    # 期望基因型频率 p^2, 2pq, q^2
    eAA = p * p * N
    eAa = 2 * p * q * N
    eaa = q * q * N
    # 卡方（自由度 = 类别3 - 估计1 = 2）
    chi2 = 0.0
    for o, e in [(AA, eAA), (Aa, eAa), (aa, eaa)]:
        if e > 0:
            chi2 += (o - e) ** 2 / e
    lines = [f"Hardy-Weinberg：AA = {AA:g}，Aa = {Aa:g}，aa = {aa:g}，总 N = {N:g}", "",
             f"等位基因频率：p(A) = {p:.4g}，q(a) = {q:.4g}（p + q = {p + q:.4g}）", "",
             f"{'基因型':>6} {'观察':>8} {'期望':>8} {'频率':>8}"]
    obs = [AA, Aa, aa]
    exp = [eAA, eAa, eaa]
    freq = [p * p, 2 * p * q, q * q]
    for lbl, o, e, f in zip(["AA", "Aa", "aa"], obs, exp, freq):
        lines.append(f"{lbl:>6} {o:>8.0f} {e:>8.1f} {f:>8.4f}")
    lines.append("")
    lines.append(f"卡方拟合优度 χ² = {chi2:.4g}（df=2，α=0.05 临界值 5.991）")
    if chi2 < 5.991:
        lines.append("➤ 判定：χ² < 5.991，群体处于 Hardy-Weinberg 平衡（差异不显著）。")
    else:
        lines.append("➤ 判定：χ² ≥ 5.991，偏离 Hardy-Weinberg 平衡（可能存在选择/近交/迁移等）。")
    lines.append("")
    lines.append("➤ 生物学意义：HWE 是无选择、无突变、无迁移、随机交配的理想大群体的基因型平衡。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "流行病学"
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


class EpidemiologyPage(_FormPage):
    NAME = "流行病学"
    EMOJI = "\U0001F30D"
    _TITLE = "🦠 流行病学建模"
    _TOOLS = {
        "sir_curve": (
            "SIR 传染模型",
            "S(易感)-I(感染)-R(康复) 仓室模型，模拟疫情过程（含潜伏的不在此）。",
            [("population", "总人口 N", "10000"),
             ("beta", "传染率 β", "0.40"),
             ("gamma", "恢复率 γ", "0.10"),
             ("exposed", "初始感染者 I0", "10"),
             ("days", "模拟天数", "120")],
            sir_curve),
        "seir_curve": (
            "SEIR 传染模型",
            "加潜伏仓室 E：先感染后潜伏再发病，比 SIR 更接近潜伏性传染病。",
            [("population", "总人口 N", "10000"),
             ("beta", "传染率 β", "0.40"),
             ("sigma", "潜伏转感染率 σ", "0.20"),
             ("gamma", "恢复率 γ", "0.10"),
             ("exposed", "初始感染者 I0", "5"),
             ("incubated", "初始潜伏者 E0", "5"),
             ("days", "模拟天数", "150")],
            seir_curve),
        "r0_herd_immunity": (
            "R0 / 群体免疫",
            "算基本再生数 R0，并由此求群体免疫阈值与所需疫苗覆盖率。",
            [("beta", "传染率 β（可与 gamma 组合）", "0.40"),
             ("gamma", "恢复率 γ", "0.10"),
             ("growth_rate", "疫情指数增长率 r（/天）", "0.15"),
             ("infectious_period", "平均传染期（天）", "")],
            r0_herd_immunity),
        "logistic_growth": (
            "Logistic 种群增长",
            "受环境容量限制的 S 形种群增长：求 t 时刻规模和增速最快点。",
            [("carrying", "环境容纳量 K", "1000"),
             ("growth_rate", "固有增长率 r", "0.30"),
             ("initial", "初始种群 P0", "50"),
             ("time", "考察时间点 t", "20")],
            logistic_growth),
        "hardy_weinberg": (
            "Hardy-Weinberg 平衡",
            "据基因型计数求等位基因频率、期望基因型频率，并做卡方检验群体是否平衡。",
            [("counts", "基因型计数 AA,Aa,aa", "490,420,90")],
            hardy_weinberg),
    }


PAGES = [EpidemiologyPage]