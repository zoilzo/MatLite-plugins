# -*- coding: utf-8 -*-
"""金融计算进阶：IRR/NPV、摊还计划、债券定价与到期收益率。AI 工具 + 页面（进阶教学）。"""
import numpy as np
from scipy import optimize
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "finance_pro",
    "name": "金融进阶",
    "version": "1.0",
    "author": "zoilzo",
    "description": "金融计算进阶：IRR 内部收益率、NPV 净现值、按揭摊还计划、债券定价与到期收益率 YTM（工具 + 页面）",
}


# ------------------------------------------------------------
# 纯计算
# ------------------------------------------------------------
def _nums(v):
    """解析现金流序列：接受逗号/空格分隔字符串或列表。"""
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    s = str(v).replace("，", ",").replace(";", ",").replace("\n", ",")
    return [float(x) for x in s.split(",") if x.strip()] if str(v).strip() else []


def _n(v, d=0.0):
    try:
        return float(str(v).replace("，", "").strip())
    except Exception:
        return d


def _npv(rate, flows):
    """NPV：Sigma(flow_t / (1+r)^t)，t 从 0 起。"""
    rate = float(rate)
    tot = 0.0
    for t, cf in enumerate(flows):
        tot += cf / ((1.0 + rate) ** t)
    return tot


def _pc(v):
    return f"{v * 100:.4g}%"


# ------------------------------------------------------------
# 工具 1：NPV 净现值
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "rate": {"type": "number", "description": "折现率（年），如 0.08 表示 8%"},
        "cashflows": {"type": "string", "description": "现金流序列（第0期起），逗号分隔，如 '-1000,300,400,500'"},
    },
}, category="金融进阶")
def npv_calc(rate="0.08", cashflows="-1000,300,400,500"):
    """计算净现值 NPV：把未来各期现金流按折现率 r 折回到第 0 期求和。NPV>0 值得投资。"""
    flows = _nums(cashflows)
    if not flows:
        raise ValueError("请提供现金流序列 cashflows。")
    r = _n(rate, 0.08)
    if r <= -1:
        raise ValueError("折现率必须大于 -1。")
    total = _npv(r, flows)
    txt = "现金流：" + " → ".join(f"{x:g}" for x in flows)
    txt += f"\n折现率 r = {_pc(r)}\n\nNPV = Σ cf_t / (1+r)^t = {total:.4g}"
    txt += "\n\n➤ 判定：" + ("NPV > 0，项目可行" if total > 0 else "NPV ≤ 0，项目不可行")
    return {"text": txt}


# ------------------------------------------------------------
# 工具 2：IRR 内部收益率
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "cashflows": {"type": "string", "description": "现金流序列（第0期起），逗号分隔，如 '-1000,300,400,500'"},
    },
}, category="金融进阶")
def irr_calc(cashflows="-1000,300,420,680"):
    """用数值优化求内部收益率 IRR：使 NPV=0 的折现率。现实中常需在区间内搜索。"""
    flows = _nums(cashflows)
    if not flows:
        raise ValueError("请提供现金流序列 cashflows。")
    if not (flows[0] < 0 or any(x < 0 for x in flows)):
        raise ValueError("至少应有一期为负的现金流（通常是初始投入）。")
    f = lambda r: _npv(r, flows)
    # 在 (-0.9999, big) 内扫描变号区间，多个根取最小的正根
    lo, hi = -0.9999, 10.0
    grid = np.linspace(lo, hi, 4000)
    vals = [f(x) for x in grid]
    root = None
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            root = float(grid[i]); break
        if vals[i] * vals[i + 1] < 0:
            try:
                root = float(optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-12))
            except Exception:
                continue
            break
    if root is None:
        return {"text": "现金流：" + " → ".join(f"{x:g}" for x in flows)
            + "\n\n在所给范围内未找到使 NPV=0 的根。请检查现金流符号是否合理（需含投入与回报）。"}
    txt = "现金流：" + " → ".join(f"{x:g}" for x in flows)
    txt += f"\n\nIRR（内部收益率）≈ {_pc(root)}\n验证：NPV(IRR) = {f(root):.3g}"
    txt += "\n\n➤ 判定：IRR 越高，项目回报越好；IRR > 资金成本/行业基准则可行。"
    return {"text": txt}


# ------------------------------------------------------------
# 工具 3：按揭/贷款摊还计划
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "principal": {"type": "number", "description": "贷款本金"},
        "annual_rate": {"type": "number", "description": "年利率，如 0.05 表示 5%"},
        "years": {"type": "number", "description": "贷款年限"},
    },
}, category="金融进阶")
def amortization_schedule(principal="100000", annual_rate="0.05", years="30"):
    """等额本息按揭摊还计划：每月还款额 = 本金 + 利息的逐期分解表（前 12 期）。"""
    p = _n(principal, 100000)
    r = _n(annual_rate, 0.05)
    years = _n(years, 30)
    if p <= 0:
        raise ValueError("本金必须为正。")
    if r < 0 or years <= 0:
        raise ValueError("年利率/年限必须为正。")
    mr = r / 12.0
    nper = int(round(years * 12))
    if mr == 0:
        pay = p / nper
        table = []
    else:
        pay = p * mr / (1.0 - (1.0 + mr) ** (-nper))
        table = []
    bal = p
    for i in range(1, nper + 1):
        interest = bal * mr
        principal_part = pay - interest if mr != 0 else pay
        if bal < 0:
            principal_part = bal
        bal -= principal_part
        table.append((i, pay, principal_part, interest, bal))
    # 前 12 期明细
    lines = [f"每期还款 = {pay:.3g} 元（等额本息）", f"共 {nper} 期（{years:g} 年，月利率 {_pc(mr)}）", ""]
    lines.append(f"{'期数':>5} {'还款额':>12} {'本金':>12} {'利息':>12} {'剩余本金':>14}")
    for row in table[:12]:
        i, pm, pp, it, bb = row
        lines.append(f"{i:>5} {pm:>12.4g} {pp:>12.4g} {it:>12.4g} {bb:>14.4g}")
    tot_int = sum(row[3] for row in table)
    lines.append("")
    lines.append(f"总还款 = {pay * nper:.3g} 元，总利息 = {tot_int:.3g} 元")
    lines.append("（明细默认展示前 12 期，后续各期以此规律递增本金、递减利息。）")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：债券定价与到期收益率 YTM
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "face": {"type": "number", "description": "债券面值（F）"},
        "coupon_rate": {"type": "number", "description": "票面年利率，如 0.06 表示 6%"},
        "years": {"type": "number", "description": "剩余年限"},
        "yield_rate": {"type": "string", "description": "到期收益率 r（给价时算价）；留空则改用 price 反算 YTM"},
        "price": {"type": "string", "description": "市场价格（给价时反算 YTM）；留空则用 yield_rate 算价"},
        "freq": {"type": "number", "description": "每年付息次数，默认 2（半年付）"},
    },
}, category="金融进阶")
def bond_price_ytm(face="1000", coupon_rate="0.06", years="10", yield_rate="0.05", price="", freq="2"):
    """债券定价与到期收益率：给 yield_rate 算出价格；给 price 则数值解得 YTM。付息频次 freq 可调。"""
    fv = _n(face, 1000)
    cr = _n(coupon_rate, 0.06)
    yrs = _n(years, 10)
    freq = max(1, int(_n(freq, 2)))
    if fv <= 0 or cr < 0 or yrs <= 0:
        raise ValueError("面值/票面利率/年限须为正（票面利率可为 0）。")
    per = int(round(yrs * freq))
    cp = fv * cr / freq  # 每期票息
    times = np.arange(1, per + 1)

    def _price(y):
        y = max(y, -0.999)
        return float(np.sum(cp / (1 + y / freq) ** times) + fv / (1 + y / freq) ** per)

    y = str(yield_rate).strip()
    pr = str(price).strip()
    if y:
        yr = _n(y, 0.05)
        val = _price(yr)
        return {"text": f"面值 F = {fv:g}，票面利率 = {_pc(cr)}，{yrs:g} 年（{freq} 次/年付息）\n"
                f"到期收益率 YTM = {_pc(yr)}\n\n债券价格 = {val:.4g} 元\n\n➤ 判定：价格 > 面值 → 溢价发行（票面利率高于 YTM）；价格 < 面值 → 折价发行"}
    if pr:
        pd = _n(pr, fv)
        # 解 YTM
        f = lambda yy: _price(yy) - pd
        sol = None
        grid = np.linspace(0.0001, 1.5, 8000)
        vals = [f(x) for x in grid]
        for i in range(len(grid) - 1):
            if vals[i] == 0.0:
                sol = float(grid[i]); break
            if vals[i] * vals[i + 1] < 0:
                try:
                    sol = float(optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-12))
                except Exception:
                    continue
                break
        if sol is None:
            return {"text": f"给定价格 {pd:g}，未在范围内找到对应的 YTM，请检查价格与票息是否匹配。"}
        return {"text": f"面值 F = {fv:g}，票面利率 = {_pc(cr)}，{yrs:g} 年\n市场价格 = {pd:g}\n\n到期收益率 YTM ≈ {_pc(sol)}"
                f"\n验证：按 YTM 回算价格 = {_price(sol):.4g}"}
    return {"text": "请提供 yield_rate（算价）或 price（反算 YTM）之一。"}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class _FormPage(ui.BasePage):
    _TITLE = "金融进阶"
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


class FinancePage(_FormPage):
    NAME = "金融进阶"
    EMOJI = "\U0001F4B1"
    _TITLE = "💹 金融计算进阶"
    _TOOLS = {
        "npv_calc": (
            "净现值 NPV",
            "输入各期现金流与折现率，计算净现值并判定项目是否可行。",
            [("rate", "折现率 r（如 0.08=8%）", "0.08"),
             ("cashflows", "现金流（第0期起，逗号分隔）", "-1000,300,400,500")],
            npv_calc),
        "irr_calc": (
            "内部收益率 IRR",
            "数值优化求使 NPV=0 的折现率。现金流需含投入（负）与回报（正）。",
            [("cashflows", "现金流（逗号分隔，首期常为负）", "-1000,300,420,680")],
            irr_calc),
        "amortization_schedule": (
            "按揭摊还计划",
            "等额本息逐期分解：每期还款 = 本金 + 利息，展示前 12 期明细。",
            [("principal", "贷款本金", "100000"),
             ("annual_rate", "年利率（0.05=5%）", "0.05"),
             ("years", "贷款年限", "30")],
            amortization_schedule),
        "bond_price_ytm": (
            "债券定价 / YTM",
            "给到期收益率算债券价格；给市场价格则反算到期收益率 YTM。",
            [("face", "面值 F", "1000"),
             ("coupon_rate", "票面年利率（0.06=6%）", "0.06"),
             ("years", "剩余年限", "10"),
             ("yield_rate", "到期收益率 r（算价用，留空则用 price 反算）", "0.05"),
             ("price", "市场价格（反算 YTM，留空则用 r 算价）", ""),
             ("freq", "每年付息次数", "2")],
            bond_price_ytm),
    }


PAGES = [FinancePage]