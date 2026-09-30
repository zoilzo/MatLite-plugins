# -*- coding: utf-8 -*-
"""金融工程与衍生品风险：Black-Scholes 期权定价、希腊字母、隐含波动率、债券久期/凸性、蒙特卡洛 VaR。AI 工具 + 页面。"""
import customtkinter as ctk
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "options_risk",
    "name": "金融工程与风险",
    "version": "1.0",
    "author": "zoilzo",
    "description": "Black-Scholes 期权定价/希腊字母、隐含波动率、债券久期凸性、蒙特卡洛 VaR（工具 + 页面）",
}


def _f(v, d):
    """稳健数值：容忍 None/字符串，失败用默认值。"""
    if v is None:
        return float(d)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("，", "").replace("%", "").strip()
    if s in ("", "-", "无", "空", "None"):
        return float(d)
    try:
        return float(s)
    except Exception:
        return float(d)


def _n(v, d):
    """整数解析：保证非负。"""
    return max(0, int(_f(v, d)))


def _fmt(x):
    """数值显示。"""
    x = float(x)
    if abs(x) >= 1e6 or (abs(x) < 1e-4 and x != 0):
        return f"{x:.4g}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _norm(x):
    """标准正态分布累积函数。"""
    from scipy.stats import norm
    return norm.cdf(x)


def _is_call(typ):
    return str(typ).strip().lower() in ("call", "c", "买", "看涨")


@ai_tools._reg
@ai_tools._tool({"properties": {"S": {"type": "number"}, "K": {"type": "number"}, "T": {"type": "number"}, "r": {"type": "number"}, "sigma": {"type": "number"}, "type": {"type": "string"}}, "required": []}, category="金融工程")
def black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, type="call"):
    """Black-Scholes 期权定价：输入标的价 S、行权价 K、期限 T(年)、无风险利率 r、波动率 sigma，返回期权价格与 Delta。"""
    S_, K_, T_, r_, sig_ = _f(S, 100), _f(K, 100), _f(T, 1), _f(r, 0.05), _f(sigma, 0.2)
    if T_ <= 0 or sig_ <= 0:
        return {"text": "期限 T 与波动率 sigma 必须为正数。"}
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ * sig_) * T_) / (sig_ * np.sqrt(T_))
    d2 = d1 - sig_ * np.sqrt(T_)
    nd = _norm
    if _is_call(type):
        price = S_ * nd(d1) - K_ * np.exp(-r_ * T_) * nd(d2)
        delta = nd(d1)
        cn = "看涨期权(call)"
    else:
        price = K_ * np.exp(-r_ * T_) * nd(-d2) - S_ * nd(-d1)
        delta = nd(d1) - 1
        cn = "看跌期权(put)"
    return {"text": ("%s 理论价格 = %.4f\nDelta = %.4f\n(D1=%.4f, D2=%.4f, S=%.2f, K=%.2f, T=%.2f, r=%.4f, σ=%.4f)" % (cn, price, delta, d1, d2, S_, K_, T_, r_, sig_))}


@ai_tools._reg
@ai_tools._tool({"properties": {"S": {"type": "number"}, "K": {"type": "number"}, "T": {"type": "number"}, "r": {"type": "number"}, "sigma": {"type": "number"}, "type": {"type": "string"}}, "required": []}, category="金融工程")
def option_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2, type="call"):
    """期权希腊字母：Delta/Gamma/Theta/Vega/Rho，用于衡量期权价格对各因素的敏感度。"""
    S_, K_, T_, r_, sig_ = _f(S, 100), _f(K, 100), _f(T, 1), _f(r, 0.05), _f(sigma, 0.2)
    if T_ <= 0 or sig_ <= 0:
        return {"text": "期限 T 与波动率 sigma 必须为正数。"}
    d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig_ * sig_) * T_) / (sig_ * np.sqrt(T_))
    d2 = d1 - sig_ * np.sqrt(T_)
    pdf = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    nd = _norm
    if _is_call(type):
        delta = nd(d1)
        theta = -(S_ * pdf * sig_) / (2 * np.sqrt(T_)) - r_ * K_ * np.exp(-r_ * T_) * nd(d2)
        rho = K_ * T_ * np.exp(-r_ * T_) * nd(d2)
    else:
        delta = nd(d1) - 1
        theta = -(S_ * pdf * sig_) / (2 * np.sqrt(T_)) + r_ * K_ * np.exp(-r_ * T_) * nd(-d2)
        rho = -K_ * T_ * np.exp(-r_ * T_) * nd(-d2)
    gamma = pdf / (S_ * sig_ * np.sqrt(T_))
    vega = S_ * pdf * np.sqrt(T_) / 100.0
    return {"text": "Delta=%.4f  Gamma=%.4f  Theta=%.4f  Vega=%.4f  Rho=%.4f" % (delta, gamma, theta, vega, rho)}


@ai_tools._reg
@ai_tools._tool({"properties": {"S": {"type": "number"}, "K": {"type": "number"}, "T": {"type": "number"}, "r": {"type": "number"}, "price": {"type": "number"}, "type": {"type": "string"}}, "required": []}, category="金融工程")
def implied_vol(S=100, K=100, T=1, r=0.05, price=10, type="call"):
    """隐含波动率：给定期权市价，反解使 Black-Scholes 定价等于市价的波动率 sigma。"""
    S_, K_, T_, r_, px = _f(S, 100), _f(K, 100), _f(T, 1), _f(r, 0.05), _f(price, 10)
    if T_ <= 0 or px <= 0:
        return {"text": "期限 T 与期权价格必须为正数。"}

    def bs(sig):
        d1 = (np.log(S_ / K_) + (r_ + 0.5 * sig * sig) * T_) / (sig * np.sqrt(T_))
        d2 = d1 - sig * np.sqrt(T_)
        if _is_call(type):
            return S_ * _norm(d1) - K_ * np.exp(-r_ * T_) * _norm(d2)
        return K_ * np.exp(-r_ * T_) * _norm(-d2) - S_ * _norm(-d1)

    from scipy.optimize import brentq
    try:
        sig = brentq(lambda s: bs(s) - px, 1e-6, 5.0)
    except Exception:
        return {"text": "未能在 (0, 5] 内找到隐含波动率（请检查期权价格是否合理）。"}
    return {"text": "隐含波动率 σ = %.4f（%.2f%%）" % (sig, sig * 100)}


@ai_tools._reg
@ai_tools._tool({"properties": {"face": {"type": "number"}, "coupon": {"type": "number"}, "rate": {"type": "number"}, "years": {"type": "number"}, "freq": {"type": "number"}}, "required": []}, category="金融工程")
def bond_duration_convexity(face=1000, coupon=0.05, rate=0.04, years=10, freq=2):
    """债券久期与凸性：输入面值、票息率、到期收益率、年限、年付息次数，返回麦考利/修正久期与凸性。"""
    face_, c_, r_, y_, f_ = _f(face, 1000), _f(coupon, 0.05), _f(rate, 0.04), _f(years, 10), _f(freq, 2)
    f_ = max(1, int(f_))
    n = int(y_ * f_)
    if n <= 0:
        return {"text": "期限必须为正数。"}
    cf = face_ * c_ / f_
    t = np.arange(1, n + 1)
    cash = np.full(n, cf)
    cash[-1] += face_
    yp = r_ / f_
    pv = cash / (1 + yp) ** t
    price = pv.sum()
    if price <= 0:
        return {"text": "债券定价异常，请检查参数。"}
    macaulay = float((t * pv).sum() / price)  # 期数
    macaulay_year = macaulay / f_
    mod_year = macaulay / ((1 + yp) * f_)
    conv = float((t * (t + 1) * pv).sum() / price) / ((1 + yp) ** 2) / (f_ * f_)
    return {"text": "麦考利久期=%.3f 年，修正久期=%.3f，凸性=%.3f（价=%.4f）" % (macaulay_year, mod_year, conv, price)}


@ai_tools._reg
@ai_tools._tool({"properties": {"portfolio": {"type": "number"}, "mu": {"type": "number"}, "sigma": {"type": "number"}, "days": {"type": "number"}, "alpha": {"type": "number"}, "seed": {"type": "number"}}, "required": []}, category="金融工程")
def monte_carlo_var(portfolio=100000, mu=0.0002, sigma=0.02, days=10, alpha=0.95, seed=7):
    """蒙特卡洛风险价值 VaR：给定组合价值、日收益率均值/波动、持有天数，模拟路径，输出 VaR 与期望缺口(CVaR)。"""
    P_, mu_, sig_, D_, a_ = _f(portfolio, 100000), _f(mu, 0.0002), _f(sigma, 0.02), _n(days, 10), _f(alpha, 0.95)
    rng = np.random.default_rng(int(_n(seed, 7)))
    nsim = 20000
    rets = rng.normal(mu_, sig_, (nsim, D_))
    final = P_ * np.exp(rets.sum(axis=1))
    pnl = final - P_
    var = float(np.percentile(pnl, (1 - a_) * 100))
    tail = pnl[pnl <= var]
    cvar = float(tail.mean()) if len(tail) else var
    var_loss = abs(var)
    cvar_loss = abs(cvar)
    return {"text": ("蒙特卡洛 VaR（%.0f%% 置信, %d 天, %d 条路径）\nVaR = %.2f 元（组合价值的 %.2f%%）\n期望缺口 CVaR = %.2f 元" % (a_ * 100, D_, nsim, var_loss, var_loss / P_ * 100, cvar_loss))}


class OptionsRiskPage(ui.BasePage):
    NAME = "金融工程与风险"
    EMOJI = "\U0001F4B0"  # 💰

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="💰 金融工程与风险", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="利率/收益率用小数（5% 记 0.05）", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))

        def row(title):
            f = ctk.CTkLabel(body, text=title, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["md"], 0))
            return f

        # 期权参数行
        row("期权：S K T r σ 类型(call/put)")
        self.opt = {}
        obox = ctk.CTkFrame(body, fg_color="transparent")
        obox.pack(fill="x")
        for i, (k, lab) in enumerate([("S", "标的"), ("K", "行权"), ("T", "期限"), ("r", "利率"), ("σ", "波动")]):
            ctk.CTkLabel(obox, text=lab).grid(row=0, column=i * 2, padx=(0, 2))
            e = ctk.CTkEntry(obox, width=70)
            e.grid(row=0, column=i * 2 + 1, padx=(0, ui.SPACE["sm"]))
            e.insert(0, {"S": "100", "K": "100", "T": "1", "r": "0.05", "σ": "0.2"}[k])
            self.opt[k] = e
        etype = ctk.CTkEntry(obox, width=60)
        etype.grid(row=1, column=1, padx=(0, ui.SPACE["sm"]), pady=(ui.SPACE["sm"], 0))
        etype.insert(0, "call")
        self.opt["type"] = etype

        # 债券参数行
        row("债券：面值 票息率 收益率 年限 年付息")
        self.bond = {}
        bbox = ctk.CTkFrame(body, fg_color="transparent")
        bbox.pack(fill="x")
        for i, (k, lab) in enumerate([("face", "面值"), ("coupon", "票息"), ("rate", "收益率"), ("years", "年限"), ("freq", "付息")]):
            ctk.CTkLabel(bbox, text=lab).grid(row=0, column=i * 2, padx=(0, 2))
            e = ctk.CTkEntry(bbox, width=64)
            e.grid(row=0, column=i * 2 + 1, padx=(0, ui.SPACE["sm"]))
            e.insert(0, {"face": "1000", "coupon": "0.05", "rate": "0.04", "years": "10", "freq": "2"}[k])
            self.bond[k] = e

        # VaR 参数行
        row("VaR：组合价值 日μ 日σ 天数 置信度")
        self.var = {}
        vbox = ctk.CTkFrame(body, fg_color="transparent")
        vbox.pack(fill="x")
        for i, (k, lab) in enumerate([("portfolio", "组合"), ("mu", "日μ"), ("sigma", "日σ"), ("days", "天数"), ("alpha", "置信")]):
            ctk.CTkLabel(vbox, text=lab).grid(row=0, column=i * 2, padx=(0, 2))
            e = ctk.CTkEntry(vbox, width=64)
            e.grid(row=0, column=i * 2 + 1, padx=(0, ui.SPACE["sm"]))
            e.insert(0, {"portfolio": "100000", "mu": "0.0002", "sigma": "0.02", "days": "10", "alpha": "0.95"}[k])
            self.var[k] = e

        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("Black-Scholes 定价", self.do_bs), ("希腊字母", self.do_greeks), ("隐含波动率", self.do_iv),
                          ("债券久期凸性", self.do_bond), ("蒙特卡洛 VaR", self.do_var)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_bs(self):
        o = self.opt
        self._run(black_scholes, _f(o["S"].get(), 1), _f(o["K"].get(), 1), _f(o["T"].get(), 1), _f(o["r"].get(), 0.05), _f(o["σ"].get(), 0.2), o["type"].get().strip())

    def do_greeks(self):
        o = self.opt
        self._run(option_greeks, _f(o["S"].get(), 1), _f(o["K"].get(), 1), _f(o["T"].get(), 1), _f(o["r"].get(), 0.05), _f(o["σ"].get(), 0.2), o["type"].get().strip())

    def do_iv(self):
        o = self.opt
        self._run(implied_vol, _f(o["S"].get(), 1), _f(o["K"].get(), 1), _f(o["T"].get(), 1), _f(o["r"].get(), 0.05), 10, o["type"].get().strip())

    def do_bond(self):
        b = self.bond
        self._run(bond_duration_convexity, _f(b["face"].get(), 1000), _f(b["coupon"].get(), 0.05), _f(b["rate"].get(), 0.04), _f(b["years"].get(), 10), _f(b["freq"].get(), 2))

    def do_var(self):
        v = self.var
        self._run(monte_carlo_var, _f(v["portfolio"].get(), 100000), _f(v["mu"].get(), 0.0002), _f(v["sigma"].get(), 0.02), _n(v["days"].get(), 10), _f(v["alpha"].get(), 0.95))


PAGES = [OptionsRiskPage]
