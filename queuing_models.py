# -*- coding: utf-8 -*-
"""排队论：M/M/1、M/M/c、M/D/1 队列指标、Erlang-C 等待概率、利特尔定律。AI 工具 + 页面。"""
import customtkinter as ctk
import math

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "queuing_models",
    "name": "排队论",
    "version": "1.0",
    "author": "zoilzo",
    "description": "M/M/1、M/M/c、M/D/1 排队指标、Erlang-C 等待概率、利特尔定律（工具 + 页面）",
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


def _erlang_c(a, c):
    """Erlang-C：M/M/c 系统中顾客必须等待的概率。a=λ/μ 为业务量，c 为服务台数。"""
    s = sum(a ** n / math.factorial(n) for n in range(c))
    last = a ** c / math.factorial(c)
    rho = a / c
    denom = (1 - rho) * s + last
    return last / denom if denom else 1.0


@ai_tools._reg
@ai_tools._tool({"properties": {"arrival": {"type": "number"}, "service": {"type": "number"}}, "required": []}, category="排队论")
def mm1_queue(arrival=5, service=6):
    """M/M/1 单服务台模型：到达率 λ、服务率 μ。返回利用率、队长、等待队长、平均流速时间等。"""
    la, mu = _f(arrival, 5), _f(service, 6)
    if la <= 0 or mu <= 0:
        return {"text": "到达率与服务率必须为正数。"}
    rho = la / mu
    if rho >= 1:
        return {"text": "不稳定队列：利用率 ρ=%.3f ≥ 1（服务率须大于到达率）。" % rho}
    L = rho / (1 - rho)
    Lq = rho * rho / (1 - rho)
    W = L / la
    Wq = Lq / la
    P0 = 1 - rho
    return {"text": ("M/M/1（λ=%.4g, μ=%.4g）\n利用率 ρ = %.4f\n系统中平均数 L = %.4f\n队列平均 Lq = %.4f\n系统逗留 W = %.4f 单位时间\n排队等待 Wq = %.4f\n空闲概率 P0 = %.4f" % (la, mu, rho, L, Lq, W, Wq, P0))}


@ai_tools._reg
@ai_tools._tool({"properties": {"arrival": {"type": "number"}, "service": {"type": "number"}, "servers": {"type": "number"}}, "required": []}, category="排队论")
def mmc_queue(arrival=5, service=6, servers=2):
    """M/M/c 多服务台模型：到达率 λ、服务率 μ、服务台数 c。基于 Erlang-C 计算等待指标。"""
    la, mu = _f(arrival, 5), _f(service, 6)
    c = max(1, _n(servers, 2))
    if la <= 0 or mu <= 0:
        return {"text": "到达率与服务率必须为正数。"}
    a = la / mu
    rho = a / c
    if rho >= 1:
        return {"text": "不稳定：业务量 a=%.4f ≥ 服务台数 %d，须增加服务台。" % (a, c)}
    pw = _erlang_c(a, c)
    Lq = pw * rho / (1 - rho)
    Wq = Lq / la
    W = Wq + 1 / mu
    L = Lq + a
    return {"text": ("M/M/%d（λ=%.4g, μ=%.4g）\n业务量 a = %.4f，每台利用率 = %.4f\n需等待概率 Erlang-C = %.4f\n队列平均 Lq = %.4f\n等待时间 Wq = %.4f 单位时间\n系统逗留 W = %.4f\n系统平均 L = %.4f" % (c, la, mu, a, rho, pw, Lq, Wq, W, L))}


@ai_tools._reg
@ai_tools._tool({"properties": {"arrival": {"type": "number"}, "service": {"type": "number"}, "servers": {"type": "number"}}, "required": []}, category="排队论")
def erlang_c(arrival=5, service=6, servers=2):
    """Erlang-C：M/M/c 系统中新到顾客须排队等待的概率（电话/服务台阻塞概率）。"""
    la, mu = _f(arrival, 5), _f(service, 6)
    c = max(1, _n(servers, 2))
    a = la / mu
    if a >= c:
        return {"text": "业务量 a=%.4f ≥ 服务台数 %d，系统不稳定。" % (a, c)}
    return {"text": "Erlang-C 等待概率 = %.4f（业务量 a=%.4f, %d 台）" % (_erlang_c(a, c), a, c)}


@ai_tools._reg
@ai_tools._tool({"properties": {"arrival": {"type": "number"}, "service": {"type": "number"}}, "required": []}, category="排队论")
def md1_queue(arrival=5, service=6):
    """M/D/1 单服务台、服务时间确定：返回队长、等待队长、逗留时间等。"""
    la, mu = _f(arrival, 5), _f(service, 6)
    if la <= 0 or mu <= 0:
        return {"text": "到达率与服务率必须为正数。"}
    rho = la / mu
    if rho >= 1:
        return {"text": "不稳定队列：利用率 ρ=%.3f ≥ 1。" % rho}
    Lq = rho * rho / (2 * (1 - rho))
    Wq = Lq / la
    W = Wq + 1 / mu
    L = Lq + rho
    return {"text": ("M/D/1（λ=%.4g, μ=%.4g）\n利用率 ρ = %.4f\n队列平均 Lq = %.4f\n等待时间 Wq = %.4f\n系统逗留 W = %.4f\n系统平均 L = %.4f" % (la, mu, rho, Lq, Wq, W, L))}


@ai_tools._reg
@ai_tools._tool({"properties": {"arrival": {"type": "number"}, "wait": {"type": "number"}}, "required": []}, category="排队论")
def little_law(arrival=5, wait=0.2):
    """利特尔定律 L=λW：由到达率与平均逗留时间求系统平均顾客数（或反求）。"""
    la, w = _f(arrival, 5), _f(wait, 0.2)
    if la <= 0:
        return {"text": "到达率必须为正数。"}
    L = la * w
    return {"text": "利特尔定律：L = λ·W = %.4f × %.4f = %.4f" % (la, w, L)}


class QueuingPage(ui.BasePage):
    NAME = "排队论"
    EMOJI = "\U0001F6E1\U0000FE0F"  # 🛡️

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🛡️ 排队论", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="利用率 ρ=λ/μ（服务率需大于到达率）；服务台数 c 仅用于 M/M/c",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.ent = {}
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["md"], 0))
        for i, (k, lab) in enumerate([("arrival", "到达率"), ("service", "服务率"), ("servers", "服务台")]):
            ctk.CTkLabel(row, text=lab).grid(row=0, column=i * 2, padx=(0, 2))
            e = ctk.CTkEntry(row, width=70)
            e.grid(row=0, column=i * 2 + 1, padx=(0, ui.SPACE["sm"]))
            e.insert(0, {"arrival": "5", "service": "6", "servers": "2"}[k])
            self.ent[k] = e
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("M/M/1", self.do_mm1), ("M/M/c", self.do_mmc), ("M/D/1", self.do_md1),
                          ("Erlang-C", self.do_ec), ("利特尔定律", self.do_ll)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _val(self, k):
        return self.ent[k].get().strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_mm1(self):
        self._run(mm1_queue, _f(self._val("arrival"), 5), _f(self._val("service"), 6))

    def do_mmc(self):
        self._run(mmc_queue, _f(self._val("arrival"), 5), _f(self._val("service"), 6), _n(self._val("servers"), 2))

    def do_md1(self):
        self._run(md1_queue, _f(self._val("arrival"), 5), _f(self._val("service"), 6))

    def do_ec(self):
        self._run(erlang_c, _f(self._val("arrival"), 5), _f(self._val("service"), 6), _n(self._val("servers"), 2))

    def do_ll(self):
        self._run(little_law, _f(self._val("arrival"), 5), 0.2)


PAGES = [QueuingPage]
