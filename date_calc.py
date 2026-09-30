# -*- coding: utf-8 -*-
"""日期时间：日期差/年龄/周几/闰年/日期加减/当前时间。纯标准库，AI 工具 + 页面。"""
import calendar
import datetime as dt

import customtkinter as ctk

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "date_calc",
    "name": "日期时间",
    "version": "1.0",
    "author": "zoilzo",
    "description": "日期差/年龄/周几/闰年/日期加减/当前时间（工具 + 页面）",
}

_WEEK = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def _parse(s):
    """把日期字符串解析成 date。支持 YYYY-MM-DD / YYYY-M-D / YYYY年M月D日 / YYYYMMDD。"""
    s = str(s).strip()
    if not s:
        raise ValueError("日期不能为空")
    s = s.replace("/", "-").replace(".", "-").replace("年", "-").replace("月", "-").replace("日", "")
    parts = [p for p in s.replace("-", " ").split() if p]
    if len(parts) == 3:
        y, m, d = parts
        return dt.date(int(y), int(m), int(d))
    if len(parts) == 1 and len(parts[0]) == 8 and parts[0].isdigit():
        s8 = parts[0]
        return dt.date(int(s8[:4]), int(s8[4:6]), int(s8[6:8]))
    for fmt in ("%Y-%m-%d", "%y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError("无法解析日期: " + str(s))


def _fmt(d):
    return d.strftime("%Y-%m-%d") + "（" + _WEEK[d.weekday()] + "）"


@ai_tools._reg
@ai_tools._tool({"properties": {"start": {"type": "string", "description": "起始日期，如 2024-01-01"}, "end": {"type": "string", "description": "截止日期，如 2024-12-31"}}, "required": ["start", "end"]}, category="日期时间")
def date_diff(start, end):
    """两个日期相隔的天数，并给出起止日各自星期与大致年月换算。"""
    d1, d2 = _parse(start), _parse(end)
    days = abs((d2 - d1).days)
    years, rem = divmod(days, 365.25)
    months = int(rem / 30.44)
    return {"text": f"从 {_fmt(d1)} 到 {_fmt(d2)}：\n相隔 {days} 天 ≈ {int(years)} 年 {months} 个月"}


@ai_tools._reg
@ai_tools._tool({"properties": {"date": {"type": "string", "description": "基础日期"}, "days": {"type": "integer", "description": "加减的天数（负数表示减）"}}, "required": ["date"]}, category="日期时间")
def add_days(date, days=0):
    """在指定日期上加上/减去若干天，返回新日期与星期。"""
    d = _parse(date)
    n = int(float(days))
    nd = d + dt.timedelta(days=n)
    return {"text": f"{_fmt(d)} 加 {n:+d} 天 = {_fmt(nd)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"birth": {"type": "string", "description": "出生日期"}, "on": {"type": "string", "description": "可选，计算基准日，默认今天"}}, "required": ["birth"]}, category="日期时间")
def age_calc(birth, on=""):
    """根据出生日期计算周岁年龄（岁/月/天），可指定计算基准日。"""
    b = _parse(birth)
    ref = _parse(on) if on and str(on).strip() else dt.date.today()
    if ref < b:
        return {"text": "基准日早于出生日，请检查日期。"}
    years = ref.year - b.year
    months = ref.month - b.month
    days = ref.day - b.day
    if days < 0:
        months -= 1
        last = (ref.replace(day=1) - dt.timedelta(days=1)).day
        days += last
    if months < 0:
        years -= 1
        months += 12
    return {"text": f"出生 {_fmt(b)}，截至 {_fmt(ref)}：\n年龄 = {years} 岁 {months} 个月 {days} 天"}


@ai_tools._reg
@ai_tools._tool({"properties": {"date": {"type": "string"}}, "required": ["date"]}, category="日期时间")
def weekday_of(date):
    """查询某日是星期几与所在周（ISO 周号）。"""
    d = _parse(date)
    iso = d.isocalendar()
    return {"text": f"{_fmt(d)}，ISO 第 {iso[1]} 周"}


@ai_tools._reg
@ai_tools._tool({"properties": {"year": {"type": "integer"}}, "required": ["year"]}, category="日期时间")
def leap_year(year):
    """判断某年是否为闰年，并给出 2 月天数。"""
    y = int(year)
    leap = calendar.isleap(y)
    feb = calendar.monthrange(y, len("ab"))[1]  # month index 2
    return {"text": f"{y} 年{'是' if leap else '不是'}闰年（2 月 {feb} 天）"}
@ai_tools._reg
@ai_tools._tool({}, category="日期时间")
def now_info():
    """当前日期时间与 Unix 时间戳。"""
    now = dt.datetime.now()
    return {"text": f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')}（{_WEEK[now.weekday()]}）\nUnix 时间戳：{int(now.timestamp())}s"}


class DatePage(ui.BasePage):
    NAME = "日期时间"
    EMOJI = "\U0001F4C5"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="📅 日期时间", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="日期格式：2024-01-01 或 2024/1/1 或 2024年1月1日",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.f_d1 = self._field(body, "起始 / 出生日期", "2000-01-01")
        self.f_d2 = self._field(body, "截止日期（可空）", "2026-09-30")
        self.f_days = self._field(body, "加减天数", "30")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("日期差", self.do_diff), ("年龄", self.do_age), ("加减日期", self.do_add), ("周几", self.do_week), ("闰年", self.do_leap), ("当前时间", self.do_now)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def _field(self, frame, label, default):
        ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        e = ctk.CTkEntry(frame)
        e.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        e.insert(0, default)
        return e

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

    def do_diff(self):
        self._run(date_diff, self.f_d1.get(), self.f_d2.get())

    def do_age(self):
        self._run(age_calc, self.f_d1.get(), self.f_d2.get())

    def do_add(self):
        self._run(add_days, self.f_d1.get(), self.f_days.get())

    def do_week(self):
        self._run(weekday_of, self.f_d1.get())

    def do_leap(self):
        self._run(leap_year, self.f_d1.get().split("-")[0] or "2024")

    def do_now(self):
        self._run(now_info)


PAGES = [DatePage]
