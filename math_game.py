# -*- coding: utf-8 -*-
"""心算挑战：随机算术题，可计时计分。AI 工具 + 页面（课堂教学）。"""
import random
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "math_game",
    "name": "心算挑战",
    "version": "1.0",
    "author": "zoilzo",
    "description": "随机心算题（加减/乘除/混合），可计时计分（工具 + 页面，适合课堂）",
}

# 难度 -> (下界, 上界, 说明)
_LEVELS = {
    "easy": (1, 12, "加减"),
    "medium": (2, 20, "乘除"),
    "hard": (2, 30, "混合"),
}
_LEVEL_LABEL = {"简单": "easy", "中等": "medium", "困难": "hard"}


def _gen_question(level="easy", rng=None):
    """生成一道心算题，返回 (题目字符串, 答案)。"""
    rng = rng or random
    lev = (level or "easy").lower()
    lo, hi, _kind = _LEVELS.get(lev, _LEVELS["easy"])
    if lev == "easy":
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        op = rng.choice(["+", "-"])
        if op == "-" and b > a:
            a, b = b, a
        ans = a + b if op == "+" else a - b
        return f"{a} {op} {b}", ans
    if lev == "medium":
        q = rng.randint(lo, hi)
        b = rng.randint(2, 9)
        op = rng.choice(["×", "÷"])
        if op == "×":
            return f"{q} × {b}", q * b
        return f"{q * b} ÷ {b}", q
    # hard: a×b ± c
    a = rng.randint(2, hi)
    b = rng.randint(2, 9)
    c = rng.randint(1, 30)
    op = rng.choice(["+", "-"])
    ans = a * b + c if op == "+" else a * b - c
    return f"{a} × {b} {op} {c}", ans


@ai_tools._reg
@ai_tools._tool({"properties": {"level": {"type": "string"}}}, category="游戏")
def math_question(level="easy"):
    """出一道心算题。level: easy/medium/hard。返回题目与答案。"""
    q, ans = _gen_question(level)
    return {"text": f"心算题（{level}）：{q} = ？ （答案：{ans}）"}


class MathGamePage(ui.BasePage):
    NAME = "心算挑战"
    EMOJI = "\U0001F3AF"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.score = 0
        self.tried = 0
        self.q_answer = 0
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🎯 心算挑战", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        # 难度选择
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["sm"], ui.SPACE["sm"]))
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="难度：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w", padx=(0, ui.SPACE["sm"]))
        self.level_var = ctk.StringVar(value="简单")
        self.level_menu = ctk.CTkOptionMenu(row, values=["简单", "中等", "困难"], variable=self.level_var, width=96)
        self.level_menu.grid(row=0, column=1, sticky="w")
        # 出题区
        self.q_label = ctk.CTkLabel(body, text="点「出题」开始", font=ctk.CTkFont(size=ui.FONT["headline"], weight="bold"), text_color=ui.body())
        self.q_label.pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        ctk.CTkLabel(body, text="输入答案：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.answer_entry = ctk.CTkEntry(body)
        self.answer_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        row2 = ctk.CTkFrame(body, fg_color="transparent")
        row2.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row2.grid_columnconfigure(0, weight=1)
        row2.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(row2, text="出题", height=h, command=self.new_q).grid(row=0, column=0, padx=(0, ui.SPACE["xs"]), sticky="ew")
        ctk.CTkButton(row2, text="对答案", height=h, fg_color="gray40", command=self.check).grid(row=0, column=1, padx=(ui.SPACE["xs"], 0), sticky="ew")
        # 成绩
        self.score_label = ctk.CTkLabel(body, text="答对 0 / 0（0%）", font=ctk.CTkFont(size=ui.FONT["body"]))
        self.score_label.pack(anchor="w", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("选择难度，点「出题」即可练习心算")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def new_q(self):
        lev = _LEVEL_LABEL.get(self.level_var.get(), "easy")
        q, ans = _gen_question(lev)
        self.q_answer = ans
        self.q_label.configure(text=f"{q} = ？")
        self.answer_entry.delete(0, "end")
        self.answer_entry.focus_set()

    def check(self):
        try:
            u = int(self.answer_entry.get().strip())
        except ValueError:
            self.msg("请输入整数答案")
            return
        self.tried += 1
        ok = (u == self.q_answer)
        if ok:
            self.score += 1
        self.score_label.configure(text=f"答对 {self.score} / {self.tried}（{100 * self.score // max(1, self.tried)}%）")
        self.msg(("答对！" if ok else f"不对，答案是 {self.q_answer}") + "\n已出下一题，继续~")
        self.new_q()


PAGES = [MathGamePage]
