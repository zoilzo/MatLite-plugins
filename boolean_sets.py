# -*- coding: utf-8 -*-
"""离散数学基础：命题逻辑真值表 / 集合运算 / 德摩根验证 / 逻辑门。

给计算机、专升本、离散数学的同学快速做真值表、集合交并补与逻辑门判断。
（AI 工具 + 页面）
"""
import itertools
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "boolean_sets",
    "name": "离散数学基础",
    "version": "1.0",
    "author": "zoilzo",
    "description": "命题逻辑真值表 / 集合交并补·德摩根 / 逻辑门 AND·OR·NOT·NAND·XOR（工具 + 页面）",
}


def _comma_list(s):
    s = s.replace("，", ",").replace(";", ",").replace("；", ",").replace("\n", ",")
    return [x.strip() for x in s.split(",") if str(x).strip() != ""]


def _set_of(s):
    return set(_comma_list(s))


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"formula": {"type": "string", "default": "(A >> B) & (B | C)"}, "vars": {"type": "string", "default": "A,B,C"}}, "required": ["formula"]}, category="离散数学")
def truth_table(formula="(A >> B) & (B | C)", vars="A,B,C"):
    """命题逻辑真值表。formula 用 & (与)、| (或)、~ (非)、>> (蕴含)、== (等值) 连接变量；vars 列出变量名。返回真值表并判断永真/永假/可满足。"""
    import sympy as sp
    names = [v.strip().upper() for v in _comma_list(vars)]
    if not names:
        names = sorted(set(ch for ch in formula if ch.isalpha()))
    syms = {n: sp.Symbol(n) for n in names}
    try:
        expr = sp.parse_expr(formula, local_dict=syms)
    except Exception as e:
        return {"text": "无法解析公式：" + repr(e) + "\n请用 & | ~ >> <-> 与变量连接，如 (A & B) | ~C。"}
    header = "  ".join(names) + " | 结果"
    rows = []
    true_count = 0
    total = 0
    for combo in itertools.product([True, False], repeat=len(names)):
        sub = {n: v for n, v in zip(names, combo)}
        val = bool(expr.subs(sub))
        total += 1
        if val:
            true_count += 1
        rows.append("  ".join(" 1 " if v else " 0 " for v in combo) + "  |  " + ("1" if val else "0"))
    if true_count == total:
        verdict = "永真式（重言式）"
    elif true_count == 0:
        verdict = "永假式（矛盾式）"
    else:
        verdict = "可满足式（非永真）"
    return {"text": header + "\n" + "\n".join(rows) + "\n共 {0} 行，真 {1} 行 → {2}".format(total, true_count, verdict)}


@ai_tools._reg
@ai_tools._tool({"properties": {"set_a": {"type": "string", "default": "1,2,3,4"}, "set_b": {"type": "string", "default": "3,4,5,6"}}, "required": ["set_a", "set_b"]}, category="离散数学")
def set_operations(set_a="1,2,3,4", set_b="3,4,5,6"):
    """集合运算：交、并、差、对称差、笛卡尔积与基数。set_a、set_b 为用逗号分隔的元素。"""
    A = _set_of(set_a)
    B = _set_of(set_b)
    inter = sorted(A & B)
    uni = sorted(A | B)
    diff = sorted(A - B)
    sym = sorted(A ^ B)
    lines = ["集合 A = {%s}" % ", ".join(sorted(A, key=str))]
    lines.append("集合 B = {%s}" % ", ".join(sorted(B, key=str)))
    lines.append("交 A∩B = {%s}   |A∩B| = %d" % (", ".join(inter), len(inter)))
    lines.append("并 A∪B = {%s}   |A∪B| = %d" % (", ".join(uni), len(uni)))
    lines.append("差 A−B = {%s}   |A−B| = %d" % (", ".join(diff), len(diff)))
    lines.append("对称差 A△B = {%s}   |A△B| = %d" % (", ".join(sym), len(sym)))
    lines.append("笛卡尔积 |A×B| = |A|·|B| = %d×%d=%d" % (len(A), len(B), len(A) * len(B)))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"universe": {"type": "string", "default": "1,2,3,4,5,6"}, "set_a": {"type": "string", "default": "1,2,3"}, "set_b": {"type": "string", "default": "3,4,5"}}, "required": ["universe", "set_a", "set_b"]}, category="离散数学")
def demorgan_check(universe="1,2,3,4,5,6", set_a="1,2,3", set_b="3,4,5"):
    """验证德摩根定律：~(A∪B) = ~A ∩ ~B 与 ~(A∩B) = ~A ∪ ~B（~为补集，相对全集 U）。"""
    U = _set_of(universe)
    A = _set_of(set_a)
    B = _set_of(set_b)
    comp = lambda s: U - s
    union_comp = comp(A | B)
    inter_of_comp = comp(A) & comp(B)
    inter_comp = comp(A & B)
    union_of_comp = comp(A) | comp(B)
    lines = ["全集 U = {%s}" % ", ".join(sorted(U, key=str))]
    lines.append("A = {%s}  |  B = {%s}" % (", ".join(sorted(A, key=str)), ", ".join(sorted(B, key=str))))
    lines.append("")
    lines.append("德摩根 ①  ~(A∪B) = {%s}" % ", ".join(sorted(union_comp, key=str)))
    lines.append("      ~A∩~B = {%s}   → %s" % (", ".join(sorted(inter_of_comp, key=str)), "✓ 成立" if union_comp == inter_of_comp else "✗ 不成立"))
    lines.append("")
    lines.append("德摩根 ②  ~(A∩B) = {%s}" % ", ".join(sorted(inter_comp, key=str)))
    lines.append("      ~A∪~B = {%s}   → %s" % (", ".join(sorted(union_of_comp, key=str)), "✓ 成立" if inter_comp == union_of_comp else "✗ 不成立"))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"gate": {"type": "string", "default": "XOR"}, "a": {"type": "number", "default": 1}, "b": {"type": "number", "default": 0}}, "required": ["gate"]}, category="离散数学")
def logic_gate(gate="XOR", a=1, b=0):
    """逻辑门输出：gate 为 AND/OR/NOT/NAND/NOR/XOR/XNOR，a、b 为输入位(0/1)。返回结果并给出完整真值表。"""
    g = (gate or "AND").strip().upper()
    a = int(a)                          if isinstance(a, (int, float)) else (1 if str(a).strip().lower() in ("1", "true", "on", "真") else 0)
    b = int(b)                          if isinstance(b, (int, float)) else (1 if str(b).strip().lower() in ("1", "true", "on", "真") else 0)

    def apply(x, y):
        if g in ("AND", "与"):
            return int(x and y)
        if g in ("OR", "或"):
            return int(x or y)
        if g in ("NOT", "非"):
            return int(not x)
        if g in ("NAND", "与非"):
            return int(not (x and y))
        if g in ("NOR", "或非"):
            return int(not (x or y))
        if g in ("XOR", "异或"):
            return int(x != y)
        if g in ("XNOR", "同或"):
            return int(x == y)
        return None

    if g == "NOT":
        res = apply(a, b)
        return {"text": "NOT {0} = {1}\n真值表：0→1，1→0".format(a, res)}
    rows = []
    for x in (0, 1):
        for y in (0, 1):
            rows.append("  {0} {1} | {2}".format(x, y, apply(x, y)))
    return {"text": "{0}({1}, {2}) = {3}\n真值表\n A B | 出\n{4}".format(g, a, b, apply(a, b), "\n".join(rows))}


# ============================================================
# 页面
# ============================================================
class BooleanSetsPage(ui.BasePage):
    NAME = "离散数学基础"
    EMOJI = "\U0001D400"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="𝐀 离散数学基础", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="命题公式（用 & | ~ >> <-> 连接，如 (A>>B)&(B|C)）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.formula = ctk.CTkEntry(body)
        self.formula.insert(0, "(A >> B) & (B | C)")
        self.formula.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="逻辑门（AND/OR/NOT/NAND/NOR/XOR/XNOR，默认试 XOR）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.gate = ctk.CTkEntry(body)
        self.gate.insert(0, "XOR")
        self.gate.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        ctk.CTkLabel(body, text="集合 A（逗号分隔） / 集合 B：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["xs"]))
        row.grid_columnconfigure(0, weight=1)
        row.grid_columnconfigure(1, weight=1)
        self.set_a = ctk.CTkEntry(row)
        self.set_a.insert(0, "1,2,3,4")
        self.set_a.grid(row=0, column=0, padx=(0, ui.SPACE["xs"]), sticky="ew")
        self.set_b = ctk.CTkEntry(row)
        self.set_b.insert(0, "3,4,5,6")
        self.set_b.grid(row=0, column=1, padx=(ui.SPACE["xs"], 0), sticky="ew")
        ctk.CTkLabel(body, text="全集 U（验德摩根，默认 1,2,3,4,5,6）：", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.universe = ctk.CTkEntry(body)
        self.universe.insert(0, "1,2,3,4,5,6")
        self.universe.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        ops = [
            ("真值表", lambda: self._run(truth_table, [self.formula.get(), "A,B,C"])),
            ("集合运算", lambda: self._run(set_operations, [self.set_a.get(), self.set_b.get()])),
            ("德摩根", lambda: self._run(demorgan_check, [self.universe.get(), self.set_a.get(), self.set_b.get()])),
            ("逻辑门", lambda: self._run(logic_gate, [self.gate.get(), 1, 0])),
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


PAGES = [BooleanSetsPage]