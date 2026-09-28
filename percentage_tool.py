# -*- coding: utf-8 -*-
"""百分比：占比/变化率/增加百分比。AI 工具。"""
from modules import ai_tools

PLUGIN = {
    "id": "percentage_tool",
    "name": "百分比计算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "占比百分数/变化率/加价百分比（AI 工具）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"part": {"type": "number"}, "total": {"type": "number"}}, "required": ["part", "total"]}, category="百分比")
def pct_of(part, total):
    """求 part 占 total 的百分比（total 为 0 时返回 0）。"""
    part, total = float(part), float(total)
    if total == 0:
        return {"text": "分母(total) 不能为 0"}
    v = part / total * 100
    return {"text": f"{part} 占 {total} 的 {v:.6g}%"}


@ai_tools._reg
@ai_tools._tool({"properties": {"old": {"type": "number"}, "new": {"type": "number"}}, "required": ["old", "new"]}, category="百分比")
def pct_change(old, new):
    """从 old 到 new 的变化百分比。"""
    old, new = float(old), float(new)
    if old == 0:
        return {"text": "旧值(old) 不能为 0"}
    v = (new - old) / old * 100
    return {"text": f"从 {old} 到 {new} 变化 {v:.6g}% （{'上升' if v >= 0 else '下降'}）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"x": {"type": "number", "description": "原值"}, "rate": {"type": "number", "description": "百分比(如 10 表示加 10%)"}, "op": {"type": "string", "description": "add=加(默认)，sub=减"}}, "required": ["x", "rate"]}, category="百分比")
def add_pct(x, rate, op="add"):
    """给 x 加上或减去 rate%。"""
    x, rate = float(x), float(rate)
    factor = 1 + rate / 100.0
    if str(op).lower() == "sub":
        factor = 1 - rate / 100.0
    return {"text": f"{x} {'加' if factor >= 1 else '减'}{rate}% = {x * factor:.6g}"}
