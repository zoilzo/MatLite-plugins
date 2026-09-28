# -*- coding: utf-8 -*-
"""数列：等差数列/等比数列的项与前 n 项和。AI 工具。"""
from modules import ai_tools

PLUGIN = {
    "id": "sequence_sum",
    "name": "数列与级数",
    "version": "1.0",
    "author": "zoilzo",
    "description": "等差/等比通项与前 n 项和、无穷等比和（AI 工具）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"a1": {"type": "number"}, "d": {"type": "number"}, "n": {"type": "number"}}, "required": ["a1", "d", "n"]}, category="数列")
def arith_seq(a1, d, n):
    """等差数列前 n 项：首项 a1、公差 d。"""
    a1, d, n = float(a1), float(d), int(n)
    terms = [a1 + i * d for i in range(n)]
    return {"text": f"首项={a1},公差={d},项数={n}: 前若干项={[round(t, 6) for t in terms[:10]]}" + (f"... 通项 a_n={a1}+(n-1)*{d}" if n > 10 else "")}


@ai_tools._reg
@ai_tools._tool({"properties": {"a1": {"type": "number"}, "d": {"type": "number"}, "n": {"type": "number"}}, "required": ["a1", "d", "n"]}, category="数列")
def arith_sum(a1, d, n):
    """等差数列前 n 项和：Sn = n/2 * (2a1 + (n-1)d)。"""
    a1, d, n = float(a1), float(d), int(n)
    s = n / 2 * (2 * a1 + (n - 1) * d)
    return {"text": f"等差数列 a1={a1},d={d},n={n}: 前 {n} 项和 = {s:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a1": {"type": "number"}, "r": {"type": "number"}, "n": {"type": "number"}}, "required": ["a1", "r", "n"]}, category="数列")
def geo_seq(a1, r, n):
    """等比数列前 n 项：首项 a1、公比 r。"""
    a1, r, n = float(a1), float(r), int(n)
    terms = [a1 * r ** i for i in range(n)]
    return {"text": f"首项={a1},公比={r},项数={n}: 前若干项={[round(t, 6) for t in terms[:10]]}" + (f"... 通项 a_n={a1}*{r}^(n-1)" if n > 10 else "")}


@ai_tools._reg
@ai_tools._tool({"properties": {"a1": {"type": "number"}, "r": {"type": "number"}, "n": {"type": "number"}}, "required": ["a1", "r", "n"]}, category="数列")
def geo_sum_finite(a1, r, n):
    """等比数列前 n 项和（r != 1）。"""
    a1, r, n = float(a1), float(r), int(n)
    if r == 1:
        return {"text": f"公比=1，前 {n} 项和 = {a1 * n:.6g}"}
    s = a1 * (1 - r ** n) / (1 - r)
    return {"text": f"等比数列 a1={a1},r={r},n={n}: 前 {n} 项和 = {s:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a1": {"type": "number"}, "r": {"type": "number"}}, "required": ["a1", "r"]}, category="数列")
def geo_sum_inf(a1, r):
    """无穷等比级数和（|r|<1 收敛）：S = a1/(1-r)。"""
    a1, r = float(a1), float(r)
    if abs(r) >= 1:
        return {"text": f"|r|={abs(r)} >= 1，级数发散（无有限和）"}
    s = a1 / (1 - r)
    return {"text": f"无穷等比 a1={a1},r={r}: 收敛和 = {s:.6g}"}
