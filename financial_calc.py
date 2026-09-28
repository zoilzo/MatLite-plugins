# -*- coding: utf-8 -*-
"""财务：单利/复利/等额本息还款/现值终值。AI 工具。"""
from modules import ai_tools

PLUGIN = {
    "id": "financial_calc",
    "name": "理财与利息",
    "version": "1.0",
    "author": "zoilzo",
    "description": "单利/复利/等额本息月供/现值终值（AI 工具）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number", "description": "本金"}, "r": {"type": "number", "description": "每期利率(小数，如0.05=5%)"}, "n": {"type": "number", "description": "期数"}}, "required": ["p", "r", "n"]}, category="金融")
def simple_interest(p, r, n):
    """单利：利息 = P*r*n，本息合计 = P*(1+rn)。"""
    p, r, n = float(p), float(r), int(n)
    interest = p * r * n
    return {"text": f"单利：本金 {p}，利率 {r:.4g}，{n} 期； 利息={interest:.6g}，本息合计={p + interest:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number", "description": "本金"}, "r": {"type": "number", "description": "每期利率(小数)"}, "n": {"type": "number", "description": "期数"}}, "required": ["p", "r", "n"]}, category="金融")
def compound_interest(p, r, n):
    """复利：终值 = P*(1+r)^n，利息 = 终值-本金。"""
    p, r, n = float(p), float(r), int(n)
    fv = p * (1 + r) ** n
    return {"text": f"复利：本金 {p}，每期利率 {r:.4g}，{n} 期； 终值={fv:.6g}，利息={fv - p:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number", "description": "本金/现值"}, "r": {"type": "number", "description": "每期利率(小数)"}, "n": {"type": "number", "description": "期数"}}, "required": ["p", "r", "n"]}, category="金融")
def future_value(p, r, n):
    """现值 P 在每期利率 r、n 期后的终值。"""
    return compound_interest(p, r, n)


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number", "description": "贷款额"}, "annual_rate": {"type": "number", "description": "年利率(小数，如0.05=5%)"}, "months": {"type": "number", "description": "月数"}}, "required": ["p", "annual_rate", "months"]}, category="金融")
def loan_pmt(p, annual_rate, months):
    """等额本息每月还款：P*r/(1-(1+r)^-n)，r=年利率/12。"""
    p, ar, months = float(p), float(annual_rate), int(months)
    if months <= 0:
        return {"text": "月数需大于 0"}
    r = ar / 12.0
    if r == 0:
        return {"text": f"贷款 {p}，{months} 个月：每月还 {p / months:.6g}"}
    pmt = p * r / (1 - (1 + r) ** (-months))
    total = pmt * months
    return {"text": f"贷款 {p}，年利率 {ar:.4g}，{months} 个月：每月还 {pmt:.6g}，总还款 {total:.6g}，总利息 {total - p:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"fv": {"type": "number", "description": "终值"}, "r": {"type": "number", "description": "每期利率(小数)"}, "n": {"type": "number", "description": "期数"}}, "required": ["fv", "r", "n"]}, category="金融")
def present_value(fv, r, n):
    """终值 fv 折现：现值 = fv/(1+r)^n。"""
    fv, r, n = float(fv), float(r), int(n)
    pv = fv / ((1 + r) ** n)
    return {"text": f"终值 {fv}，每期利率 {r:.4g}，{n} 期； 现值 = {pv:.6g}"}
