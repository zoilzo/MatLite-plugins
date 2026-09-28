# -*- coding: utf-8 -*-
"""单位换算：长度/质量/体积/温度。AI 工具。"""
from modules import ai_tools

PLUGIN = {
    "id": "unit_converter",
    "name": "单位换算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "长度/质量/体积/温度单位互转（AI 工具）",
}

# 各单位的“基准单位”换算系数
_LEN = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "inch": 0.0254, "ft": 0.3048, "mile": 1609.344}
_MASS = {"g": 0.001, "kg": 1.0, "t": 1000.0, "lb": 0.45359237, "oz": 0.028349523125}
_VOL = {"ml": 0.001, "l": 1.0, "m3": 1000.0, "gal": 3.785411784}


def _pick(unit, table):
    u = str(unit).lower().strip()
    if u not in table:
        raise ValueError("不支持的单位: " + u + "（可选: " + "/".join(table) + "）")
    return u


@ai_tools._reg
@ai_tools._tool({"properties": {"value": {"type": "number"}, "frm": {"type": "string"}, "to": {"type": "string"}}, "required": ["value", "frm", "to"]}, category="单位换算")
def convert_length(value, frm, to):
    """长度换算。单位：mm/cm/m/km/inch/ft/mile。"""
    frm = _pick(frm, _LEN)
    to = _pick(to, _LEN)
    v = float(value) * _LEN[frm] / _LEN[to]
    return {"text": f"{value} {frm} = {v:.6g} {to}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"value": {"type": "number"}, "frm": {"type": "string"}, "to": {"type": "string"}}, "required": ["value", "frm", "to"]}, category="单位换算")
def convert_mass(value, frm, to):
    """质量换算。单位：g/kg/t/lb/oz。"""
    frm = _pick(frm, _MASS)
    to = _pick(to, _MASS)
    v = float(value) * _MASS[frm] / _MASS[to]
    return {"text": f"{value} {frm} = {v:.6g} {to}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"value": {"type": "number"}, "frm": {"type": "string"}, "to": {"type": "string"}}, "required": ["value", "frm", "to"]}, category="单位换算")
def convert_volume(value, frm, to):
    """体积/容量换算。单位：ml/l/m3/gal。"""
    frm = _pick(frm, _VOL)
    to = _pick(to, _VOL)
    v = float(value) * _VOL[frm] / _VOL[to]
    return {"text": f"{value} {frm} = {v:.6g} {to}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"value": {"type": "number"}, "frm": {"type": "string"}, "to": {"type": "string"}}, "required": ["value", "frm", "to"]}, category="单位换算")
def convert_temperature(value, frm, to):
    """温度换算。单位：c（摄氏）/f（华氏）/k（开尔文）。"""
    frm = str(frm).lower().strip()
    to = str(to).lower().strip()
    if frm not in ("c", "f", "k") or to not in ("c", "f", "k"):
        raise ValueError("温度单位只支持 c/f/k")
    v = float(value)
    if frm == "c":
        c = v
    elif frm == "f":
        c = (v - 32.0) * 5.0 / 9.0
    else:
        c = v - 273.15
    if to == "c":
        out = c
    elif to == "f":
        out = c * 9.0 / 5.0 + 32.0
    else:
        out = c + 273.15
    return {"text": f"{value} {frm} = {out:.6g} {to}"}
