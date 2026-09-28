# -*- coding: utf-8 -*-
"""几何：圆/球/圆柱/圆锥/长方体/三角形海伦公式。AI 工具。"""
import math
from modules import ai_tools

PLUGIN = {
    "id": "geometry_kit",
    "name": "几何计算",
    "version": "1.0",
    "author": "zoilzo",
    "description": "圆/球/圆柱/圆锥/长方体/三角形海伦公式（AI 工具）",
}


@ai_tools._reg
@ai_tools._tool({"properties": {"r": {"type": "number"}}, "required": ["r"]}, category="几何")
def circle_props(r):
    """圆：半径 r 的面积与周长。"""
    r = float(r)
    return {"text": f"圆 r={r}: 面积={math.pi * r * r:.6g}，周长={2 * math.pi * r:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"r": {"type": "number"}}, "required": ["r"]}, category="几何")
def sphere_props(r):
    """球：半径 r 的体积与表面积。"""
    r = float(r)
    return {"text": f"球 r={r}: 体积={4 / 3 * math.pi * r ** 3:.6g}，表面积={4 * math.pi * r * r:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"r": {"type": "number"}, "h": {"type": "number"}}, "required": ["r", "h"]}, category="几何")
def cylinder_props(r, h):
    """圆柱：半径 r、高 h 的体积/侧面积/全面积。"""
    r, h = float(r), float(h)
    return {"text": f"圆柱 r={r},h={h}: 体积={math.pi * r * r * h:.6g}，侧面积={2 * math.pi * r * h:.6g}，全面积={2 * math.pi * r * (r + h):.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"r": {"type": "number"}, "h": {"type": "number"}}, "required": ["r", "h"]}, category="几何")
def cone_props(r, h):
    """圆锥：半径 r、高 h 的体积与母线长。"""
    r, h = float(r), float(h)
    slant = math.hypot(r, h)
    return {"text": f"圆锥 r={r},h={h}: 体积={math.pi * r * r * h / 3:.6g}，母线={slant:.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"l": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}}, "required": ["l", "w", "h"]}, category="几何")
def box_volume(l, w, h):
    """长方体：长宽高的体积与表面积。"""
    l, w, h = float(l), float(w), float(h)
    return {"text": f"长方体 {l}x{w}x{h}: 体积={l * w * h:.6g}，表面积={2 * (l * w + l * h + w * h):.6g}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "number"}}, "required": ["a", "b", "c"]}, category="几何")
def heron_area(a, b, c):
    """三角形三边求面积（海伦公式），并提示是否为直角三角形。"""
    a, b, c = float(a), float(b), float(c)
    if a + b <= c or a + c <= b or b + c <= a:
        return {"text": f"边({a},{b},{c}) 不能构成三角形"}
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    sides = sorted([a, b, c])
    right = abs(sides[0] ** 2 + sides[1] ** 2 - sides[2] ** 2) < 1e-6
    return {"text": f"三角形({a},{b},{c}): 面积={area:.6g}，{'直角三角形' if right else '非直角三角形'}"}
