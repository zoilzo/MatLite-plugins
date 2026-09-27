# -*- coding: utf-8 -*-
"""随机样本：生成分布样本并给统计量。AI 工具。"""
import numpy as np
from modules import ai_tools

PLUGIN = {
    "id": "random_samples",
    "name": "随机样本",
    "version": "1.0",
    "author": "zoilzo",
    "description": "生成随机样本并给均值/标准差（AI 工具）",
}

@ai_tools._reg  # 注意：_reg 在上、_tool 在下
@ai_tools._tool({"properties": {
    "n": {"type": "number", "description": "样本量"},
    "dist": {"type": "string", "description": "分布：normal 或 uniform"},
    "mean": {"type": "number", "description": "均值（正态，默认 0）"},
    "sd": {"type": "number", "description": "标准差（正态，默认 1）"},
    "seed": {"type": "number", "description": "随机种子（默认 0）"}}, "required": ["n"]}, category="随机数")
def random_sample(n=10, dist="normal", mean=0, sd=1, seed=0):
    """生成随机样本并给统计量。dist: normal / uniform"""
    n = int(n)
    rng = np.random.default_rng(int(seed) if str(seed).strip() not in ("", "None") else None)
    if str(dist).lower() in ("normal", "norm"):
        a = rng.normal(float(mean), max(1e-9, float(sd)), n)
        name = "正态"
    else:
        a = rng.uniform(0, 1, n)
        name = "均匀(0-1)"
    return {"text": f"{name}样本 n={n}： 均值={a.mean():.4g} 标准差={a.std():.4g}； 前 8 个={list(np.round(a[:8], 4))}"}
