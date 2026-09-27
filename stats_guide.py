# -*- coding: utf-8 -*-
"""统计检验向导：根据研究问题推荐统计方法。AI 工具。"""
from modules import ai_tools

PLUGIN = {
    "id": "stats_guide",
    "name": "统计检验向导",
    "version": "1.0",
    "author": "zoilzo",
    "description": "按研究问题推荐统计检验方法（AI 工具）",
}

_GUIDE = {
    "两组均值": "独立样本 t 检验（Independent t-test）：比较两组连续变量的均值是否不同。"
          "曼-惠特尼 U（Mann-Whitney U）为其非参数版本。",
    "配对": "配对 t 检验（Paired t-test）：同一组前后两次测量。"
          "威尔科克森符号秩（Wilcoxon）为其非参数版本。",
    "多组均值": "单因素方差分析 ANOVA（one-way ANOVA）：比较三组及以上均值。"
          "克鲁斯卡尔-沃利斯（Kruskal-Wallis）为其非参数版本。",
    "分类关系": "卡方检验（Chi-square）：检验两个分类变量是否独立。",
    "相关性": "皮尔逊相关（Pearson）看线性相关；变量不满足正态时用斯皮尔曼（Spearman）。",
    "回归": "简单线性回归看一个自变量对因变量的影响；多自变量用多元线性回归。",
    "二分类结局": "逻辑回归（Logistic regression）：因变量是 0/1 时用。",
    "生存时间": "生存分析：比较两组生存曲线用 Log-rank 检验；多因素用 Cox 比例风险模型。",
}

@ai_tools._reg  # 注意：_reg 在上、_tool 在下
@ai_tools._tool({"properties": {
    "keyword": {"type": "string", "description": "研究问题关键词，如 两组均值/配对/多组均值/分类关系/相关性/回归/二分类结局/生存时间"}},
    "required": ["keyword"]}, category="统计向导")
def stats_guide(keyword):
    """按研究问题推荐统计方法与对应的非参数替代方案。"""
    key = str(keyword).strip()
    for k, v in _GUIDE.items():
        if k in key:
            return {"text": f"推荐：{v}"}
    allk = "、".join(_GUIDE.keys())
    return {"text": f"未识别关键词。可用：{allk}"}

@ai_tools._reg
@ai_tools._tool({"properties": {
    "p": {"type": "number", "description": "p 值"}},
    "required": ["p"]}, category="统计向导")
def p_value_help(p):
    """解读 p 值大小对应的统计结论。"""
    try:
        p = float(p)
    except Exception:
        return {"text": "请输入数值 p 值。"}
    if p < 0.001:
        return {"text": f"p={p:.6g}：极显著，拒绝 H0（P<0.001）。"}
    if p < 0.01:
        return {"text": f"p={p:.6g}：很显著，拒绝 H0（P<0.01）。"}
    if p < 0.05:
        return {"text": f"p={p:.6g}：显著，拒绝 H0（P<0.05）。"}
    return {"text": f"p={p:.6g}：不显著（P≥0.05），不能拒绝 H0。"}
