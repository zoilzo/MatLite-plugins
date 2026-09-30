# -*- coding: utf-8 -*-
"""数据挖掘：Apriori 频繁项集与关联规则、PCA 载荷、异常检测(z 分数)、聚类质量评估。AI 工具 + 页面。"""
import customtkinter as ctk
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "data_mining",
    "name": "数据挖掘",
    "version": "1.0",
    "author": "zoilzo",
    "description": "Apriori 频繁项集/关联规则、PCA 载荷、异常检测、聚类质量（工具 + 页面）",
}


def _f(v, d):
    """稳健数值：容忍 None/字符串，失败用默认值。"""
    if v is None:
        return float(d)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").replace("，", "").replace("%", "").strip()
    if s in ("", "-", "无", "空", "None"):
        return float(d)
    try:
        return float(s)
    except Exception:
        return float(d)


def _n(v, d):
    """整数解析：保证非负。"""
    return max(0, int(_f(v, d)))


def _fmt(x):
    """数值显示。"""
    x = float(x)
    if abs(x) >= 1e6 or (abs(x) < 1e-4 and x != 0):
        return f"{x:.4g}"
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _mat(s):
    """解析数据矩阵（行以分号分隔，行内数字以空格或逗号分隔）为 numpy 二维数组。"""
    s = str(s).strip()
    if not s:
        raise ValueError("矩阵为空")
    rows = []
    for r0 in s.split(";"):
        r = r0.replace(",", " ").replace("[", "").replace("]", "").strip()
        parts = [p for p in r.split() if p]
        if not parts:
            continue
        rows.append([float(p) for p in parts])
    if not rows:
        raise ValueError("矩阵无法解析")
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("矩阵各行长度不一致")
    return np.asarray(rows, dtype=float)


def _tx(transactions):
    """解析交易集：每笔交易以分号或换行分隔，项以逗号分隔。"""
    s = str(transactions).strip()
    if not s:
        return []
    out = []
    for row in s.replace("\n", ";").split(";"):
        row = row.strip()
        if not row:
            continue
        items = [x.strip() for x in row.replace(" ", ",").split(",") if x.strip()]
        if items:
            out.append(items)
    return out


@ai_tools._reg
@ai_tools._tool({"properties": {"transactions": {"type": "string"}, "min_support": {"type": "number"}}, "required": []}, category="数据挖掘")
def apriori_itemsets(transactions="A,B,C; B,C,D; A,C; A,B,C,D", min_support=0.4):
    """Apriori 频繁项集：每笔交易以分号分隔、项以逗号分隔，返回支持度达标的频繁项集。"""
    tx = _tx(transactions)
    if not tx:
        return {"text": "交易数据为空。"}
    n = len(tx)
    minsup = _f(min_support, 0.4)
    from collections import Counter
    cnt = Counter()
    for t in tx:
        for it in set(t):
            cnt[it] += 1
    freq = {frozenset([it]): cnt[it] / n for it in cnt if cnt[it] / n >= minsup}
    out = list(freq.items())

    def has_freq(c):
        for x in c:
            sub = c - {x}
            if sub and frozenset(sub) not in freq:
                return False
        return True

    k = 2
    while True:
        prev = [frozenset(s) for s in freq if len(s) == k - 1]
        cand = set()
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                u = prev[i] | prev[j]
                if len(u) == k and has_freq(u):
                    cand.add(u)
        if not cand:
            break
        new = {}
        for c in cand:
            csup = sum(1 for t in tx if c.issubset(set(t))) / n
            if csup >= minsup:
                new[c] = csup
        if not new:
            break
        freq.update(new)
        out.extend(new.items())
        k += 1
    if not out:
        return {"text": "无满足最小支持度的频繁项集。"}
    out.sort(key=lambda kv: (len(kv[0]), tuple(sorted(kv[0]))))
    lines = ["{%s}: %.3f" % (", ".join(sorted(s)), sup) for s, sup in out]
    return {"text": "频繁项集（支持度≥%.2f，%d 笔交易）\n%s" % (minsup, n, "\n".join(lines))}


@ai_tools._reg
@ai_tools._tool({"properties": {"transactions": {"type": "string"}, "min_conf": {"type": "number"}}, "required": []}, category="数据挖掘")
def association_rules(transactions="A,B,C; B,C,D; A,C; A,B,C,D", min_conf=0.6):
    """关联规则：由频繁项集生成 X→Y 规则（置信度=sup(X∪Y)/sup(X)），返回达标规则。"""
    tx = _tx(transactions)
    if not tx:
        return {"text": "交易数据为空。"}
    n = len(tx)
    minconf = _f(min_conf, 0.6)
    from collections import Counter
    from itertools import combinations
    cnt = Counter()
    for t in tx:
        for it in set(t):
            cnt[it] += 1
    items = sorted(set(cnt))
    occ = {frozenset([it]): cnt[it] / n for it in items}

    def sup(s):
        ss = set(s)
        return sum(1 for t in tx if ss.issubset(set(t))) / n

    rules = []
    for size in range(2, len(items) + 1):
        for comb in combinations(items, size):
            c = frozenset(comb)
            s = sup(c)
            if s <= 0:
                continue
            for r in range(1, size):
                for ant in combinations(comb, r):
                    ant = frozenset(ant)
                    cons = c - ant
                    if not cons:
                        continue
                    s_ant = occ.get(ant, sup(ant))
                    conf = (s / s_ant) if s_ant else 0.0
                    if conf >= minconf:
                        rules.append((ant, cons, conf, s))
                    occ[ant] = s_ant
            occ[c] = s
    if not rules:
        return {"text": "无满足最小置信度的关联规则。"}
    rules.sort(key=lambda r: -r[2])
    lines = ["{%s} → {%s} (conf=%.3f, sup=%.3f)" % (", ".join(sorted(a)), ", ".join(sorted(c)), cf, s) for a, c, cf, s in rules]
    return {"text": "关联规则（置信度≥%.2f，%d 条）\n%s" % (minconf, len(rules), "\n".join(lines))}


@ai_tools._reg
@ai_tools._tool({"properties": {"matrix": {"type": "string"}, "components": {"type": "number"}}, "required": []}, category="数据挖掘")
def pca_loadings(matrix="5 4; 4 3; 5 3; 6 5", components=2):
    """主成分分析：行=样本，列=特征。返回各主成分方差占比与特征载荷。"""
    a = _mat(matrix)
    if a.shape[0] < 2:
        return {"text": "至少需要 2 个样本。"}
    from sklearn.decomposition import PCA
    k = max(1, min(_n(components, 2), a.shape[1], a.shape[0]))
    X = a - a.mean(axis=0)
    pca = PCA(n_components=k)
    pca.fit(X)
    ev = pca.explained_variance_ratio_
    lines = ["主成分 %d: 方差占比=%.3f" % (i + 1, e) for i, e in enumerate(ev)]
    for i in range(k):
        load = pca.components_[i]
        lines.append("  PC%d 载荷: %s" % (i + 1, ", ".join("%.3f" % c for c in load)))
    return {"text": "PCA（%d 特征，%d 样本）\n%s" % (a.shape[1], a.shape[0], "\n".join(lines))}


@ai_tools._reg
@ai_tools._tool({"properties": {"series": {"type": "string"}, "z": {"type": "number"}}, "required": []}, category="数据挖掘")
def anomaly_zscore(series="10,12,11,100,13,12,10", z=2.5):
    """异常检测：按 z 分数找出离群值（|z| 超过阈值者）。"""
    xs = [_f(x, 0.0) for x in str(series).replace(" ", ",").split(",") if str(x).strip() != ""]
    if len(xs) < 3:
        return {"text": "至少需要 3 个数据点。"}
    arr = np.array(xs, dtype=float)
    mu = arr.mean()
    sd = arr.std(ddof=0)
    if sd < 1e-12:
        return {"text": "数据无波动（标准差为 0），无离群值。"}
    zt = _f(z, 2.5)
    zs = [(float(x), (float(x) - mu) / sd) for x in arr]
    flags = [(x, zz) for x, zz in zs if abs(zz) > zt]
    if not flags:
        return {"text": "未发现 |z|>%.2f 的离群值（均值=%.3f，标准差=%.3f）。" % (zt, mu, sd)}
    return {"text": "离群值（|z|>%.2f，%d 个）：\n%s" % (zt, len(flags), "\n".join("%s (z=%.3f)" % (x, zz) for x, zz in flags))}


@ai_tools._reg
@ai_tools._tool({"properties": {"points": {"type": "string"}, "labels": {"type": "string"}}, "required": []}, category="数据挖掘")
def cluster_quality(points="0,0;1,1;2,0;10,10;11,11;9,10", labels="0,0,0,1,1,1"):
    """聚类质量：给定点的坐标与簇标签，计算簇内平方和(WCSS)与轮廓系数。"""
    X = _mat(points)
    lab = [_f(x, 0.0) for x in str(labels).replace(" ", ",").split(",") if str(x).strip() != ""]
    if X.shape[0] != len(lab):
        return {"text": "点数(%d)与标签数(%d)不一致。" % (X.shape[0], len(lab))}
    labs = np.array(lab, dtype=int)
    if len(set(labs.tolist())) < 2:
        return {"text": "至少需要 2 个不同簇。"}
    wcss = 0.0
    for c in set(labs.tolist()):
        pts = X[labs == c]
        if len(pts):
            wcss += float(((pts - pts.mean(axis=0)) ** 2).sum())
    try:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(X, labs))
    except Exception:
        sil = None
    out = "簇内平方和(WCSS)=%.4f" % wcss
    if sil is not None:
        out += "，轮廓系数=%.4f" % sil
    return {"text": out}


class DataMiningPage(ui.BasePage):
    NAME = "数据挖掘"
    EMOJI = "\U0001F9ED"  # 🧭

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🧭 数据挖掘", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="交易集以分号分、项以逗号分（A,B,C; B,C,D）；数据行以分号分、列以空格/逗号分",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.tx = self._box(body, "交易 / 数据")
        self.tx.insert("1.0", "A,B,C; B,C,D; A,C; A,B,C,D")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("频繁项集", self.do_itemset), ("关联规则", self.do_rules), ("PCA 载荷", self.do_pca),
                          ("异常检测", self.do_anom), ("聚类质量", self.do_sil)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def _box(self, frame, title):
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        t = ctk.CTkTextbox(frame, height=70, font=ui.mono(ui.FONT["body"]))
        t.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        return t

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _val(self):
        return self.tx.get("1.0", "end").strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_itemset(self):
        self._run(apriori_itemsets, self._val())

    def do_rules(self):
        self._run(association_rules, self._val())

    def do_pca(self):
        self._run(pca_loadings, "5 4; 4 3; 5 3; 6 5")

    def do_anom(self):
        self._run(anomaly_zscore, "10,12,11,100,13,12,10")

    def do_sil(self):
        self._run(cluster_quality, "0,0;1,1;2,0;10,10;11,11;9,10", "0,0,0,1,1,1")


PAGES = [DataMiningPage]
