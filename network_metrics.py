# -*- coding: utf-8 -*-
"""网络分析：度/特征向量中心性、PageRank、社区发现(Girvan-Newman)、链路预测。AI 工具 + 页面。"""
import customtkinter as ctk
import numpy as np

from modules import ai_tools
from modules import ui_kit as ui

PLUGIN = {
    "id": "network_metrics",
    "name": "网络分析",
    "version": "1.0",
    "author": "zoilzo",
    "description": "度/特征向量中心性、PageRank、社区发现、链路预测（工具 + 页面）",
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


def _edges(edges="1-2,2-3,3-4,4-1,2-4"):
    """解析边表："1-2,2-3"（可含空格/中文逗号），返回 (邻接表 dict, 有序节点列表)。"""
    s = str(edges).strip()
    if not s:
        s = "1-2,2-3,3-4,4-1,2-4"
    s = s.replace("；", ",").replace("，", ",").replace("|", ",").replace(" ", ",")
    adj = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split("-")
        if len(parts) != 2:
            continue
        a, b = parts[0].strip(), parts[1].strip()
        if not a or not b or a == b:
            continue
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    nodes = sorted(adj.keys())
    return adj, nodes


def _adj_matrix(adj, nodes):
    """由邻接表构建对称 0/1 邻接矩阵。"""
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for a, neigh in adj.items():
        ia = idx[a]
        for b in neigh:
            A[ia, idx[b]] = 1.0
    return A, idx, n


def _edge_betweenness(adj, nodes):
    """Brandes 算法计算无权图各条边的介数（键为 frozenset({a,b})）。"""
    between = {frozenset((a, b)): 0.0 for a in nodes for b in adj.get(a, set())}
    for s in nodes:
        stack = []
        pred = {nd: [] for nd in nodes}
        sig = {nd: 0 for nd in nodes}
        sig[s] = 1
        dist = {nd: -1 for nd in nodes}
        dist[s] = 0
        q = [s]
        while q:
            v = q.pop(0)
            stack.append(v)
            for w in adj.get(v, set()):
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sig[w] += sig[v]
                    pred[w].append(v)
        delta = {nd: 0.0 for nd in nodes}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                coeff = (sig[v] / sig[w]) if sig[w] else 0.0
                add = coeff * (1.0 + delta[w])
                delta[v] += add
                between[frozenset((v, w))] += add
    return between


def _components(adj, nodes):
    """返回邻接表（可已删边的副本）的连通分量列表。"""
    seen = set()
    comps = []
    for nd in nodes:
        if nd in seen:
            continue
        stack = [nd]
        seen.add(nd)
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj.get(v, set()):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        comps.append(comp)
    return comps


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}}, "required": []}, category="网络分析")
def degree_centrality(edges="1-2,2-3,3-4,4-1,2-4"):
    """度中心性：节点连接数 / (节点数-1)。返回各节点中心性。"""
    adj, nodes = _edges(edges)
    n = len(nodes)
    if n <= 1:
        return {"text": "至少需要 2 个节点。"}
    out = [f"{nd}: {len(adj.get(nd, set())) / (n - 1):.3f}" for nd in nodes]
    return {"text": "度中心性（节点数=%d）\n%s" % (n, "，".join(out))}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "iters": {"type": "number"}}, "required": []}, category="网络分析")
def eigenvector_centrality(edges="1-2,2-3,3-4,4-1,2-4", iters=100):
    """特征向量中心性：主特征向量幂迭代，衡量节点对重要邻居的贡献。返回各节点值。"""
    adj, nodes = _edges(edges)
    n = len(nodes)
    if n <= 1:
        return {"text": "至少需要 2 个节点。"}
    A, idx, _ = _adj_matrix(adj, nodes)
    x = np.ones(n) / np.sqrt(n)
    for _ in range(max(1, _n(iters, 100))):
        y = A @ x
        norm = np.linalg.norm(y)
        if norm < 1e-12:
            break
        x = y / norm
    return {"text": "特征向量中心性\n%s" % ", ".join(f"{nd}: {x[idx[nd]]:.4f}" for nd in nodes)}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "damping": {"type": "number"}, "iters": {"type": "number"}}, "required": []}, category="网络分析")
def pagerank(edges="1-2,2-3,3-4,4-1,2-4", damping=0.85, iters=50):
    """PageRank：阻尼系数 + 随机跳转的幂迭代。返回各节点权重（归一化）。"""
    adj, nodes = _edges(edges)
    n = len(nodes)
    if n == 0:
        return {"text": "空网络。"}
    d = min(1.0, max(0.0, _f(damping, 0.85)))
    idx = {nd: i for i, nd in enumerate(nodes)}
    P = np.zeros((n, n), dtype=float)
    for a, neigh in adj.items():
        i = idx[a]
        if neigh:
            for b in neigh:
                P[i, idx[b]] = 1.0 / len(neigh)
    pr = np.ones(n) / n
    for _ in range(max(1, _n(iters, 50))):
        dangling = pr[P.sum(axis=1) == 0].sum()
        nxt = d * (P.T @ pr) + (1 - d) / n + d * dangling / n
        nxt = nxt / nxt.sum()
        pr = nxt
    return {"text": "PageRank\n%s" % ", ".join(f"{nd}: {pr[idx[nd]]:.4f}" for nd in nodes)}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}}, "required": []}, category="网络分析")
def community_detect(edges="1-2,2-3,3-4,4-1,2-4"):
    """社区发现（Girvan-Newman）：反复移除边介数最大的边，保留模块度最高的划分。返回各社区节点。"""
    adj0, nodes = _edges(edges)
    n = len(nodes)
    if n < 2:
        return {"text": "至少需要 2 个节点。"}
    m = sum(len(v) for v in adj0.values()) // 2
    if m == 0:
        return {"text": "网络无边。"}

    def modularity(comms):
        k = {nd: len(adj0.get(nd, set())) for nd in nodes}
        q = 0.0
        for comm in comms:
            comm = set(comm)
            lc = 0
            dc = 0
            for a in comm:
                dc += k[a]
                for b in adj0.get(a, set()):
                    if b in comm:
                        lc += 1
            lc //= 2
            q += (lc / m) - (dc / (2 * m)) ** 2
        return q

    working = {nd: set(adj0.get(nd, set())) for nd in nodes}
    best_comms, best_q = None, -1.0
    while True:
        comps = [c for c in _components(working, nodes) if len(c) > 1]
        if not comps:
            break
        q = modularity(comps)
        if q > best_q:
            best_q, best_comms = q, [list(c) for c in comps]
        bet = _edge_betweenness(working, nodes)
        if not bet:
            break
        maxv = max(bet.values())
        if maxv <= 0:
            break
        for e in [e for e, v in bet.items() if v == maxv]:
            a, b = tuple(e)
            working.get(a, set()).discard(b)
            working.get(b, set()).discard(a)
        if sum(len(v) for v in working.values()) == 0:
            break

    if best_comms is None:
        return {"text": "未能划分社区。"}
    lines = ["社区 %d: %s" % (i + 1, ", ".join(c)) for i, c in enumerate(best_comms)]
    return {"text": "社区发现（模块度 Q=%.4f）\n%s" % (best_q, "\n".join(lines))}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}}, "required": []}, category="网络分析")
def link_prediction(edges="1-2,2-3,3-4,4-1,2-4"):
    """链路预测：按共同邻居 / 雅卡尔系数，推荐最可能相连的节点对。返回 Top5 候选。"""
    adj, nodes = _edges(edges)
    if len(nodes) < 2:
        return {"text": "至少需要 2 个节点。"}
    pairs = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if b in adj.get(a, set()):
                continue
            na, nb = adj.get(a, set()), adj.get(b, set())
            union = na | nb
            common = len(na & nb)
            jac = (common / len(union)) if union else 0.0
            pairs.append((a, b, common, jac))
    if not pairs:
        return {"text": "没有可推荐的节点对（网络已是完全图）。"}
    pairs.sort(key=lambda x: (-x[2], -x[3]))
    top = pairs[:5]
    lines = [f"{a} — {b}（共同邻居={c}, 雅卡尔={j:.3f}）" for a, b, c, j in top]
    return {"text": "链路预测 Top%d\n%s" % (len(top), "\n".join(lines))}


class NetworkPage(ui.BasePage):
    NAME = "网络分析"
    EMOJI = "\U0001F578\U0000FE0F"  # 🕸️

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🕸️ 网络分析", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="中心性 / PageRank / 社区发现 / 链路预测（边表格式：1-2,2-3,…）",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.ta = ctk.CTkTextbox(body, height=70, font=ui.mono(ui.FONT["body"]))
        self.ta.pack(fill="x", pady=(ui.SPACE["md"], 0))
        self.ta.insert("1.0", "1-2,2-3,3-4,4-1,2-4")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        for text, cmd in (("度中心性", self.do_deg), ("特征向量中心性", self.do_eig), ("PageRank", self.do_pr),
                          ("社区发现", self.do_comm), ("链路预测", self.do_link)):
            ctk.CTkButton(body, text=text, height=h, command=cmd).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ui.mono(ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def _e(self):
        return self.ta.get("1.0", "end").strip()

    def _run(self, fn, *args):
        try:
            self.msg(fn(*args)["text"])
        except Exception as e:
            self.msg(f"出错：{e}")

    def do_deg(self):
        self._run(degree_centrality, self._e())

    def do_eig(self):
        self._run(eigenvector_centrality, self._e())

    def do_pr(self):
        self._run(pagerank, self._e())

    def do_comm(self):
        self._run(community_detect, self._e())

    def do_link(self):
        self._run(link_prediction, self._e())


PAGES = [NetworkPage]
