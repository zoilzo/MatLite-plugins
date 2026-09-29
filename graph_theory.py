# -*- coding: utf-8 -*-
"""图论：BFS/DFS、Dijkstra、Floyd-Warshall、Prim/Kruskal、拓扑排序、二分图检测、欧拉路径。
AI 工具 + 页面（教学，展示算法分步结果）。"""
import heapq
from collections import deque
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "graph_theory",
    "name": "图论",
    "version": "1.0",
    "author": "zoilzo",
    "description": "图的 BFS/DFS、最短路（Dijkstra/Floyd）、最小生成树（Prim/Kruskal）、拓扑排序等（工具 + 页面）",
}

# 默认示例图：A-B:4, B-C:3, A-C:8, C-D:2, B-D:5, A-E:9
_DEFAULT_EDGES = "A-B:4,B-C:3,A-C:8,C-D:2,B-D:5,A-E:9,E-D:6"


def _parse_edges(edges):
    """把边字符串（"A-B:4,B-C,.."）解析为：
    (adj, nodes, edge_list)，其中 adj={u:{v:weight(默认1)}}。"""
    adj = {}
    nodes = set()
    elist = []
    if not edges:
        edges = _DEFAULT_EDGES
    if isinstance(edges, (list, tuple)):
        items = [str(e) for e in edges]
    else:
        items = [x.strip() for x in str(edges).replace("，", ",").split(",") if x.strip()]
    for it in items:
        if ":" in it:
            uv, w = it.split(":", 1)
            try:
                w = float(w)
            except ValueError:
                w = 1.0
        else:
            uv, w = it, 1.0
        if "-" in uv:
            u, v = uv.split("-", 1)
        elif ">" in uv:
            u, v = uv.split(">", 1)
        else:
            continue
        u, v = u.strip(), v.strip()
        adj.setdefault(u, {})[v] = w
        adj.setdefault(v, {}).setdefault(u, w)  # 无向默认，有向时由函数覆盖
        nodes.add(u)
        nodes.add(v)
        elist.append((u, v, w))
    return adj, sorted(nodes), elist


def _resolve_start(start, nodes):
    if start and start in nodes:
        return start
    return nodes[0] if nodes else None


# ---------------- 遍历 ----------------
def _bfs(adj, start):
    seen = [start]
    q = deque([start])
    visited = {start}
    order = [start]
    while q:
        u = q.popleft()
        for v in sorted(adj.get(u, {})):
            if v not in visited:
                visited.add(v)
                q.append(v)
                order.append(v)
    return order


def _dfs(adj, start):
    visited = {start}
    order = []
    st = [start]
    while st:
        u = st.pop()
        if u not in visited:
            visited.add(u)
        order.append(u)
        for v in sorted(adj.get(u, {}), reverse=True):
            if v not in visited:
                st.append(v)
    return order


# ---------------- 最短路 ----------------
def _dijkstra(adj, start):
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def _floyd(adj, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    nn = len(nodes)
    INF = float("inf")
    d = [[INF] * nn for _ in range(nn)]
    for i in range(nn):
        d[i][i] = 0.0
    nxt = [[None] * nn for _ in range(nn)]
    for u, nbrs in adj.items():
        for v, w in nbrs.items():
            i, j = idx[u], idx[v]
            d[i][j] = w
            nxt[i][j] = j
    for k in range(nn):
        for i in range(nn):
            for j in range(nn):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
                    nxt[i][j] = nxt[i][k]
    return d, nxt, idx


# ---------------- 最小生成树 ----------------
def _prim(adj, nodes, start):
    start = _resolve_start(start, nodes)
    in_tree = {start}
    mst = []
    edges = []
    for v, w in adj.get(start, {}).items():
        heapq.heappush(edges, (w, start, v))
    total = 0.0
    while edges:
        w, u, v = heapq.heappop(edges)
        if v in in_tree:
            continue
        in_tree.add(v)
        mst.append((u, v, w))
        total += w
        for nxt, nw in adj.get(v, {}).items():
            if nxt not in in_tree:
                heapq.heappush(edges, (nw, v, nxt))
    return mst, total


def _kruskal(nodes, elist):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst = []
    total = 0.0
    for u, v, w in sorted(elist, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            mst.append((u, v, w))
            total += w
    return mst, total


# ---------------- 拓扑排序 ----------------
def _toposort(nodes, elist):
    indeg = {n: 0 for n in nodes}
    outs = {n: [] for n in nodes}
    for u, v, _w in elist:
        indeg[v] = indeg.get(v, 0) + 1
        outs.setdefault(u, []).append(v)
    q = deque(sorted([n for n in nodes if indeg.get(n, 0) == 0]))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in outs.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order, len(order) == len(nodes)


# ---------------- 二分图检测 ----------------
def _is_bipartite(adj, nodes):
    color = {}
    ok = True
    for s in nodes:
        if s in color or not adj.get(s):
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj.get(u, {}):
                if v not in color:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    ok = False
        if not ok:
            break
    return ok, color


# ---------------- 欧拉路径 ----------------
def _euler_path(nodes, elist, directed=False):
    # 用 Fleury/逐步法：按邻接删除边做层次遍历
    adj = {n: [] for n in nodes}
    for u, v, _w in elist:
        adj.setdefault(u, []).append(v)
        if not directed:
            adj.setdefault(v, []).append(u)
    # 奇度点
    odd = [n for n in nodes if len(adj[n]) % 2 == 1]
    if odd and len(odd) != 2:
        return None, odd, "奇度点有 %d 个（≠0 且 ≠2），不存在欧拉路径/回路" % len(odd)
    # 连通性（忽略 0 度点）
    start = None
    visited = set()
    for n in nodes:
        if adj[n]:
            start = n
            break
    if start is None:
        return [], odd, "无边的图（平凡欧拉回路）"
    comp = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in comp:
                comp.add(v)
                q.append(v)
    active = {n for n in nodes if adj[n]}
    if not comp.issuperset(active):
        return None, odd, "图不连通，无欧拉路径"
    # Hierholzer 找欧拉回路/路径
    local = {n: list(adj[n]) for n in nodes}
    if odd:
        start = max(odd, key=lambda n: len(local[n]))
    st = [start]
    path = []
    while st:
        u = st[-1]
        if local[u]:
            v = local[u].pop()
            st.append(v)
        else:
            path.append(st.pop())
    path.reverse()
    return path, odd, "欧拉路径/回路存在"


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "start": {"type": "string"}, "mode": {"type": "string"}}}, category="图论")
def graph_traverse(edges="", start="", mode="BFS"):
    """图的广度/深度优先遍历。edges 形如 'A-B:4,B-C'，mode: BFS/DFS。"""
    adj, nodes, _e = _parse_edges(edges)
    s = _resolve_start(start, nodes)
    if not s:
        return {"text": "没有可遍历的节点。"}
    if (mode or "BFS").upper() == "DFS":
        order = _dfs(adj, s)
        tag = "DFS（深度优先）"
    else:
        order = _bfs(adj, s)
        tag = "BFS（广度优先）"
    return {"text": f"图 {tag} 从 {s} 出发：\n遍历顺序 → {' → '.join(order)}\n（共 {len(order)} 个结点可达）"}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "start": {"type": "string"}, "method": {"type": "string"}}}, category="图论")
def graph_shortest(edges="", start="", method="dijkstra"):
    """单源最短路（Dijkstra）或全源最短路（Floyd-Warshall）。method: dijkstra/floyd。"""
    adj, nodes, _e = _parse_edges(edges)
    if (method or "dijkstra")[:1].lower() == "f":
        d, _nxt, idx = _floyd(adj, nodes)
        lines = ["Floyd-Warshall 全源最短路（∞ 表示不可达）："]
        hdr = "      " + "".join("%8s" % n for n in nodes)
        lines.append(hdr)
        for u in nodes:
            row = "%6s" % u
            i = idx[u]
            for v in nodes:
                j = idx[v]
                val = d[i][j]
                row += "%8s" % ("∞" if val == float("inf") else ("%g" % val))
            lines.append(row)
        return {"text": "\n".join(lines)}
    s = _resolve_start(start, nodes)
    if not s:
        return {"text": "没有节点。"}
    dist, prev = _dijkstra(adj, s)
    lines = [f"Dijkstra 从 {s} 出发的最短路："]
    for n in nodes:
        dv = dist.get(n, float("inf"))
        p = prev.get(n)
        if dv == float("inf"):
            lines.append(f"  {s}→{n}: 不可达")
        else:
            lines.append(f"  {s}→{n}: 距离 {dv:g}，前驱 {p}")
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "start": {"type": "string"}, "method": {"type": "string"}}}, category="图论")
def graph_mst(edges="", start="", method="prim"):
    """最小生成树。method: prim/kruskal。返回选边与总权重。"""
    adj, nodes, elist = _parse_edges(edges)
    if (method or "prim")[:1].lower() == "k":
        mst, total = _kruskal(nodes, elist)
        tag = "Kruskal（按边权升序+并查集）"
    else:
        mst, total = _prim(adj, nodes, start)
        tag = "Prim（从起点逐步扩展）"
    lines = [f"最小生成树（{tag}）："]
    for u, v, w in mst:
        lines.append(f"  选边 {u}–{v}  权重 {w:g}")
    lines.append(f"  总权重 = {total:g}")
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"edges": {"type": "string"}, "start": {"type": "string"}}}, category="图论")
def graph_theory_check(edges="", start=""):
    """综合检查：连通分量、拓扑排序、二分图检测、欧拉路径。edges 形如 'A-B,B-C'。"""
    adj, nodes, elist = _parse_edges(edges)
    lines = []
    # 拓扑排序（用有向边）
    order, acyclic = _toposort(nodes, elist)
    if acyclic and len(order) == len(nodes):
        lines.append("拓扑排序（DAG）：" + " → ".join(order))
    else:
        lines.append(f"拓扑排序：存在环（不能得到全序），已排出前序 {len(order)} 个。 环可能导致无解。")
    # 二分图
    bo, color = _is_bipartite(adj, nodes)
    lines.append("二分图检测：" + ("是（可二染色）" if bo else "否（含奇环，无法二染色）"))
    if bo and color:
        g0 = [n for n, c in color.items() if c == 0]
        g1 = [n for n, c in color.items() if c == 1]
        lines.append("  染色分组 A：%s" % (", ".join(g0) or "-"))
        lines.append("  染色分组 B：%s" % (", ".join(g1) or "-"))
    # 欧拉路径
    path, odd, note = _euler_path(nodes, elist)
    if path is not None:
        lines.append(f"欧拉路径：{' → '.join(path)}")
        if odd:
            lines.append("  起点/终点为两个奇度点：%s" % ", ".join(odd))
        else:
            lines.append("  所有点偶度，存在欧拉回路。")
    else:
        lines.append("欧拉路径：不存在。 " + note)
    # 连通分量（并查集，无向）
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v, _w in elist:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    comps = {}
    for n in nodes:
        comps.setdefault(find(n), []).append(n)
    lines.append("连通分量（无向）：%d 个" % len(comps))
    for grp in comps.values():
        lines.append("  " + ", ".join(grp))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class GraphTheoryPage(ui.BasePage):
    NAME = "图论"
    EMOJI = "\U0001F517"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🔗 图论", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="边格式：A-B:B，逗号分隔，如 A-B:4,B-C:3。:后可省略（默认权重 1）。",
                     font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(0, ui.SPACE["xs"]))
        self.edges_entry = ctk.CTkEntry(body)
        self.edges_entry.insert(0, _DEFAULT_EDGES)
        self.edges_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="起点：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w", padx=(0, ui.SPACE["xs"]))
        self.start_var = ctk.StringVar(value="A")
        ctk.CTkEntry(row, textvariable=self.start_var, width=96).grid(row=0, column=1, sticky="w")
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (txt, cmd) in enumerate([
            ("遍历(BFS)", lambda: self.run("traverse")),
            ("最短路", lambda: self.run("shortest")),
            ("最小生成树", lambda: self.run("mst")),
            ("全面检查", lambda: self.run("check")),
            ("全源最短路", lambda: self.run("floyd")),
        ]):
            btns.grid_columnconfigure(i % 2, weight=1)
            r, c = divmod(i, 2)
            ctk.CTkButton(btns, text=txt, height=h, command=cmd).grid(row=r, column=c, sticky="ew", padx=2, pady=2)
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入边与起点，点上方按钮计算；「全面检查」会做拓扑/二分/欧拉/连通分量。")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def run(self, which):
        edges = self.edges_entry.get()
        start = self.start_var.get()
        try:
            if which == "traverse":
                r = graph_traverse(edges, start, "BFS")
                r2 = graph_traverse(edges, start, "DFS")
                r["text"] += "\n\n" + r2["text"]
            elif which == "shortest":
                r = graph_shortest(edges, start, "dijkstra")
            elif which == "floyd":
                r = graph_shortest(edges, start, "floyd")
            elif which == "mst":
                r = graph_mst(edges, start, "prim")
                r2 = graph_mst(edges, start, "kruskal")
                r["text"] += "\n\n" + r2["text"]
            else:
                r = graph_theory_check(edges, start)
        except Exception as e:
            self.msg(f"出错：{e}")
            return
        self.msg(r.get("text", str(r)))


PAGES = [GraphTheoryPage]