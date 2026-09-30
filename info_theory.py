# -*- coding: utf-8 -*-
"""信息论与编码：香农熵、归一化熵、互信息、Huffman 编码、汉明码(7,4)编解码、二元对称信道容量。
AI 工具 + 页面（交叉领域：把概率论用到信息/编码）。依赖 numpy（可选 scipy/无额外）。"""
import heapq
import math
import numpy as np
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

PLUGIN = {
    "id": "info_theory",
    "name": "信息论与编码",
    "version": "1.0",
    "author": "zoilzo",
    "description": "信息论与编码：香农熵/归一化熵、互信息、Huffman 无损压缩、Hamming(7,4)纠错码、二元对称信道容量（工具 + 页面）",
}


# ------------------------------------------------------------
# 通用辅助（纯计算，无 GUI）
# ------------------------------------------------------------
def _num(v, default):
    """稳健数值解析：容忍空串、中文逗号、空格。解析失败回退默认值。"""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("，", ",").replace(" ", "").replace("%", "")
    if not s or s in ("-", "无", "空"):
        return default
    try:
        return float(s)
    except Exception:
        return default


def _bits(v):
    """解析比特串：接受 '1010' 或 '1,0,1,0'；返回 0/1 整数列表。"""
    s = str(v).strip().replace("，", ",").replace(" ", "").replace(",", "")
    if not s:
        return []
    out = [1 if ch in "1xX" else (0 if ch in "0oO" else None) for ch in s]
    return [b for b in out if b is not None]


def _freq(text):
    """统计字符频次 dict；未给文本用默认演示 'hello world'。"""
    s = str(text) if str(text).strip() else "hello world"
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    return counts


# ------------------------------------------------------------
# 工具 1：香农熵
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "text": {"type": "string", "description": "文本/符号序列（统计各符号频次）；留空用默认演示"},
    },
}, category="信息论")
def shannon_entropy(text=""):
    """香农熵 H = -Σ p·log2(p)：度量消息的不确定性/含信息量，单位比特/符号。分布越均匀熵越大。"""
    counts = _freq(text)
    n = sum(counts.values())
    if n <= 0 or not counts:
        return {"text": "没有可统计的符号。"}
    H = 0.0
    for c in counts.values():
        p = c / n
        H -= p * math.log2(p)
    K = len(counts)
    hmax = math.log2(K) if K > 1 else 0.0
    lines = ["香农熵（按符号频次估计）：", "",
             "符号总数 = %d，符号种类 = %d" % (n, K), "",
             "H = %.4f 比特/符号" % H,
             "最大熵 Hmax = %.4f（均匀分布）" % hmax, "",
             "相对熵 H/Hmax = %.3f" % (H / hmax if hmax > 0 else 0.0)]
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:8]
    lines.append("主要符号概率：" + ("  ".join("%s:%.3f" % (k, c / n) for k, c in top)))
    if hmax <= 0:
        pass
    elif H / hmax >= 0.90:
        lines.append("➤ 接近均匀分布 → 信息量大、冗余低，适合直接高效存储。")
    else:
        lines.append("➤ 分布不均 → 存在冗余，可用 Huffman/算术编码压缩。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 2：互信息
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "x": {"type": "string", "description": "变量 X 观测序列（逗号分隔，类别/数值均可）；留空用默认演示"},
        "y": {"type": "string", "description": "变量 Y 观测序列（与 X 等长、逗号分隔）"},
    },
}, category="信息论")
def mutual_information(x="", y=""):
    """互信息 I(X;Y) = ΣΣ p(x,y)·log2( p(x,y)/(p(x)p(y)) )：度量两变量共享的信息量，单位比特。
    值越接近 0 越独立，越大关联越强；可用于特征筛选、相关性与独立性检验。"""
    if not str(x).strip() or not str(y).strip():
        xs = list("aababbabba")
        ys = list("1101001001")   # 与 X 完全耦合（a→1, b→0）
        notch = "（默认演示：X∈{a,b}，Y∈{0,1}，Y 完全由 X 决定）"
    else:
        xs = [t for t in str(x).replace("，", ",").split(",") if t.strip()]
        ys = [t for t in str(y).replace("，", ",").split(",") if t.strip()]
        notch = ""
    if len(xs) != len(ys) or not xs:
        raise ValueError("X 与 Y 需等长且非空。")
    n = len(xs)
    joint = {}
    for a, b in zip(xs, ys):
        joint[(a, b)] = joint.get((a, b), 0) + 1
    px, py = {}, {}
    for (a, b), c in joint.items():
        px[a] = px.get(a, 0) + c
        py[b] = py.get(b, 0) + c
    mi = 0.0
    for (a, b), c in joint.items():
        pxy = c / n
        p_x = px[a] / n
        p_y = py[b] / n
        if pxy > 0 and p_x > 0 and p_y > 0:
            mi += pxy * math.log2(pxy / (p_x * p_y))
    hx = -sum(v / n * math.log2(v / n) for v in px.values()) if px else 0.0
    hy = -sum(v / n * math.log2(v / n) for v in py.values()) if py else 0.0
    nmimax = min(hx, hy) if min(hx, hy) > 0 else 0.0
    nmi = mi / nmimax if nmimax > 0 else 0.0
    lines = ["互信息 I(X;Y)%s：" % notch, "",
             "样本量 = %d，X 取值 %d 种，Y 取值 %d 种" % (n, len(px), len(py)), "",
             "I(X;Y) = %.4f 比特" % mi,
             "H(X) = %.4f，H(Y) = %.4f 比特" % (hx, hy), "",
             "归一化互信息 NMI = %.3f（0~1，越大越相关）" % nmi]
    if nmi < 0.10:
        lines.append("➤ 近 0 → X 与 Y 接近独立，彼此几乎不含对方信息。")
    elif nmi < 0.50:
        lines.append("➤ 中等关联 → X 与 Y 有一定信息重叠。")
    else:
        lines.append("➤ 强关联 → 从一变量能较好地预测另一变量。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 3：Huffman 编码
# ------------------------------------------------------------
def _huff_assign(node, prefix, code_map):
    if isinstance(node, str):
        code_map[node] = prefix
    else:
        left, right = node
        _huff_assign(left, prefix + "0", code_map)
        _huff_assign(right, prefix + "1", code_map)


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "text": {"type": "string", "description": "文本（统计各字符频次）；留空用默认演示 'hello world'"},
    },
}, category="信息论")
def huffman_codes(text=""):
    """Huffman 编码：按频次为符号生成前缀码，使加权平均码长最短，是经典无损压缩。给出编码表与压缩效率。"""
    counts = _freq(text)
    if not counts:
        return {"text": "没有可编码的符号。"}
    K = len(counts)
    if K <= 1:
        return {"text": "只有 %d 种符号，无需编码（平均码长 0）。" % K}
    heap = [(c, i, ch) for i, (ch, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    code_map = {ch: "" for ch in counts}
    while len(heap) > 1:
        c1, i1, n1 = heapq.heappop(heap)
        c2, i2, n2 = heapq.heappop(heap)
        _huff_assign(n1, "0", code_map)
        _huff_assign(n2, "1", code_map)
        heapq.heappush(heap, (c1 + c2, i1 + i2, (n1, n2)))
    n = sum(counts.values())
    avg = sum(len(code_map.get(ch, "")) * c for ch, c in counts.items()) / n
    fixed = int(math.ceil(math.log2(K)))  # 定长码位数
    lines = ["Huffman 编码：", "",
             "符号种类 = %d，总频次 = %d" % (K, n), "",
             "加权平均码长 = %.3f 比特/符号（定长需 %d 比特）" % (avg, fixed), "",
             "编码表：" + "  ".join('%s:"%s"' % (ch, code_map[ch]) for ch in sorted(counts, key=lambda c: -counts[c]))]
    savings = (1 - avg / fixed) * 100 if fixed > 0 else 0.0
    lines.append("压缩率 = %.1f%%" % savings)
    lines.append("➤ Huffman 码长接近 Shannon 熵下限；频次高者码更短，因此平均码长最短。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 工具 4：Hamming(7,4) 纠错码
# ------------------------------------------------------------
def _huff_check(cw):
    """校验 7 位码字是否有错（偶校验三个位组）。cw: 0/1 列表，位序 [p1,p2,d0,p4,d1,d2,d3]。"""
    if len(cw) != 7:
        return False
    p1, p2, d0, p4, d1, d2, d3 = cw
    return (p1 ^ d0 ^ d1 ^ d3) == 0 and (p2 ^ d0 ^ d2 ^ d3) == 0 and (p4 ^ d1 ^ d2 ^ d3) == 0


@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "data": {"type": "string", "description": "4 位数据（编码模式）或 7 位码字（解码模式），如 1010"},
        "mode": {"type": "string", "description": "encode 编码 / decode 解码（默认 encode）"},
    },
}, category="信息论")
def hamming_code(data="", mode="encode"):
    """Hamming(7,4) 线性纠错码：encode 把 4 位数据编为 7 位码字（含 3 位校验）；decode 对 7 位码字做单错纠正并还原 4 位数据。
    最小码距为 3，可纠正任意 1 位错误，用于存储/传输检纠错。"""
    mode = str(mode or "encode").strip().lower()
    if mode.startswith("de"):
        # 解码
        bits = _bits(data) if str(data).strip() else [1, 0, 1, 1, 0, 1, 0]
        if len(bits) != 7:
            raise ValueError("解码需 7 位码字（例如 1011010）。")
        if _huff_check(bits):
            d0, d1, d2, d3 = bits[2], bits[4], bits[5], bits[6]
            return {"text": "Hamming 解码：\n\n接收码字 = %s（有效，无错误）\n还原数据 = %s" %
                             ("".join(map(str, bits)), "".join(map(str, [d0, d1, d2, d3])))}
        fixed = None
        for i in range(7):
            trial = list(bits)
            trial[i] ^= 1
            if _huff_check(trial):
                fixed = i
                bits = trial
                break
        if fixed is None:
            return {"text": "Hamming 解码失败：接收码字有 2 位及以上错误，超出单纠错能力。\n接收码字 = %s" % "".join(map(str, bits))}
        d0, d1, d2, d3 = bits[2], bits[4], bits[5], bits[6]
        return {"text": "Hamming 解码（检测并纠正 1 位错误）：\n\n接收码字 = %s\n纠正位序 = 第 %d 位\n纠正后码字 = %s\n还原数据 = %s" %
                        ("".join(map(str, bits)), fixed + 1,
                         "".join(map(str, bits)), "".join(map(str, [d0, d1, d2, d3])))}
    # 编码
    bits = _bits(data) if str(data).strip() else [1, 0, 1, 0]
    if len(bits) != 4:
        raise ValueError("编码需 4 位数据（例如 1010）。")
    b0, b1, b2, b3 = bits
    p1 = b0 ^ b1 ^ b3
    p2 = b0 ^ b2 ^ b3
    p4 = b1 ^ b2 ^ b3
    cw = [p1, p2, b0, p4, b1, b2, b3]
    return {"text": "Hamming(7,4) 编码：\n\n数据 = %s\n码字 = %s（校验位 p1=%d p2=%d p4=%d）" %
                    ("".join(map(str, bits)), "".join(map(str, cw)), p1, p2, p4)}


# ------------------------------------------------------------
# 工具 5：二元对称信道容量
# ------------------------------------------------------------
@ai_tools._reg
@ai_tools._tool({
    "properties": {
        "p": {"type": "number", "description": "二进制对称信道错误概率（0~1，默认 0.1）"},
    },
}, category="信息论")
def channel_capacity(p=0.1):
    """二元对称信道(BSC)容量 C = 1 - H_b(p)，其中 H_b 为二元熵函数、p 为每比特翻转错误概率。单位：比特/信道使用。
    p=0 或 1 时容量最大(1)，p=0.5 时容量为 0（随机噪声无信息可传）。"""
    pv = _num(p, 0.1)
    pv = min(max(pv, 0.0), 1.0)
    if pv in (0.0, 1.0):
        cap = 1.0
    elif pv == 0.5:
        cap = 0.0
    else:
        hb = -pv * math.log2(pv) - (1 - pv) * math.log2(1 - pv)
        cap = 1.0 - hb
    hb = -pv * math.log2(pv) - (1 - pv) * math.log2(1 - pv) if 0 < pv < 1 else 0.0
    lines = ["二元对称信道(BSC)容量：", "",
             "错误率 p = %.4f" % pv,
             "二元熵 H_b(p) = %.4f 比特" % hb, "",
             "信道容量 C = %.4f 比特/信道使用" % cap, ""]
    if pv == 0.5:
        lines.append("➤ p=0.5 → 完全随机，容量为 0，无法可靠传任何信息。")
    else:
        lines.append("➤ C 是纠错编码可达到的上限（香农极限）；p 越小可可靠传输的速率越高。")
    return {"text": "\n".join(lines)}


# ------------------------------------------------------------
# 页面
# ------------------------------------------------------------
class InfoFormPage(ui.BasePage):
    _TITLE = "信息论与编码"
    _TOOLS = {}

    def __init__(self, master):
        super().__init__(master, layout=False)
        self._fields = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text=self._TITLE, font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(row, text="功能：", font=ctk.CTkFont(size=ui.FONT["body"])).grid(row=0, column=0, sticky="w", padx=(0, ui.SPACE["sm"]))
        self._names = [v[0] for v in self._TOOLS.values()]
        self._key_of = {v[0]: k for k, v in self._TOOLS.items()}
        self.mode_var = ctk.StringVar(value=self._names[0])
        ctk.CTkOptionMenu(row, values=self._names, variable=self.mode_var,
                          command=lambda _: self._rebuild()).grid(row=0, column=1, sticky="w")
        self.desc = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=ui.FONT["body"]), text_color="gray60",
                                 wraplength=560, justify="left")
        self.desc.pack(anchor="w", pady=(ui.SPACE["xs"], 0))
        self.field_frame = ctk.CTkFrame(body, fg_color="transparent")
        self.field_frame.pack(fill="x", pady=(ui.SPACE["sm"], 0))
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        ctk.CTkButton(body, text="计算", height=h, fg_color=ui.body(), command=self._run).pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self._rebuild()

    def _rebuild(self):
        for w in self.field_frame.winfo_children():
            w.destroy()
        self._fields = {}
        key = self._key_of[self.mode_var.get()]
        label, desc, fields, func = self._TOOLS[key]
        self.desc.configure(text=desc)
        self._current = func
        for r, (fkey, flabel, fdefault) in enumerate(fields):
            ctk.CTkLabel(self.field_frame, text=flabel, font=ctk.CTkFont(size=ui.FONT["body"])).grid(
                row=r, column=0, sticky="w", pady=(0, ui.SPACE["xs"]))
            e = ctk.CTkEntry(self.field_frame)
            e.insert(0, str(fdefault))
            e.grid(row=r, column=1, sticky="ew", pady=(0, ui.SPACE["xs"]))
            self.field_frame.grid_columnconfigure(1, weight=1)
            self._fields[fkey] = e

    def _run(self):
        vals = {k: e.get().strip() for k, e in self._fields.items()}
        out = self.out
        out.configure(state="normal")
        out.delete("1.0", "end")
        out.configure(state="disabled")
        try:
            res = self._current(**vals)
            text = res.get("text", str(res)) if isinstance(res, dict) else str(res)
        except Exception as ex:
            text = "计算出错：%s" % ex
        out.configure(state="normal")
        out.insert("1.0", text)
        out.configure(state="disabled")


class InfoTheoryPage(InfoFormPage):
    NAME = "信息论与编码"
    EMOJI = "📡"
    _TITLE = "📡 信息论与编码"
    _TOOLS = {
        "shannon_entropy": (
            "香农熵",
            "度量一段符号序列的不确定性/信息量；分布越均匀熵越大。",
            [("text", "文本(留空=默认演示 hello world)", "")],
            shannon_entropy),
        "mutual_information": (
            "互信息",
            "度量两变量的共享信息量(0~1)；用于特征筛选与独立性分析。",
            [("x", "变量X序列(逗号分隔，留空=默认演示)", ""),
             ("y", "变量Y序列(逗号分隔)", "")],
            mutual_information),
        "huffman_codes": (
            "Huffman 编码",
            "按频次生成最短前缀码的无损压缩，给出编码表与压缩率。",
            [("text", "文本(留空=默认演示 hello world)", "")],
            huffman_codes),
        "hamming_code": (
            "Hamming(7,4) 纠错",
            "4 位数据编码成 7 位码字；或解码 7 位码字并纠正单比特错误。",
            [("data", "4位数据(encode)或7位码字(decode)", "1010"),
             ("mode", "encode / decode", "encode")],
            hamming_code),
        "channel_capacity": (
            "信道容量",
            "二元对称信道容量 C = 1 - H_b(p)，纠错编码可达的速率上限。",
            [("p", "错误概率 p(0~1)", "0.1")],
            channel_capacity),
    }


PAGES = [InfoTheoryPage]