# -*- coding: utf-8 -*-
"""密码学入门：Caesar/Vigenère/affine/Atbash 加解密、模幂、RSA 演示、素性测试与欧拉函数。
AI 工具 + 页面（教学，确定性演示，不用真实随机密钥）。"""
import math
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools

try:
    import sympy as _sp
except Exception:  # pragma: no cover
    _sp = None

PLUGIN = {
    "id": "crypto_kit",
    "name": "密码学入门",
    "version": "1.0",
    "author": "zoilzo",
    "description": "经典密码（Caesar/Vigenère/affine/Atbash）、模幂、RSA 与素性/欧拉函数（工具 + 页面）",
}

_ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _norm(s):
    """仅保留英文字母并统一小写，返回 (clean, 掩码)，掩码记录原字符是否字母。"""
    mask = [c.isalpha() for c in s]
    return "".join(c.lower() for c in s if c.isalpha()), mask


def _restore(clean, mask):
    """用掩码把纯小写字母串还原成原格式（非字母位置原样跳过）。"""
    out = []
    it = iter(clean)
    for m in mask:
        out.append(next(it) if m else " ")
    return "".join(out)


# ---------------- 各密码核心 ----------------
def _caesar(text, shift, decode=False):
    clean, mask = _norm(text)
    sh = int(shift) % 26
    if decode:
        sh = -sh
    res = "".join(_ALPHA[(_ALPHA.index(ch) + sh) % 26] for ch in clean)
    return _restore(res, mask)


def _atbash(text, decode=False):
    clean, mask = _norm(text)
    res = "".join(_ALPHA[25 - _ALPHA.index(ch)] for ch in clean)
    return _restore(res, mask)


def _affine(text, a, b, decode=False):
    clean, mask = _norm(text)
    a = int(a) % 26
    b = int(b) % 26
    if math.gcd(a, 26) != 1:
        raise ValueError("a 与 26 不互质，无法解密 (需 gcd(a,26)=1)")
    if decode:
        inv = pow(a, -1, 26)
        res = "".join(_ALPHA[(inv * (_ALPHA.index(ch) - b)) % 26] for ch in clean)
    else:
        res = "".join(_ALPHA[(a * _ALPHA.index(ch) + b) % 26] for ch in clean)
    return _restore(res, mask)


def _vigenere(text, key, decode=False):
    clean, mask = _norm(text)
    key = "".join(c for c in key.lower() if c.isalpha()) or "a"
    res = []
    ki = 0
    for ch in clean:
        k = _ALPHA.index(key[ki % len(key)])
        idx = _ALPHA.index(ch)
        if decode:
            res.append(_ALPHA[(idx - k) % 26])
        else:
            res.append(_ALPHA[(idx + k) % 26])
        ki += 1
    return _restore("".join(res), mask)


# ---------------- 数论 ----------------
def _is_prime(n):
    n = int(n)
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _euler_phi(n):
    n = int(n)
    if n <= 0:
        return 0
    res = n
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            res -= res // p
        p += 1
    if m > 1:
        res -= res // m
    return res


# ============================================================
# AI 工具
# ============================================================
@ai_tools._reg
@ai_tools._tool({"properties": {"text": {"type": "string"}, "method": {"type": "string"}, "shift": {"type": "number"}, "key": {"type": "string"}, "a": {"type": "number"}, "b": {"type": "number"}}}, category="密码学")
def cipher_encode(text="hello world", method="caesar", shift=3, key="KEY", a=5, b=8):
    """经典密码加密。method: caesar/vigenere/affine/atbash。返回密文。"""
    meth = (method or "caesar").lower()
    if meth in ("caesar", "julius"):
        return {"text": "Caesar 加密（位移 {0}）：{1}".format(shift, _caesar(text, shift))}
    if meth == "vigenere":
        return {"text": "Vigenère 加密（密钥 {0}）：{1}".format(key, _vigenere(text, key))}
    if meth == "affine":
        try:
            return {"text": "Affine 加密（a={0}, b={1}）：{2}".format(a, b, _affine(text, a, b))}
        except ValueError as e:
            return {"text": str(e)}
    if meth in ("atbash", "at"):
        return {"text": "Atbash 加密：{0}".format(_atbash(text))}
    return {"text": "未知方法 {0}，可选 caesar/vigenere/affine/atbash".format(meth)}


@ai_tools._reg
@ai_tools._tool({"properties": {"text": {"type": "string"}, "method": {"type": "string"}, "shift": {"type": "number"}, "key": {"type": "string"}, "a": {"type": "number"}, "b": {"type": "number"}}}, category="密码学")
def cipher_decode(text="", method="caesar", shift=3, key="KEY", a=5, b=8):
    """经典密码解密（同函数加解密互逆）。text 为密文。返回明文。"""
    if not text:
        return {"text": "请传入密文 text，如 'khuur zruog'。"}
    meth = (method or "caesar").lower()
    if meth in ("caesar", "julius"):
        return {"text": "Caesar 解密（位移 -{0}）：{1}".format(shift, _caesar(text, shift, True))}
    if meth == "vigenere":
        return {"text": "Vigenère 解密（密钥 {0}）：{1}".format(key, _vigenere(text, key, True))}
    if meth == "affine":
        try:
            return {"text": "Affine 解密（a={0}, b={1}）：{2}".format(a, b, _affine(text, a, b, True))}
        except ValueError as e:
            return {"text": str(e)}
    if meth in ("atbash", "at"):
        return {"text": "Atbash 解密：{0}".format(_atbash(text, True))}
    return {"text": "未知方法 {0}，可选 caesar/vigenere/affine/atbash".format(meth)}


@ai_tools._reg
@ai_tools._tool({"properties": {"base": {"type": "number"}, "exp": {"type": "number"}, "mod": {"type": "number"}, "n": {"type": "number"}}}, category="密码学")
def crypto_number(base=7, exp=5, mod=13, n=0):
    """模幂运算（base^exp mod mod）；若 n>0 则同时给出素性测试与欧拉函数 φ(n)。"""
    lines = []
    if mod:
        val = pow(int(base), int(exp), int(mod))
        lines.append("模幂：{0}^{1} mod {2} = {3}".format(base, exp, mod, val))
        # 演示逐次平方
        bits = bin(int(exp))[2:]
        cur = int(base) % int(mod)
        squarings = []
        for i, bit in enumerate(bits):
            squarings.append("第{0}位(bit={1}): 值 {2}".format(i + 1, bit, cur))
            cur = (cur * cur) % int(mod)
        lines.append("  快速幂（平方-乘）中间值：")
        lines.extend("    " + s for s in squarings)
    if n:
        n = int(n)
        lines.append("素性测试：{0} 是{1}素数".format(n, " " if _is_prime(n) else " 合数（非"))
        lines.append("欧拉函数 φ({0}) = {1}".format(n, _euler_phi(n)))
    return {"text": "\n".join(lines)}


@ai_tools._reg
@ai_tools._tool({"properties": {"p": {"type": "number"}, "q": {"type": "number"}, "message": {"type": "string"}, "e": {"type": "number"}}}, category="密码学")
def rsa_demo(p=61, q=53, message=97, e=17):
    """RSA 演示：给定两个素数 p,q 与公钥指数 e，演示密钥生成、加密、解密（靠欧拉函数求私钥）。"""
    if _sp is None:
        return {"text": "未安装 sympy，无法演示 RSA 数论。"}
    p = int(p)
    q = int(q)
    if not (_is_prime(p) and _is_prime(q)):
        return {"text": "p 与 q 需为素数。"}
    n = p * q
    phi = (p - 1) * (q - 1)
    e = int(e)
    if math.gcd(e, phi) != 1:
        return {"text": "e 需与 φ(n)={0} 互质。".format(phi)}
    d = pow(e, -1, phi)
    m = int(message)
    if m >= n:
        return {"text": "明文 m 需小于 n={0}。".format(n)}
    c = pow(m, e, n)
    m2 = pow(c, d, n)
    lines = ["RSA 演示（公钥/私钥）："]
    lines.append("  选素数 p = {0}, q = {1}".format(p, q))
    lines.append("  n = p·q = {0}".format(n))
    lines.append("  φ(n) = (p-1)(q-1) = {0}".format(phi))
    lines.append("  公钥指数 e = {0}（与 φ 互质）".format(e))
    lines.append("  私钥 d = e⁻¹ mod φ = {0}".format(d))
    lines.append("  公钥 = (n, e) = ({0}, {1})".format(n, e))
    lines.append("  私钥 = (n, d) = ({0}, {1})".format(n, d))
    lines.append("  加密：c = m^e mod n = {0}^{1} mod {2} = {3}".format(m, e, n, c))
    lines.append("  解密：m = c^d mod n = {0}^{1} mod {2} = {3}".format(c, d, n, m2))
    lines.append("  结果：解密后 m = {0}（加密前 {1}）→ {2}".format(m2, m, "一致 ✔" if m2 == m else "不一致 ✘"))
    return {"text": "\n".join(lines)}


# ============================================================
# 页面
# ============================================================
class CryptoKitPage(ui.BasePage):
    NAME = "密码学入门"
    EMOJI = "\U0001F510"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="🔐 密码学入门", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="明文/密文（自动转为小写，仅处理英文字母）：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.text_entry = ctk.CTkEntry(body)
        self.text_entry.insert(0, "hello world")
        self.text_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        for i, (lbl, val) in enumerate([("方法", "caesar"), ("位移", "3"), ("密钥", "KEY"), ("a", "5"), ("b", "8")]):
            row.grid_columnconfigure(i * 2, weight=1)
            ctk.CTkLabel(row, text=lbl + ":", font=ctk.CTkFont(size=ui.FONT["caption"])).grid(row=0, column=i * 2, sticky="w", padx=(0, ui.SPACE["xs"]))
        self.method_var = ctk.StringVar(value="caesar")
        ctk.CTkOptionMenu(row, values=["caesar", "vigenere", "affine", "atbash"], variable=self.method_var, width=120).grid(row=0, column=1, padx=(0, ui.SPACE["sm"]))
        self.shift_var = ctk.StringVar(value="3")
        ctk.CTkEntry(row, textvariable=self.shift_var, width=48).grid(row=0, column=3, padx=(0, ui.SPACE["sm"]))
        self.key_var = ctk.StringVar(value="KEY")
        ctk.CTkEntry(row, textvariable=self.key_var, width=64).grid(row=0, column=5, padx=(0, ui.SPACE["sm"]))
        self.a_var = ctk.StringVar(value="5")
        ctk.CTkEntry(row, textvariable=self.a_var, width=44).grid(row=0, column=7, padx=(0, ui.SPACE["sm"]))
        self.b_var = ctk.StringVar(value="8")
        ctk.CTkEntry(row, textvariable=self.b_var, width=44).grid(row=0, column=9)
        h = ui.SPACE["lg"] + ui.SPACE["sm"]
        btns = ctk.CTkFrame(body, fg_color="transparent")
        btns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(btns, text="加密", height=h, command=lambda: self.run(True)).grid(row=0, column=0, sticky="ew", padx=(0, ui.SPACE["xs"]))
        ctk.CTkButton(btns, text="解密", height=h, fg_color="gray40", command=lambda: self.run(False)).grid(row=0, column=1, sticky="ew", padx=(ui.SPACE["xs"], 0))
        ctk.CTkLabel(body, text="数论演示（base^exp mod mod；n>0 时做素性/欧拉；RSA 用 p,q,e）：", font=ctk.CTkFont(size=ui.FONT["body"])).pack(anchor="w", pady=(ui.SPACE["sm"], 0))
        self.math_entry = ctk.CTkEntry(body)
        self.math_entry.insert(0, "7 5 13 0")
        self.math_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        self.rsa_entry = ctk.CTkEntry(body)
        self.rsa_entry.insert(0, "61 53 97 17")
        self.rsa_entry.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        rbtns = ctk.CTkFrame(body, fg_color="transparent")
        rbtns.pack(fill="x", pady=(0, ui.SPACE["sm"]))
        rbtns.grid_columnconfigure(0, weight=1)
        rbtns.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(rbtns, text="数论", height=h, command=lambda: self.run_math()).grid(row=0, column=0, sticky="ew", padx=(0, ui.SPACE["xs"]))
        ctk.CTkButton(rbtns, text="RSA 演示", height=h, fg_color="gray40", command=lambda: self.run_rsa()).grid(row=0, column=1, sticky="ew", padx=(ui.SPACE["xs"], 0))
        self.out = ctk.CTkTextbox(body, font=ctk.CTkFont(family="Consolas", size=ui.FONT["body"]))
        self.out.pack(fill="both", expand=True, pady=(0, ui.SPACE["xs"]))
        self.msg("输入明文，选方法后点加密；粘贴密文可点解密。数论/RSA 演示见下方。")

    def msg(self, s):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", str(s))
        self.out.configure(state="disabled")

    def run(self, enc):
        text = self.text_entry.get()
        meth = self.method_var.get()
        try:
            shift = int(self.shift_var.get() or 3)
            a = int(self.a_var.get() or 5)
            b = int(self.b_var.get() or 8)
        except ValueError:
            self.msg("位移/a/b 需为整数")
            return
        if enc:
            r = cipher_encode(text, meth, shift, self.key_var.get(), a, b)
        else:
            r = cipher_decode(text, meth, shift, self.key_var.get(), a, b)
        self.msg(r.get("text", str(r)))

    def run_math(self):
        parts = [p for p in self.math_entry.get().replace("，", " ").split() if p.strip()]
        try:
            base = float(parts[0]) if len(parts) > 0 else 7
            exp = float(parts[1]) if len(parts) > 1 else 5
            mod = float(parts[2]) if len(parts) > 2 else 13
            n = float(parts[3]) if len(parts) > 3 else 0
        except ValueError:
            self.msg("请输入数字：base exp mod n")
            return
        r = crypto_number(base, exp, mod, n)
        self.msg(r.get("text", str(r)))

    def run_rsa(self):
        parts = [p for p in self.rsa_entry.get().replace("，", " ").split() if p.strip()]
        try:
            p = int(parts[0]) if len(parts) > 0 else 61
            q = int(parts[1]) if len(parts) > 1 else 53
            m = int(parts[2]) if len(parts) > 2 else 97
            e = int(parts[3]) if len(parts) > 3 else 17
        except ValueError:
            self.msg("请输入 p q message e")
            return
        r = rsa_demo(p, q, m, e)
        self.msg(r.get("text", str(r)))


PAGES = [CryptoKitPage]
