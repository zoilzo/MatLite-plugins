# -*- coding: utf-8 -*-
"""高等线性代数：LU/QR/SVD/Cholesky 分解、Gram-Schmidt、零空间、条件数、最小二乘（工具 + 页面）。"""
import customtkinter as ctk
from modules import ui_kit as ui
from modules import ai_tools
import numpy as np

PLUGIN = {
    "id": "linear_algebra_pro",
    "name": "高等线性代数",
    "version": "1.0",
    "author": "zoilzo",
    "description": "矩阵分解/正交化/零空间/最小二乘（工具 + 页面，高等代数）",
}


def _mat(s):
    """把 1,2;3,4 解析成 numpy 矩阵（分号换行、逗号分隔）。"""
    s = s.strip()
    if s.startswith("["):
        import json
        return np.array(json.loads(s.replace("'", '"')), dtype=float)
    rows = [r.split() for r in s.replace(";", "\n").replace(",", " ").split("\n") if r.strip()]
    return np.array([[float(x) for x in r] for r in rows], dtype=float)


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "1,2;3,5"}}, "required": ["a"]}, category="线性代数")
def lu_decomp(a="1,2;3,5"):
    """LU 分解 A=P.L.U，验证重建。"""
    from scipy.linalg import lu
    A = _mat(a)
    P, L, U = lu(A)
    err = float(np.max(np.abs((P @ L @ U) - A)))
    return {"text": f"LU 分解重建误差={err:.2e}（约 0 即正确）。L=\n{np.round(L,4)}\nU=\n{np.round(U,4)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "1,2;0,1"}}, "required": ["a"]}, category="线性代数")
def qr_decomp(a="1,2;0,1"):
    """QR 分解 A=Q.R，Q 正交、R 上三角。"""
    A = _mat(a)
    Q, R = np.linalg.qr(A)
    orth = float(np.max(np.abs((Q.T @ Q) - np.eye(Q.shape[1]))))
    return {"text": f"QR 分解：Q 正交性={orth:.2e}（约 0 即正确）。Q=\n{np.round(Q,4)}\nR=\n{np.round(R,4)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "1,1;0,2"}}, "required": ["a"]}, category="线性代数")
def svd_decomp(a="1,1;0,2"):
    """SVD 分解 A=U.Sigma.V^T，给出奇异值与重建误差。"""
    A = _mat(a)
    U, s, Vt = np.linalg.svd(A)
    recon = (U[:, :len(s)] * s) @ Vt[:len(s)]
    err = float(np.max(np.abs(recon - A)))
    return {"text": f"SVD 分解：奇异值 sigma={np.round(s,4)}，重建误差={err:.2e}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "2,1,1;1,2,1;1,1,2"}}, "required": ["a"]}, category="线性代数")
def cholesky_decomp(a="2,1,1;1,2,1;1,1,2"):
    """Cholesky 分解 A=L.L^T（仅适用正定矩阵），验证重建。"""
    A = _mat(a)
    try:
        L = np.linalg.cholesky(A).T
    except Exception as e:
        return {"text": "矩阵不是正定的，无法 Cholesky 分解：" + str(e)}
    err = float(np.max(np.abs((L @ L.T) - A)))
    return {"text": f"Cholesky 分解重建误差={err:.2e}。L=\n{np.round(L,4)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"vecs": {"type": "string", "default": "1,1,0;1,0,1;0,1,1"}}, "required": ["vecs"]}, category="线性代数")
def gram_schmidt(vecs="1,1,0;1,0,1;0,1,1"):
    """对一组列向量做 Gram-Schmidt 正交化，返回正交规范基。"""
    V = _mat(vecs).T
    Q = []
    for v in V:
        w = v.astype(float)
        for q in Q:
            w = w - (w @ q) * q
        n = np.linalg.norm(w)
        if n > 1e-12:
            Q.append(w / n)
    return {"text": f"Gram-Schmidt 正交规范基（每行一个基向量）：\n{np.round(np.array(Q),4)}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "1,2,3;2,4,6"}}, "required": ["a"]}, category="线性代数")
def nullspace_basis(a="1,2,3;2,4,6"):
    """求零空间 Ax=0 的基（SVD 法），并给条件数。"""
    A = _mat(a)
    U, s, Vt = np.linalg.svd(A)
    tol = max(A.shape) * s[0] * 1e-12 if s.size else 0
    ns = Vt[s < tol].T
    cond = (s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return {"text": f"零空间基（每列一个基向量）：\n{np.round(ns,4)}\n条件数 kappa={cond:.2e}"}


@ai_tools._reg
@ai_tools._tool({"properties": {"a": {"type": "string", "default": "1,2;3,4"}, "b": {"type": "string", "default": "5,6;7,8"}}, "required": ["a", "b"]}, category="线性代数")
def least_squares(a="1,2;3,4", b="5,6;7,8"):
    """最小二乘解 min ||Ax-b||，给出解、残差与秩。"""
    A = _mat(a)
    bb = _mat(b)
    if bb.ndim == 1:
        bb = bb[:, None]
    x, res, rank, sv = np.linalg.lstsq(A, bb, rcond=None)
    resid = float(np.linalg.norm((A @ x) - bb))
    return {"text": f"最小二乘解 x=\n{np.round(x.ravel(),4)}\n残差范数={resid:.4e}\n秩={rank}"}


class LAProPage(ui.BasePage):
    NAME = "高等线性代数"
    EMOJI = "\U0001F4CA"

    def __init__(self, master):
        super().__init__(master, layout=False)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        body = ctk.CTkScrollableFrame(self, corner_radius=ui.RADIUS["md"])
        body.grid(row=0, column=0, sticky="nsew", padx=ui.SPACE["md"], pady=ui.SPACE["md"])
        body.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(body, text="\U0001F4CA 高等线性代数（矩阵分解）", font=ctk.CTkFont(size=ui.FONT["title"], weight="bold")).pack(anchor="w")
        ctk.CTkLabel(body, text="A 用隔行隔列输入，例：1,2;3,5（分号换行、逗号分隔）", font=ctk.CTkFont(size=ui.FONT["caption"])).pack(anchor="w", pady=(0, 4))
        self.a = ctk.CTkEntry(body, placeholder_text="矩阵 A")
        self.a.insert(0, "1,2;3,5")
        self.a.pack(fill="x")
        btn = ctk.CTkFrame(body, fg_color="transparent")
        btn.pack(fill="x", pady=ui.SPACE["sm"])
        ops = [
            ("LU", lambda: self._run(lu_decomp, [self.a.get()])),
            ("QR", lambda: self._run(qr_decomp, [self.a.get()])),
            ("SVD", lambda: self._run(svd_decomp, [self.a.get()])),
            ("Cholesky", lambda: self._run(cholesky_decomp, [self.a.get()])),
            ("正交化", lambda: self._run(gram_schmidt, [self.a.get()])),
            ("零空间", lambda: self._run(nullspace_basis, [self.a.get()])),
        ]
        for i, (txt, fn) in enumerate(ops):
            ctk.CTkButton(btn, text=txt, command=fn, width=80).grid(row=0, column=i, padx=4, pady=4)
        self.out = ui.mono_textbox(body, ui.FONT["body"])
        self.out.pack(fill="both", expand=True, pady=(ui.SPACE["sm"], 0))

    def _run(self, fn, args):
        try:
            r = fn(*args)
        except Exception as e:
            r = {"text": "计算失败：" + repr(e)}
        try:
            self.out.configure(state="normal")
            self.out.delete("1.0", "end")
            self.out.insert("end", r.get("text", "") + "\n")
            self.out.configure(state="disabled")
        except Exception:
            pass


PAGES = [LAProPage]
