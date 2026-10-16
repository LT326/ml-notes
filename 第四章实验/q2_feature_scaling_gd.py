# -*- coding: utf-8 -*-
"""
第 4 章 精通级 · 第 2 题
------------------------------------------------------------
题目：构造两个量纲差异极大的特征（面积、房间数），分别在
      【不标准化】与【标准化】下跑梯度下降，比较收敛速度并解释原因。

做法上的两个关键点（也是本题最容易做错的地方）：
  1. 必须带偏置列 x0=1。否则特征中心化之后模型无法表达 y 的均值，
     损失会被虚高，两者不可比。
  2. 标准化问题的解要【换算回原始参数空间】再比较
     （θ_原始 = θ_标准化 / σ），因为损失曲面在原始空间里只有一张。

运行： python q2_feature_scaling_gd.py
       图保存到同目录 figures/q2_scaling.png

============================================================
【阅读指南】三级标注：
    ★★★ 必掌握   —— 核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解"为什么"
    ★   脚手架   —— 造数据、画图、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 收敛速度看【条件数】κ = λ_max / λ_min，不能只看 λ_max：
         学习率的【上限】由 λ_max 决定；收敛的【快慢】由 κ 决定
    2. 量纲差异的后果链：
         量纲差 100 倍 → λ_max 被抬高数千倍 → α 上限被迫压到极小
         → 连最容易走的那个方向也只能挪极小的步子 → 几乎不动
    3. 几何图像：损失等高线是【细长椭圆】。梯度方向几乎垂直于谷底方向，
         于是要么沿谷底慢慢爬，要么在谷壁间来回震荡（α 接近 2/λ_max 时）
    4. 标准化的作用：把各维拉到同一尺度 → κ 从上千降到 ≈1 → 等高线变圆
         → 梯度直指最优解 → 一两步就到位
    5. 中心化是"免费"的（不改变问题难度），缩放（除以 σ）才是关键
============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
RNG = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
N_ITERS = 2000
GAP_TARGET = 0.01  # "收敛"的判据：损失高于最优不超过 1%


# ==========================================================================
# ★ 脚手架（可跳过）：造两个量纲差异极大的特征
#   面积 ∈ [50, 300]（量纲 ~10²），房间数 ∈ [1, 5]（量纲 ~10⁰）。
# ==========================================================================
def make_data(n: int = 300) -> tuple[np.ndarray, np.ndarray]:
    area = RNG.uniform(50.0, 300.0, size=n)     # 面积（平米）
    rooms = RNG.uniform(1.0, 5.0, size=n)        # 房间数（间）
    y = 1.0 + 0.05 * area + 3.0 * rooms + RNG.normal(0.0, 0.5, size=n)
    return np.column_stack([area, rooms]), y


def add_bias(X: np.ndarray) -> np.ndarray:
    """★★★ 必掌握：加一列全 1，让模型能表达常数项（否则损失会虚高）。"""
    return np.column_stack([np.ones(len(X)), X])


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = len(y)
    r = X @ theta - y
    return float(r @ r / (2 * m))


def bgd(
    X: np.ndarray, y: np.ndarray, alpha: float, n_iters: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量梯度下降；额外返回 θ 的完整轨迹（用于画"走法"）。"""
    m, n = X.shape
    theta = np.zeros(n)
    hist = np.empty(n_iters + 1)
    path = np.empty((n_iters + 1, n))
    hist[0] = cost(X, y, theta)
    path[0] = theta
    for k in range(n_iters):
        grad = X.T @ (X @ theta - y) / m       # ★★★ 梯度公式
        theta = theta - alpha * grad           # ★★★ 更新公式
        hist[k + 1] = cost(X, y, theta)
        path[k + 1] = theta
    return theta, hist, path


def design_stats(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """返回 (λ_max, λ_min, κ)，特征值取自 M = XᵀX/m。"""
    lam = np.linalg.eigvalsh(X.T @ X / len(y))
    # ★★★ 必掌握：λ_max 定学习率上限，λ_max/λ_min 定收敛快慢
    return float(lam.max()), float(lam.min()), float(lam.max() / lam.min())


def first_below(hist: np.ndarray, thresh: float) -> int:
    """★ 脚手架：首次低于阈值的步数；没达到返回 -1。"""
    idx = np.nonzero(hist < thresh)[0]
    return int(idx[0]) if idx.size else -1


# ==========================================================================
# ★ 脚手架（可跳过）：画图
#   左图：原始参数空间的损失等高线 + 两条轨迹 —— 本题的"几何答案"
#   右图：损失曲线（对数纵轴）—— 本题的"量化答案"
# ==========================================================================
def plot(
    X_c: np.ndarray, y_c: np.ndarray, j_star: float,
    path_unstd: np.ndarray, path_std_orig: np.ndarray,
    hist_unstd: np.ndarray, hist_std: np.ndarray,
) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左图：等高线（原始参数空间，θ0 已被"消去"）----
    m = len(y_c)
    XtX = X_c.T @ X_c
    Xty = X_c.T @ y_c
    t1 = np.linspace(-0.15, 0.25, 400)   # θ_面积
    t2 = np.linspace(-1.0, 7.0, 400)     # θ_房间数
    T1, T2 = np.meshgrid(t1, t2)
    J = (
        XtX[0, 0] * T1**2 + 2 * XtX[0, 1] * T1 * T2 + XtX[1, 1] * T2**2
        - 2 * Xty[0] * T1 - 2 * Xty[1] * T2 + y_c @ y_c
    ) / (2 * m)
    levels = j_star * np.array([1.02, 1.1, 1.3, 1.7, 2.5, 4.0, 8.0, 16.0, 32.0])
    ax.contour(T1, T2, J, levels=levels, cmap="viridis",
               linewidths=1.1)

    ax.plot(path_unstd[:, 0], path_unstd[:, 1], "-o", ms=2.5, lw=1.6,
            color="crimson", label="未标准化（沿谷底慢慢爬）")
    ax.plot(path_std_orig[:, 0], path_std_orig[:, 1], "-o", ms=3.5, lw=1.6,
            color="darkorange", label="标准化（几乎一步直达）")
    ax.plot(0.05, 3.0, "k*", ms=16, label="最优解")
    ax.set_xlabel("θ_面积")
    ax.set_ylabel("θ_房间数")
    ax.set_title("损失等高线：明显被拉成细长椭圆")
    ax.legend(loc="upper left", fontsize=9)

    # ---- 右图：损失曲线（对数纵轴）----
    ax2.plot(np.clip(hist_unstd, 1e-6, None), color="crimson",
             label="未标准化")
    ax2.plot(np.clip(hist_std, 1e-6, None), color="darkorange",
             label="标准化")
    ax2.axhline(j_star, ls="--", c="gray", lw=1, label="最优损失 J(θ*)")
    ax2.set_yscale("log")
    ax2.set_xlabel("迭代步数 k")
    ax2.set_ylabel("J(θ)（对数轴）")
    ax2.set_title("损失下降速度对比")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "q2_scaling.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    X_feat, y = make_data()
    mu, sd = X_feat.mean(axis=0), X_feat.std(axis=0)

    # ★★★ 必掌握：中心化是"免费"的 —— 它只是把参数原点平移到数据中心，
    #     不改变损失曲面的形状，也不改变收敛难度。
    #     （真正起作用的是后面的"除以 σ"。）
    X_c = X_feat - mu
    X_unstd = add_bias(X_c)            # 只中心化，不缩放
    X_std = add_bias(X_c / sd)         # 中心化 + 缩放，即 StandardScaler 的效果

    print("=" * 78)
    print("第 4 章 精通级 · 第 2 题：特征标准化对梯度下降收敛的影响")
    print("=" * 78)
    print(f"特征量纲：面积 {X_feat[:, 0].min():.0f}~{X_feat[:, 0].max():.0f}（σ={sd[0]:.1f}）"
          f"   房间数 {X_feat[:, 1].min():.1f}~{X_feat[:, 1].max():.1f}（σ={sd[1]:.1f}）")
    print(f"最优损失下界 J(θ*) = "
          f"{cost(X_unstd, y, np.linalg.lstsq(X_unstd, y, rcond=None)[0]):.6f}")
    print("-" * 78)

    cases = [("未标准化", X_unstd), ("标准化后", X_std)]
    info: dict[str, dict] = {}
    for name, X in cases:
        lam_max, lam_min, kappa = design_stats(X, y)
        # ★★★ 必掌握：两者用【同一个】学习率，才能公平地比收敛速度
        alpha = 1.0 / lam_max
        theta, hist, path = bgd(X, y, alpha, N_ITERS)

        # ★★ 工程规范：用正规方程定出真正的下界，作为"收敛判据"的参照
        theta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
        j_star = cost(X, y, theta_ls)
        steps = first_below(hist, j_star * (1 + GAP_TARGET))

        # 理论估计：慢方向的衰减率决定了要多少步
        slow_rate = abs(1.0 - alpha * lam_min)
        est = (np.log(GAP_TARGET) / np.log(slow_rate)) if 0 < slow_rate < 1 else np.inf

        info[name] = dict(
            lam_max=lam_max, lam_min=lam_min, kappa=kappa, alpha=alpha,
            theta=theta, hist=hist, path=path, j_star=j_star,
            steps=steps, est=est,
        )
        print(f"【{name}】")
        print(f"  λ_max = {lam_max:11.3f}   λ_min = {lam_min:.3f}   "
              f"条件数 κ = {kappa:9.1f}")
        print(f"  学习率上限 2/λ_max = {2 / lam_max:.6f}，"
              f"实际取 α = 1/λ_max = {alpha:.6f}（两种情形相同）")
        print(f"  {N_ITERS} 步后 J(θ) = {hist[-1]:.6f}   "
              f"（高于最优 {(hist[-1] - j_star) / j_star:.1%}）")
        if steps >= 0:
            print(f"  达到「高于最优 {GAP_TARGET:.0%}」只用了 {steps} 步")
        else:
            print(f"  {N_ITERS} 步内【未】达到「高于最优 {GAP_TARGET:.0%}」；"
                  f"按慢方向衰减率估算需约 {est:,.0f} 步")
        print("-" * 78)

    print("【结论】")
    ku, ks = info["未标准化"], info["标准化后"]
    print(f"  条件数 κ：{ku['kappa']:,.0f}  →  {ks['kappa']:.2f}"
          f"（降低约 {ku['kappa'] / ks['kappa']:,.0f} 倍）")
    print("  因果链：量纲差 → λ_max 被抬高数千倍 → 学习率上限 2/λ_max 被迫压小")
    print("          → 同一 α 下所有方向都只能挪极小的步子 → 收敛极慢。")
    print("  几何图像：等高线被拉成细长椭圆，梯度方向几乎垂直于谷底，")
    print("            只能沿谷底慢慢爬；若把 α 放大到接近 2/λ_max，就会在")
    print("            谷壁之间来回震荡（走之字形）。")
    print("  标准化把各维拉到同一尺度，κ 降到 ≈1，等高线变圆，梯度直指最优解。")

    # ★★ 工程规范：标准化问题的解，换算回原始参数空间才能和上面比
    #     z_j = (x_j − μ_j)/σ_j  ⇒  θ_原始_j = θ_标准化_j / σ_j
    path_std_orig = ks["path"][:, 1:3] / sd

    out = plot(
        X_c, y - y.mean(), ku["j_star"],
        ku["path"][:, 1:3], path_std_orig,
        ku["hist"], ks["hist"],
    )
    print(f"\n图已保存：{out}")

    # ★★ 工程规范：断言锁死结论
    assert ku["kappa"] > 100 > ks["kappa"], "条件数对比应符合预期"
    assert ks["steps"] >= 0 and (ku["steps"] < 0 or ks["steps"] < ku["steps"]), \
        "标准化应明显更快达到目标"
    # 正确性校验：标准化解换算回原始空间后，应当等于"真正的最优解"
    # （注意不能拿上面那个"还没收敛的未标准化解"当基准，它离最优还很远）
    theta_ls_unstd = np.linalg.lstsq(X_unstd, y, rcond=None)[0]
    assert np.allclose(
        ks["path"][-1, 1:3] / sd, theta_ls_unstd[1:3], atol=1e-2
    ), "标准化解换算回原始空间后应与正规方程解一致"
    print("自检通过：标准化显著改善条件数与收敛速度，且两者收敛到同一组系数。")


if __name__ == "__main__":
    main()
