# -*- coding: utf-8 -*-
"""
第 4 章 精通级 · 第 1 题
------------------------------------------------------------
题目：用 numpy 手写批量梯度下降(BGD)训练线性回归，画出损失曲线；
      学习率分别取 0.001 / 0.01 / 0.1 / 1.0，观察四种现象并解释。

本脚本会打印：
  - 每个学习率的最终 θ、最终 J(θ)、发生了什么现象
  - 收敛判据：α < 2 / λ_max(XᵀX/m)   ← 本题最关键的一句话
  - α=1.0 为什么必然发散：其数值可以从 λ_max 反推出来，不是巧合

运行： python q1_gradient_descent_lr.py
       图保存到同目录 figures/q1_lr_curves.png

============================================================
【阅读指南】三级标注：
    ★★★ 必掌握   —— 算法核心，要能手推、手写
    ★★  工程规范 —— 工程化的正确写法，重点理解"为什么"
    ★   脚手架   —— 造数据、画图、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 梯度公式： ∂J/∂θ = (1/m) · Xᵀ(Xθ − y)      —— 矩阵形式，一行搞定
    2. 更新公式： θ ← θ − α · ∂J/∂θ
    3. 收敛条件： α < 2 / λ_max    （λ_max 是 XᵀX/m 的最大特征值）
       —— 这才是"学习率不能太大"的数学根源，别只背结论
    4. 四种现象的因果：
         太小  → 每步挪得太少，损失几乎不降（欠训练）
         合适  → 平滑地单调降到最优
         偏大  → 能收敛但来回震荡（在最优值两侧跳跃）
         过大  → 每步越跳越远，损失指数级爆炸（发散）
============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体，避免图上中文变方块
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
RNG = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
N_ITERS = 1000
ALPHAS = (0.001, 0.01, 0.1, 1.0)


# ==========================================================================
# ★ 脚手架（可跳过）：造一份单特征线性数据
#   考量：这里【故意不做标准化】。x 取自 [0, 2]，于是 λ_max ≈ 2.2、
#   收敛上限 2/λ_max ≈ 0.9 —— α=0.1 能收敛、α=1.0 会发散，
#   四种现象正好都落在 0.001~1.0 这个常用区间里，便于一次看全。
#   （若先标准化，λ_max≈1、上限≈2，α=1.0 就变成"临界"而非发散。）
# ==========================================================================
def make_data(n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """返回 (X, y)，X 已包含全 1 的偏置列。"""
    x = RNG.uniform(0.0, 2.0, size=n)
    y = 2.0 + 3.0 * x + RNG.normal(0.0, 0.3, size=n)
    # ★★★ 必掌握：加一列 x0=1，把偏置 θ0 也当成一个"特征"统一处理，
    #     这样 h(x) = Xθ 一个矩阵乘法就写完了，不用给 θ0 单独开分支。
    X = np.column_stack([np.ones(n), x])
    return X, y


def cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """J(θ) = 1/(2m) · Σ(h_θ(xᵢ) − yᵢ)²   —— 与第 4.2 节的损失公式一致。"""
    m = len(y)
    residual = X @ theta - y
    return float(residual @ residual / (2 * m))


# ==========================================================================
# ★★★ 必掌握：手写批量梯度下降（Batch Gradient Descent）
#   所谓"批量"，就是每一步都用【全部 m 个样本】算一次梯度。
# ==========================================================================
def bgd_fit(
    X: np.ndarray, y: np.ndarray, alpha: float, n_iters: int
) -> tuple[np.ndarray, np.ndarray]:
    """批量梯度下降，返回 (最终的 theta, 每一步的 J 值序列)。"""
    m, n = X.shape
    theta = np.zeros(n)                       # ★★★ 初始化；线性回归损失是凸的，全 0 起步即可
    history = np.empty(n_iters + 1)
    history[0] = cost(X, y, theta)

    for k in range(n_iters):
        residual = X @ theta - y              # ★★★ h_θ(x) − y（m 维残差向量）
        grad = X.T @ residual / m             # ★★★ 梯度：(1/m)·Xᵀ(Xθ − y)
        theta = theta - alpha * grad          # ★★★ 更新：θ ← θ − α·∇J
        history[k + 1] = cost(X, y, theta)    # ★ 脚手架：仅用于画损失曲线
    return theta, history


# ==========================================================================
# ★ 脚手架（可跳过）：把可能爆炸的数字压成固定宽度字符串，仅用于打印对齐
# ==========================================================================
def fmt(v: float) -> str:
    if not np.isfinite(v):
        return "inf"
    if abs(v) >= 1e4:
        return f"{v:.2e}"
    return f"{v:.4f}"


# ==========================================================================
# ★ 脚手架（可跳过）：根据损失序列判断"这种现象叫什么"
#   关键：损失有【下界】J(θ*)（由噪声决定，见 main 里的说明），
#   所以判断收敛要用相对差距 (J − J*)/J*，不能拍脑袋定一个绝对值。
# ==========================================================================
def classify(history: np.ndarray, j_star: float) -> str:
    with np.errstate(over="ignore", invalid="ignore"):
        final, first = history[-1], history[0]
    if not np.isfinite(final) or final > first * 1e3:
        return "发散（损失爆炸）"
    gap = (final - j_star) / j_star
    if gap < 1e-3:
        return "已收敛到最优"
    if gap < 0.1:
        return "接近最优，仍有余量"
    return "离最优还很远（欠训练）"


# ==========================================================================
# ★ 脚手架（可跳过）：画图
# ==========================================================================
def plot(results: dict[float, np.ndarray]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左图：完整 1000 步，log 纵轴 —— 才看得清 5 个数量级以上的差距
    ax = axes[0]
    for alpha, hist in results.items():
        ax.plot(np.clip(hist, 1e-8, 1e6), label=f"α={alpha}")
    ax.set_yscale("log")
    ax.set_xlabel("迭代步数 k")
    ax.set_ylabel("J(θ)（对数轴）")
    ax.set_title("四种学习率的损失曲线（越靠下越快）")
    ax.legend()
    ax.grid(alpha=0.3)

    # 右图：α=1.0 前 20 步 —— 看它如何一步步翻倍上冲
    ax = axes[1]
    hist = results[1.0][:21]
    ax.plot(hist, marker="o", ms=4, color="crimson")
    ax.set_xlabel("迭代步数 k")
    ax.set_ylabel("J(θ)")
    ax.set_title("α=1.0 前 20 步：指数级爆炸")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "q1_lr_curves.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    X, y = make_data()
    m = len(y)

    # ★★★ 必掌握：理论收敛上限 α < 2 / λ_max
    #     XᵀX/m 是对称矩阵，用 eigvalsh 求特征值（必为实数，比 eig 更快更稳）
    lam = np.linalg.eigvalsh(X.T @ X / m)
    lam_max = float(lam.max())
    alpha_bound = 2.0 / lam_max

    # ★★ 工程规范：先用正规方程算"标准答案" J(θ*)。
    #    J(θ*) 由数据里的噪声决定，是损失的【下界】—— 梯度下降不可能降到它以下。
    #    所以后面判断"收敛没收敛"必须拿它当参照，不能拍脑袋定一个绝对值。
    theta_star = np.linalg.lstsq(X, y, rcond=None)[0]
    j_star = cost(X, y, theta_star)

    print("=" * 88)
    print("第 4 章 精通级 · 第 1 题：手写批量梯度下降 + 学习率实验")
    print("=" * 88)
    print(f"样本数 m={m}，参数量 n={X.shape[1]}（含偏置列 x0=1），迭代 {N_ITERS} 步")
    print(f"XᵀX/m 的特征值 λ = {np.round(lam, 4)}")
    print(f"λ_max = {lam_max:.4f}   →   理论收敛上限 α < 2/λ_max = {alpha_bound:.4f}")
    print(f"最优损失下界 J(θ*) = {j_star:.6e}（噪声决定，损失不可能低于它）")
    print("-" * 88)
    print(f"{'学习率α':<10}{'最终J(θ)':>14}{'高于最优':>12}{'θ0':>12}{'θ1':>12}   现象")
    print("-" * 88)

    results: dict[float, np.ndarray] = {}
    for alpha in ALPHAS:
        theta, hist = bgd_fit(X, y, alpha, N_ITERS)
        results[alpha] = hist
        final = hist[-1]
        gap = (final - j_star) / j_star
        # 发散的 α 会让 gap 变成天文数字，百分比格式会撑爆排版 → 统一显示 inf
        gap_str = f"{gap:>12.1%}" if np.isfinite(gap) and gap < 1e4 else f"{'inf':>12}"
        print(
            f"{alpha:<10}{fmt(final):>14}{gap_str}"
            f"{fmt(theta[0]):>12}{fmt(theta[1]):>12}   {classify(hist, j_star)}"
        )

    print("-" * 88)
    print(f"参考解（正规方程 lstsq）: θ0={theta_star[0]:.4f}, θ1={theta_star[1]:.4f}")
    print("-" * 88)

    def gap_of(a: float) -> float:
        return (results[a][-1] - j_star) / j_star

    print("【四种现象的解释】")
    print(f"  α=0.001：1000 步后仍高于最优 {gap_of(0.001):.0%} —— 步子太小，欠训练")
    print(f"  α=0.01 ：仅高于最优 {gap_of(0.01):.1%} —— 已很接近，"
          f"残留来自小特征值 λ_min 那一维（它收敛最慢）")
    print(f"  α=0.1  ：高于最优 {gap_of(0.1):.2%} —— 收敛到最优，且稳定不震荡")
    print(f"  α=1.0  ：α > 2/λ_max = {alpha_bound:.4f}，每步把误差放大"
          f" |1−α·λ_max| ≈ {abs(1 - 1.0 * lam_max):.2f} 倍 → 指数级发散")

    out = plot(results)
    print(f"\n图已保存：{out}")

    # ★★ 工程规范：用断言把"结论"锁死，脚本自己就是一份可执行的测试
    assert not np.isfinite(results[1.0][-1]) or results[1.0][-1] > 1e3, "α=1.0 应当发散"
    assert gap_of(0.1) < 1e-3, "α=0.1 应当收敛到最优"
    assert gap_of(0.001) > 1.0, "α=0.001 应明显欠训练"
    print("自检通过：四种现象均符合预期。")


if __name__ == "__main__":
    main()
