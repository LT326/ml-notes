# -*- coding: utf-8 -*-
"""
第 4 章 精通级 · 第 3 题
------------------------------------------------------------
题目：用 PolynomialFeatures + StandardScaler + LinearRegression 拟合
      一份非线性数据，分别取 1 阶 / 2 阶 / 8 阶，画出拟合曲线，
      并判断哪种是欠拟合、哪种是过拟合。

本脚本会打印：
  - 每档阶数的参数量、训练 MSE、测试 MSE，以及"测试/训练"的差距倍数
  - 阶数从 1 扫到 12 的整条曲线：训练误差一路下降，测试误差先降后升
  - 结论：2 阶（真实结构）最好；1 阶欠拟合（高偏差）；8 阶过拟合（高方差）

运行： python q3_polynomial_degree.py
       图保存到同目录 figures/q3_polynomial.png

============================================================
【阅读指南】三级标注：
    ★★★ 必掌握   —— 核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解"为什么"
    ★   脚手架   —— 造数据、画图、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 多项式回归的本质：还是【线性】回归 —— 只是先把 x 升维成
       [x, x², x³, …]，再对 θ 做线性拟合。模型对 θ 线性，对 x 非线性。
    2. 升维后【必须】标准化：x=3 时 x⁸=6561，各列量纲差好几个数量级，
       不缩放会让 XᵀX 的条件数爆炸，解不稳定。
    3. 判据：看【训练误差与验证误差的差距】，而不是只看某一个：
         两个都高、且贴得很近      → 欠拟合（高偏差，模型太简单）
         训练极低、验证明显更高    → 过拟合（高方差，模型太复杂）
         两个都低、差距小          → 合适
    4. 阶数就是"模型复杂度"的旋钮。复杂度↑：训练误差单调↓，
       验证误差先降后升 —— 形似字母 U。
    5. 工程要点：PolynomialFeatures 必须放进 Pipeline。
       如果在 train_test_split 之前就先升维、先标准化，就是数据泄露。
============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
RNG = np.random.default_rng(SEED)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
TRUE_COEF = (0.5, 1.0, 2.0)  # y = 0.5x² + x + 2
DEGREES_SHOW = (1, 2, 8)     # 要画曲线的三档
DEGREES_SWEEP = range(1, 13)  # 扫一遍找"最合适"的阶数


# ==========================================================================
# ★ 脚手架（可跳过）：造非线性数据
#   真实关系是二次的，但我们只给模型 x —— 它得自己用 x² 去拟合。
# ==========================================================================
def make_data(n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    x = RNG.uniform(-3.0, 3.0, size=n)
    a, b, c = TRUE_COEF
    y = a * x**2 + b * x + c + RNG.normal(0.0, 1.2, size=n)
    return x, y


# ==========================================================================
# ★★★ 必掌握：多项式回归的工程写法 —— Pipeline 三件套
#   1) PolynomialFeatures：把 [x] 升维成 [x, x², …, x^d]
#   2) StandardScaler：把每一列标准化（升维后各列量纲差极大，必需）
#   3) LinearRegression：对升维后的特征做普通线性回归
#   全部塞进 Pipeline，是为了保证"先划分、再预处理"不被破坏。
# ==========================================================================
def build_model(degree: int) -> Pipeline:
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),          # ★★★ 必掌握：标准化不可省
        ("lr", LinearRegression()),
    ])


def evaluate(x_tr, y_tr, x_te, y_te, degree: int) -> dict:
    model = build_model(degree)
    model.fit(x_tr.reshape(-1, 1), y_tr)        # ★★ 工程规范：sklearn 要二维 X
    tr = mean_squared_error(y_tr, model.predict(x_tr.reshape(-1, 1)))
    te = mean_squared_error(y_te, model.predict(x_te.reshape(-1, 1)))
    # ★★ 工程规范：参数量 = 特征数(d 个) + 1 个截距，是"模型复杂度"的直接度量
    n_params = degree + 1
    return {"degree": degree, "n_params": n_params, "model": model,
            "train_mse": tr, "test_mse": te}


# ==========================================================================
# ★★★ 必掌握：欠拟合 / 合适 / 过拟合的判据（本题的核心考点）
#     - 测试误差远高于训练误差             → 过拟合（高方差）
#     - 测试误差明显高于"最好能做到的水平" → 欠拟合（高偏差）
#     - 其余                               → 合适
#   关键：欠拟合必须拿"最好水平"当参照，所以要先扫完所有阶数才判得准。
#        只看 train/test 比值会误判（1 阶比值 1.85，不高不低，其实是欠拟合）。
# ==========================================================================
def verdict(train_mse: float, test_mse: float, best_test_mse: float) -> str:
    if test_mse / train_mse > 2.0:
        return "过拟合（高方差）"
    if test_mse > best_test_mse * 1.5:
        return "欠拟合（高偏差）"
    return "合适"


# ==========================================================================
# ★ 脚手架（可跳过）：画图
# ==========================================================================
def plot(
    x_tr, y_tr, x_te, y_te, curve_results: dict, sweep: list[dict], best_degree: int
) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- 左图：三档阶数的拟合曲线 ----
    xs = np.linspace(-3.2, 3.2, 400).reshape(-1, 1)
    a, b, c = TRUE_COEF
    ax.plot(xs, a * xs.ravel() ** 2 + b * xs.ravel() + c,
            "k--", lw=1.2, label="真实关系 0.5x²+x+2")
    ax.scatter(x_tr, y_tr, s=28, c="steelblue", label="训练集", zorder=3)
    ax.scatter(x_te, y_te, s=28, c="orange", marker="^", label="测试集", zorder=3)

    colors = {1: "gray", 2: "green", 8: "crimson"}
    for d in DEGREES_SHOW:
        r = curve_results[d]
        ax.plot(xs, r["model"].predict(xs), color=colors[d], lw=1.8,
                label=f"{d} 阶（测试 MSE={r['test_mse']:.2f}）")
    ax.set_ylim(-3, 12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("拟合曲线：1 阶太平直、2 阶贴合、8 阶乱窜")
    ax.legend(fontsize=8.5, loc="upper center")

    # ---- 右图：阶数扫描的 U 形曲线 ----
    degs = [s["degree"] for s in sweep]
    ax2.plot(degs, [s["train_mse"] for s in sweep], "-o", color="steelblue",
             label="训练 MSE")
    ax2.plot(degs, [s["test_mse"] for s in sweep], "-s", color="crimson",
             label="测试 MSE")
    ax2.axvline(best_degree, ls=":", c="gray")
    ax2.annotate(f"测试误差最低\n{best_degree} 阶", (best_degree, 
                 min(s["test_mse"] for s in sweep)),
                 textcoords="offset points", xytext=(8, 28), fontsize=9)
    ax2.set_xlabel("多项式阶数（模型复杂度）")
    ax2.set_ylabel("MSE")
    ax2.set_title("复杂度扫描：训练误差↓、测试误差先降后升（U 形）")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "q3_polynomial.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    x, y = make_data()

    # ★★ 工程规范：先划分，再让 Pipeline 在训练集内部完成升维+标准化
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.4, random_state=SEED
    )

    print("=" * 84)
    print("第 4 章 精通级 · 第 3 题：多项式阶数与欠拟合/过拟合")
    print("=" * 84)
    print(f"训练集 {len(x_tr)} 条，测试集 {len(x_te)} 条；真实关系 y=0.5x²+x+2，噪声 σ=1.2")
    print(f"数据范围 x ∈ [{x.min():.2f}, {x.max():.2f}]  →  "
          f"8 阶时 x⁸ 最大可达 {x.max() ** 8:,.0f}（这就是必须标准化的原因）")
    print("-" * 84)
    print(f"{'阶数':<6}{'参数量':>8}{'训练MSE':>12}{'测试MSE':>12}{'测试/训练':>12}   判断")
    print("-" * 84)

    # ★★ 工程规范：先扫一遍所有阶数。因为"欠拟合"必须拿"这份数据最好能做到
    #    的水平"当参照，而那个最优点只有扫完之后才知道。
    sweep = [evaluate(x_tr, y_tr, x_te, y_te, d) for d in DEGREES_SWEEP]
    best = min(sweep, key=lambda s: s["test_mse"])
    best_test = best["test_mse"]

    curve_results: dict[int, dict] = {}
    for d in DEGREES_SHOW:
        r = evaluate(x_tr, y_tr, x_te, y_te, d)
        curve_results[d] = r
        ratio = r["test_mse"] / r["train_mse"]
        # ★★★ 必掌握：欠拟合 / 过拟合的判据（见上方 verdict 函数）
        tag = verdict(r["train_mse"], r["test_mse"], best_test)
        print(f"{d:<6}{r['n_params']:>8}{r['train_mse']:>12.3f}{r['test_mse']:>12.3f}"
              f"{ratio:>12.2f}   {tag}")

    print("-" * 84)
    print("阶数扫描（训练误差单调下降，测试误差先降后升）：")
    for s in sweep:
        mark = "  ← 测试误差最低" if s["degree"] == best["degree"] else ""
        print(f"  {s['degree']:>2} 阶   训练={s['train_mse']:>8.3f}   "
              f"测试={s['test_mse']:>8.3f}{mark}")

    print("-" * 84)
    print("【判断依据与结论】")
    print("  1 阶：训练、测试误差都偏高且贴得近 → 模型太简单，欠拟合（高偏差）")
    print("  2 阶：两者都低、差距小 → 与真实结构匹配，最合适")
    print(f"  8 阶：训练误差小但测试误差 {curve_results[8]['test_mse']:>6.2f}，"
          f"是训练的 {curve_results[8]['test_mse'] / curve_results[8]['train_mse']:.1f} 倍"
          f" → 过拟合（高方差）")
    print(f"  整个扫描的最优点落在 {best['degree']} 阶 —— 与真实关系（二次）一致。")
    print("  注意：高端阶不是纯噪声 —— 继续加阶数训练误差仍在降，")
    print("        但测试误差已经回升，这正是「方差变大」的直接证据。")

    out = plot(x_tr, y_tr, x_te, y_te, curve_results, sweep, best["degree"])
    print(f"\n图已保存：{out}")

    # ★★ 工程规范：断言锁死结论
    r1, r2, r8 = (curve_results[d] for d in DEGREES_SHOW)
    assert r1["train_mse"] > r2["train_mse"], "1 阶欠拟合，训练误差应高于 2 阶"
    assert r8["train_mse"] < r2["train_mse"], "8 阶更复杂，训练误差应低于 2 阶"
    assert r8["test_mse"] > r2["test_mse"], "8 阶过拟合，测试误差应高于 2 阶"
    assert 2 <= best["degree"] <= 4, f"最合适阶数应接近 2，实际 {best['degree']}"
    print("自检通过：欠拟合/过拟合/最合适三档均符合预期。")


if __name__ == "__main__":
    main()
