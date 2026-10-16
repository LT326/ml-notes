# -*- coding: utf-8 -*-
"""
第 4 章 精通级 · 第 5 题
------------------------------------------------------------
题目：在同一份数据上分别训练 Ridge 与 Lasso，打印系数，观察 Lasso 是否
      产生 0 系数；然后改变 alpha，记录被"筛掉"的特征数与 R² 的变化。

本脚本会打印：
  - OLS / Ridge / Lasso 三者的系数对照，以及 Lasso 的零系数个数
  - alpha 从 10⁻² 到 10² 扫描时，Lasso 的非零系数个数与测试 R² 的变化
  - 结论：Lasso 通过把系数压到【恰好 0】实现自动特征选择，Ridge 做不到

运行： python q5_ridge_vs_lasso.py
       图保存到同目录 figures/q5_ridge_lasso.png

============================================================
【阅读指南】三级标注：
    ★★★ 必掌握   —— 核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解"为什么"
    ★   脚手架   —— 造数据、画图、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 两种正则化的目标函数（对照第 2 章的 L1/L2 范数）：
         Ridge: J(θ) + α·Σθⱼ²      ← L2 惩罚
         Lasso: J(θ) + α·|Σ|θⱼ||   ← L1 惩罚
       （sklearn 里 alpha 越大，正则化越强）
    2. 核心差异：Ridge 的系数持续变小但一般【不为 0】；
       Lasso 会把一部分系数压到【恰好 0】→ 自动做特征选择、输出稀疏解。
    3. 为什么 L1 会稀疏（几何直觉）：约束区域从"圆"变成"菱形"，
       菱形的尖角正好落在坐标轴上，最优解很容易顶在尖角 → 某些 θⱼ=0。
       代数上则是 |θ| 在 0 点不可导，梯度下降无法"越过"0。
    4. alpha 的作用：变大 → 系数更小、Lasso 非零个数下降、
       模型偏差↑方差↓；测试 R² 通常呈现"先略升后降"的形态。
    5. 重要前提：必须先标准化！否则惩罚项对量纲大的特征惩罚更重，
       等于在"偷偷"做不公平的特征选择。
    6. 实战里 alpha 要用交叉验证选（RidgeCV / LassoCV），不能拍脑袋定。
============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)
ALPHA_FIX = 1.0                              # 做系数对照时用的固定 alpha
ALPHAS = np.logspace(-2, 2, 40)              # 扫描区间：正则化由弱到强
EPS_ZERO = 1e-8                              # 判定"系数是否恰好为 0"


# ==========================================================================
# ★ 脚手架（可跳过）：画图
# ==========================================================================
def plot(names, coef_dict, sweep, best_alpha) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- 左图：三种方法的系数对照 ----
    idx = np.arange(len(names))
    w = 0.27
    ax.bar(idx - w, coef_dict["OLS"], w, label="OLS（无正则化）", color="gray")
    ax.bar(idx, coef_dict["Ridge"], w, label=f"Ridge (α={ALPHA_FIX:g})", color="steelblue")
    ax.bar(idx + w, coef_dict["Lasso"], w, label=f"Lasso (α={ALPHA_FIX:g})", color="crimson")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylabel("标准化后的系数")
    ax.set_title("系数对照：Lasso 把一部分系数压成了 0")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3, axis="y")

    # ---- 右图：alpha 扫描（非零个数 + 测试 R²）----
    ax2.plot(ALPHAS, sweep["n_nonzero"], "-o", ms=3.5, color="crimson",
             label="Lasso 非零系数个数")
    ax2.set_xscale("log")
    ax2.set_xlabel("alpha（正则化强度，对数轴）")
    ax2.set_ylabel("非零系数个数", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim(-0.3, len(names) + 0.3)
    ax2.grid(alpha=0.3)

    ax3 = ax2.twinx()
    ax3.plot(ALPHAS, sweep["r2_lasso"], "-s", ms=3.5, color="seagreen",
             label="Lasso 测试 R²")
    ax3.plot(ALPHAS, sweep["r2_ridge"], "--", color="steelblue",
             label="Ridge 测试 R²")
    ax3.set_ylabel("测试 R²", color="seagreen")
    ax3.tick_params(axis="y", labelcolor="seagreen")
    ax3.axvline(best_alpha, ls=":", c="gray")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="center right")
    ax2.set_title(f"alpha 扫描：非零个数递减，R² 先稳后降（Lasso 最优 α≈{best_alpha:.3g}）")

    fig.tight_layout()
    out = OUT_DIR / "q5_ridge_lasso.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    data = load_diabetes()
    X, y = data.data, data.target
    names = list(data.feature_names)

    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # ★★★ 必掌握：正则化之前【必须】标准化。
    #     否则 α 对量纲大的特征惩罚更重，等于在偷偷做不公平的特征筛选。
    #     ★★ 工程规范：scaler 只在训练集上 fit。
    scaler = StandardScaler().fit(x_tr)
    x_tr_s, x_te_s = scaler.transform(x_tr), scaler.transform(x_te)

    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=ALPHA_FIX),
        "Lasso": Lasso(alpha=ALPHA_FIX, max_iter=100_000),
    }
    coef_dict, r2_dict = {}, {}
    for name, model in models.items():
        model.fit(x_tr_s, y_tr)
        coef_dict[name] = model.coef_
        r2_dict[name] = r2_score(y_te, model.predict(x_te_s))

    print("=" * 92)
    print("第 4 章 精通级 · 第 5 题：Ridge(L2) 与 Lasso(L1) 的系数对比")
    print("=" * 92)
    print(f"数据：load_diabetes，{len(y_tr)} 训练 / {len(y_te)} 测试；"
          f"系数已标准化，可直接横向比较")
    print("-" * 92)
    print(f"{'特征':<9}{'OLS':>11}{'Ridge':>11}{'Lasso':>11}   说明")
    print("-" * 92)
    for i, nm in enumerate(names):
        o, r, l = coef_dict["OLS"][i], coef_dict["Ridge"][i], coef_dict["Lasso"][i]
        note = "Lasso 压成 0（被筛掉）" if abs(l) < EPS_ZERO else ""
        print(f"{nm:<9}{o:>11.2f}{r:>11.2f}{l:>11.2f}   {note}")

    nz = {k: int(np.sum(np.abs(v) > EPS_ZERO)) for k, v in coef_dict.items()}
    print("-" * 92)
    print(f"{'非零系数个数':<12}"
          + "".join(f"{k}={nz[k]:<6}" for k in coef_dict))
    print(f"{'测试 R²':<12}"
          + "".join(f"{k}={r2_dict[k]:<8.4f}" for k in coef_dict))
    print("-" * 92)
    print(f"Lasso 在 α={ALPHA_FIX:g} 时把 {len(names) - nz['Lasso']} 个特征的系数压到恰好 0，"
          f"实现了自动特征选择；")
    print(f"Ridge 的非零个数 = {nz['Ridge']}（全部保留），只是整体缩小了。")

    # ★★ 工程规范：横向扫描 alpha，观察"稀疏度 vs 泛化能力"的权衡
    rows = []
    for a in ALPHAS:
        las = Lasso(alpha=a, max_iter=100_000).fit(x_tr_s, y_tr)
        rid = Ridge(alpha=a).fit(x_tr_s, y_tr)
        rows.append({
            "alpha": a,
            "n_nonzero": int(np.sum(np.abs(las.coef_) > EPS_ZERO)),
            "r2_lasso": r2_score(y_te, las.predict(x_te_s)),
            "r2_ridge": r2_score(y_te, rid.predict(x_te_s)),
        })
    sweep = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "alpha"}
    best = max(rows, key=lambda r: r["r2_lasso"])

    print("-" * 92)
    print("alpha 扫描（Lasso）：")
    print(f"{'alpha':>10}{'非零个数':>10}{'测试R²(Lasso)':>16}{'测试R²(Ridge)':>16}")
    for r in rows[::5]:  # 每 5 行打印一次，避免刷屏
        print(f"{r['alpha']:>10.3g}{r['n_nonzero']:>10}{r['r2_lasso']:>16.4f}"
              f"{r['r2_ridge']:>16.4f}")
    print("-" * 92)
    print("【结论】")
    print(f"  1. Lasso 的非零系数个数随 alpha 增大而单调下降："
          f"{rows[0]['n_nonzero']} → {rows[-1]['n_nonzero']}。")
    print(f"     这就是「正则化强度↑ → 模型越稀疏 → 越像在做特征选择」。")
    print(f"  2. 测试 R² 并非越大 alpha 越好：在 α≈{best['alpha']:.3g} 附近取到"
          f" Lasso 的峰值 {best['r2_lasso']:.4f}，")
    print(f"     再往上加就快速下滑（欠拟合）。Ridge 的曲线更平滑、衰减更慢。")
    print("  3. 选用原则：")
    print("     - 特征多、且相信只有少数真正有用 → Lasso（顺带得到稀疏可解释模型）")
    print("     - 特征间高度相关、都想保留 → Ridge（Lasso 会在相关特征里随机挑一个）")
    print("     - 两者兼顾 → ElasticNet")
    print("  4. 实战：alpha 一定用交叉验证来选（LassoCV / RidgeCV），不要手工定。")

    out = plot(names, coef_dict, sweep, best["alpha"])
    print(f"\n图已保存：{out}")

    # ★★ 工程规范：断言锁死结论
    assert nz["Lasso"] < nz["Ridge"] == len(names), "Lasso 应产生稀疏解，Ridge 不应"
    assert sweep["n_nonzero"][0] >= sweep["n_nonzero"][-1], "alpha 越大应越稀疏"
    assert best["r2_lasso"] > rows[-1]["r2_lasso"], "过大 alpha 会导致 R² 下降"
    print("自检通过：Lasso 的稀疏性与 alpha-R² 权衡均符合预期。")


if __name__ == "__main__":
    main()
