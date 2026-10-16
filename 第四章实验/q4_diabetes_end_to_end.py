# -*- coding: utf-8 -*-
"""
第 4 章 精通级 · 第 4 题
------------------------------------------------------------
题目：用 load_diabetes 数据集完成一次端到端回归：
      加载 → 划分 → 标准化 → 训练 → 评估(MSE/RMSE/MAE/R²)
      → 画"真实值 vs 预测值"散点图 → 解释 R² 的含义。

本脚本会打印：
  - 四个指标的具体数值，以及各自的量纲与用途
  - R² 的含义：相对"永远预测均值"这个基准基线提升了多少
  - 各特征（标准化后）的系数，便于解释哪些因素影响最大

运行： python q4_diabetes_end_to_end.py
       图保存到同目录 figures/q4_diabetes.png

============================================================
【阅读指南】三级标注：
    ★★★ 必掌握   —— 核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解"为什么"
    ★   脚手架   —— 造数据、画图、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 四个指标的含义与量纲：
         MSE  = 1/m Σ(y−ŷ)²       单位是 y²（不好解释）
         RMSE = √MSE              单位与 y 相同（最常用）
         MAE  = 1/m Σ|y−ŷ|        对离群点不敏感（更稳健）
         R²   = 1 − SSE/SST        无量纲，"相对基线提升多少"
    2. R² 的精确解释：SST 是"永远预测训练集均值"的平方误差。
       R²=0.45 表示"比无脑猜均值好 45%"。R² 可以为负 → 还不如猜均值。
    3. R² 与 MSE 的取舍：MSE 有量纲、看绝对误差；R² 无量纲、看相对表现。
       汇报时两个都给最稳妥。
    4. 标准化必须只 fit 训练集（Pipeline 自动保证）。
    5. 光看指标不够：散点图看是否存在系统性偏差，残差图看"异方差"
       —— 残差随预测值呈漏斗状，说明线性模型的假设不成立。
============================================================
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_diabetes  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 42
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ==========================================================================
# ★ 脚手架（可跳过）：画图 —— 散点图 + 残差图
# ==========================================================================
def plot(y_te, pred, resid, r2: float) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ---- 左图：真实值 vs 预测值 ----
    lo, hi = float(y_te.min()) - 20, float(y_te.max()) + 20
    ax.scatter(y_te, pred, s=32, c="steelblue", alpha=0.75, edgecolor="w")
    # ★★★ 必掌握：理想情况下点应贴着 y=x 这条线
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.4, label="理想线 y = ŷ")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("真实值 y")
    ax.set_ylabel("预测值 ŷ")
    ax.set_title(f"真实值 vs 预测值（测试集 R² = {r2:.3f}）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- 右图：残差图（诊断用）----
    ax2.scatter(pred, resid, s=32, c="darkorange", alpha=0.75, edgecolor="w")
    ax2.axhline(0, c="r", ls="--", lw=1.2)
    ax2.set_xlabel("预测值 ŷ")
    ax2.set_ylabel("残差 y − ŷ")
    ax2.set_title("残差图：理想应是无规律的云带")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "q4_diabetes.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    data = load_diabetes()
    X, y = data.data, data.target
    names = list(data.feature_names)

    # ★★ 工程规范：先划分，再让 Pipeline 只在训练集上 fit 标准化器
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipe.fit(x_tr, y_tr)

    pred_tr = pipe.predict(x_tr)
    pred_te = pipe.predict(x_te)
    resid_te = y_te - pred_te

    # ★★★ 必掌握：四个指标一起算，理解各自的量纲与用途
    mse = mean_squared_error(y_te, pred_te)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_te, pred_te)
    r2 = r2_score(y_te, pred_te)
    r2_tr = r2_score(y_tr, pred_tr)

    # R² 的分母 SST 就是"永远预测均值"的误差平方和
    sst = float(np.sum((y_te - y_te.mean()) ** 2))
    sse = float(np.sum(resid_te**2))
    baseline_rmse = float(np.sqrt(sst / len(y_te)))

    print("=" * 78)
    print("第 4 章 精通级 · 第 4 题：load_diabetes 端到端回归")
    print("=" * 78)
    print(f"样本 {len(y)} 条 × 特征 {X.shape[1]} 个；"
          f"训练集 {len(y_tr)} / 测试集 {len(y_te)}")
    print(f"目标 y 的范围 [{y.min():.0f}, {y.max():.0f}]，均值 {y.mean():.1f}，"
          f"标准差 {y.std():.1f}")
    print("-" * 78)
    print(f"{'指标':<8}{'数值':>14}{'量纲/含义':>32}")
    print("-" * 78)
    print(f"{'MSE':<8}{mse:>14.2f}   单位是 y²，数值被平方放大")
    print(f"{'RMSE':<8}{rmse:>14.2f}   单位与 y 相同，最直观")
    print(f"{'MAE':<8}{mae:>14.2f}   单位与 y 相同；对离群点更稳健")
    print(f"{'R²':<8}{r2:>14.4f}   无量纲；1=完美，0=和猜均值一样")
    print(f"{'R²(训练)':<8}{r2_tr:>14.4f}   与测试 R² 差距小 → 没有明显过拟合")
    print("-" * 78)
    print("【R² 的含义（重点）】")
    print(f"  SST（永远预测均值的误差平方和）= {sst:,.0f}  →  对应 RMSE≈{baseline_rmse:.2f}")
    print(f"  SSE（模型的残差平方和）        = {sse:,.0f}  →  对应 RMSE≈{rmse:.2f}")
    print(f"  R² = 1 − SSE/SST = 1 − {sse / sst:.3f} = {r2:.4f}")
    print(f"  解读：本模型比「无脑预测均值」把误差平方和降低了 {r2:.1%}。")
    print("  注意：R² 是【相对】指标 —— 换一个本身很难预测的数据集，")
    print("        同样的 RMSE 会得到不同的 R²；两个都要报才不会被误导。")
    print("  另外：R² 可以为负，含义是「还不如直接猜均值」。")
    print("-" * 78)
    print("【标准化后的系数（绝对值越大影响越强）】")
    coef = pipe.named_steps["lr"].coef_
    order = np.argsort(-np.abs(coef))
    for i in order:
        print(f"  {names[i]:<8}{coef[i]:>10.2f}")
    print("-" * 78)
    print("【残差诊断】")
    rel_bias = abs(resid_te.mean()) / y_te.std()
    print(f"  残差均值 {resid_te.mean():.2f}，占 y 标准差 {y_te.std():.1f} 的 {rel_bias:.1%}"
          f"（训练集上残差均值严格为 0，测试集略有偏移是正常的）")
    print(f"  残差标准差 {resid_te.std():.2f}")
    corr = float(np.corrcoef(pred_te, np.abs(resid_te))[0, 1])
    print(f"  |残差| 与预测值的相关系数 {corr:.3f}"
          f"  →  {'存在轻微异方差迹象，线性假设可能不够' if corr > 0.3 else '未见明显异方差'}")

    out = plot(y_te, pred_te, resid_te, r2)
    print(f"\n图已保存：{out}")

    # ★★ 工程规范：断言锁死"这份脚本跑出来的结论"
    assert 0.2 < r2 < 0.8, "糖尿病数据集的线性 R² 应落在一个合理区间"
    assert mae < rmse, "MAE 通常小于 RMSE（RMSE 被大误差放大）"
    assert abs(resid_te.mean()) < 0.15 * y_te.std(), "测试集残差均值不应偏离过大"
    print("自检通过：四个指标与 R² 的解读均符合预期。")


if __name__ == "__main__":
    main()
