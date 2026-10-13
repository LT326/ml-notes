# -*- coding: utf-8 -*-
"""
第 3 章 3.12.3 精通级 · 第 4 题
------------------------------------------------------------
题目：造一份含“城市”列（类别数 100、分布严重长尾）的数据，
      分别用三种方式编码后用逻辑回归做 5 折交叉验证，比较【时间】与【准确率】：
  ① 全部独热编码；
  ② 把出现次数 < 20 的类别归为“其他”后再独热编码；
  ③ 带平滑的目标编码（m = 30，必须用 K 折交叉方式计算）。

工程要点：
  - 目标编码用了标签，必须 K 折交叉 + 平滑，且整条链放进 cross_val_score；
  - 自定义 Transformer 通过实现 fit_transform（折内 out-of-fold 编码）
    与 transform（全量映射）来天然防泄露。

运行： python q4_encoding_comparison.py

============================================================
【阅读指南】本文件按“需掌握程度”做了三级标注：
    ★★★ 必掌握   —— 数据处理核心考点，要能自己讲清原理
    ★★  工程规范 —— 工程化的正确写法，重点理解“为什么”
    ★   脚手架   —— 造数据、打印排版；读代码时可跳过

本题的 ★★★ 必掌握清单：
    1. 长尾类别列的三种处理：全部独热 / 合并稀有 / 目标编码，各自优缺点
    2. 目标编码的平滑公式：(count*类别均值 + m*全局均值) / (count + m)
    3. 目标编码必须 “平滑 + K 折交叉”，否则训练集虚高、测试集崩盘
    4. 用到标签的编码步骤，整条链要放进 cross_val_score 里
============================================================
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
RNG = np.random.default_rng(SEED)
N_SAMPLES = 6000
N_CITIES = 100
RARE_THRESHOLD = 20
SMOOTH_M = 30


# ==========================================================================
# ★ 脚手架（可跳过）：造一份“长尾城市列”的数据
#   用幂律权重让少数城市占绝大多数样本，目的是复现业务里的长尾分布，
#   好让下面的三种编码方式拉开差距。随机数公式本身不是知识点。
# ==========================================================================
def build_long_tail_data(n: int = N_SAMPLES, n_cities: int = N_CITIES) -> pd.DataFrame:
    """用幂律权重采样城市，使样本量分布严重长尾（少数城市占大多数样本）。"""
    cities = [f"城市{i:03d}" for i in range(n_cities)]
    weights = 1.0 / np.arange(1, n_cities + 1) ** 1.4  # 幂律，制造长尾
    weights = weights / weights.sum()
    city = RNG.choice(cities, size=n, p=weights)

    # 每个城市一个真实违约率，与城市 id 相关，制造“可被目标编码学到”的信号
    city_effect = RNG.normal(0, 1.0, size=n_cities)
    effect_map = {c: e for c, e in zip(cities, city_effect)}
    base = np.array([effect_map[c] for c in city])
    logit = 1.8 * base + RNG.normal(0, 0.8, size=n)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (RNG.random(n) < prob).astype(int)

    return pd.DataFrame({"city": city, "target": y})


# ★★★ 必掌握：稀有类别合并（长尾场景的标准操作）
#   把出现次数 < threshold（本题 20）的类别统一归为“其他”，
#   避免产生一大堆“只有 1~2 个样本”的独热列 —— 那种列学不到规律、只会加噪声。
#   小提醒：合并阈值也应只看训练集来定，生产代码里要放进 Pipeline。
def merge_rare_categories(df: pd.DataFrame, col: str, threshold: int) -> tuple[pd.DataFrame, int]:
    """把出现次数 < threshold 的类别合并为“其他”，返回新表与被合并的类别数。"""
    df = df.copy()
    counts = df[col].value_counts()
    rare = counts[counts < threshold].index
    df[col] = df[col].where(~df[col].isin(rare), "其他")
    return df, len(rare)


# ==========================================================================
# ★★★ 必掌握：目标编码（Target Encoding）的两个防泄露要点
#   1) 平滑：编码值 = (count*类别均值 + m*全局均值) / (count + m)
#      公式就在下面 _smooth_map 里，必须记住。
#      m 越大，稀有类别越向全局均值收缩，避免“1~2 个样本的均值被当成规律”。
#   2) K 折交叉：编码某一行时，只能用“它所在折之外”的标签来算均值，
#      即 out-of-fold 编码 —— 否则等于把这一行自己的标签写进它自己的特征，
#      就是最典型的“标签泄露”。
#
# ★★ 工程规范：继承 BaseEstimator + TransformerMixin，实现三个方法：
#      fit           -> 学“全量映射”（供 transform 用），只接收折内数据
#      fit_transform -> 返回 out-of-fold 编码（训练时走这里）
#      transform     -> 查表编码，未见过的类别回退全局均值（预测时走这里）
#   三者配合塞进 Pipeline 后，cross_val_score 会自动隔离各折，天然防泄露。
# ==========================================================================
class KFoldSmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """目标编码器（平滑 + K 折交叉），可安全放进 Pipeline / cross_val_score。

    - fit(X, y)：用传入数据（交叉验证时即“折内训练数据”）算出**全量映射**，供 transform 使用；
    - fit_transform(X, y)：对训练数据做 K 折 out-of-fold 编码（每行编码不含自身标签）；
    - transform(X)：查表编码，未见过的类别回退到训练折的全局均值。

    关键：全程只使用 fit 时传入的 X, y，绝不触碰外部数据，因此不会泄露。
    """

    def __init__(self, col: str, m: int = SMOOTH_M, n_splits: int = 5, random_state: int = 0):
        self.col = col
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state

    @staticmethod
    def _smooth_map(city: pd.Series, y: pd.Series, m: int) -> tuple[dict, float]:
        # ★★★ 必掌握：平滑公式 —— 让样本少的类别向全局均值收缩，是目标编码的灵魂
        global_mean = float(y.mean())
        stat = pd.DataFrame({"city": city.values, "y": y.values}).groupby("city")["y"].agg(["mean", "count"])
        mapping = (stat["count"] * stat["mean"] + m * global_mean) / (stat["count"] + m)
        return mapping.to_dict(), global_mean

    def fit(self, X: pd.DataFrame, y) -> "KFoldSmoothedTargetEncoder":
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        self.mapping_, self.global_mean_ = self._smooth_map(X[self.col], y, self.m)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).reset_index(drop=True)
        enc = X[self.col].map(self.mapping_).fillna(self.global_mean_)
        return enc.to_numpy(dtype=float).reshape(-1, 1)

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> np.ndarray:
        self.fit(X, y)
        X = pd.DataFrame(X).reset_index(drop=True)
        y_arr = np.asarray(y)
        out = np.full(len(X), np.nan)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr, va in kf.split(X):
            mapping, gm = self._smooth_map(
                X.loc[tr, self.col].reset_index(drop=True),
                pd.Series(y_arr[tr]),
                self.m,
            )
            out[va] = pd.Series(X.loc[va, self.col].values).map(mapping).fillna(gm).to_numpy()
        out = np.where(np.isnan(out), self.global_mean_, out)
        return out.reshape(-1, 1)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def run_cv(name: str, pipe: Pipeline, df: pd.DataFrame) -> dict:
    # ★ 脚手架：统一的“跑一次 5 折交叉验证 + 计时”封装，仅用于打分，可跳过
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy()
    t0 = time.perf_counter()
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=1)
    elapsed = time.perf_counter() - t0
    return {"方案": name, "耗时(s)": round(elapsed, 3), "准确率均值": round(scores.mean(), 4), "标准差": round(scores.std(), 4), "列数": None}


# ★ 脚手架：main 负责跑三种方案并打印对比表，流程可扫过；重点是结尾那段“结论”
def main() -> None:
    df = build_long_tail_data()
    counts = df["city"].value_counts()
    print("=" * 72)
    print("数据概况")
    print("=" * 72)
    print(f"样本数            : {len(df)}")
    print(f"城市类别数        : {df['city'].nunique()}")
    print(f"最大类别样本数    : {counts.max()} ({counts.idxmax()})")
    print(f"最小类别样本数    : {counts.min()} ({counts.idxmin()})")
    print(f"出现次数 < {RARE_THRESHOLD} 的稀有类别数 : {int((counts < RARE_THRESHOLD).sum())}")

    clf = LogisticRegression(max_iter=2000)

    # ★★★ 必掌握：三种编码方式，注意“自定义编码器”放在 Pipeline 的什么位置
    # ① 全部独热：100 个城市 -> 100 列；稀有类别的列在训练折内几乎全 0，学不到东西
    pipe1 = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("clf", clf)])
    res1 = run_cv("① 全部独热", pipe1, df)

    # ② 合并稀有类别后独热：先归并成“其他”再独热，维度大幅下降
    df2, n_rare = merge_rare_categories(df, "city", RARE_THRESHOLD)
    pipe2 = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore")), ("clf", clf)])
    res2 = run_cv("② 合并稀有(<20)后独热", pipe2, df2)

    # ③ 平滑 + K 折交叉目标编码：自定义 Transformer 放进 Pipeline，交给 CV 自动隔离
    pipe3 = Pipeline(
        [("te", KFoldSmoothedTargetEncoder(col="city", m=SMOOTH_M, n_splits=5, random_state=0)), ("clf", clf)]
    )
    res3 = run_cv("③ 平滑目标编码(m=30, K折)", pipe3, df)

    # 列数信息（独热后维度）
    res1["列数"] = df["city"].nunique()          # 100 个城市 -> 100 列
    res2["列数"] = df2["city"].nunique()         # 合并稀有后 -> 剩余类别数
    res3["列数"] = 1                             # 目标编码只产出 1 列

    print("\n" + "=" * 72)
    print("三种编码方式对比（逻辑回归 + 5 折交叉验证）")
    print("=" * 72)
    table = pd.DataFrame([res1, res2, res3])[["方案", "列数", "耗时(s)", "准确率均值", "标准差"]]
    print(table.to_string(index=False))

    best = table.loc[table["准确率均值"].idxmax()]
    print("\n结论（结合本次运行结果）：")
    print(f"  - 列数：① {res1['列数']} 列 -> ② {res2['列数']} 列 -> ③ {res3['列数']} 列，")
    print("    合并稀有类别与目标编码把维度大幅压缩，内存与训练时间随之下降；")
    print(f"  - 本次准确率最高的是 {best['方案']}（{best['准确率均值']:.4f}）；")
    print("  - 全部独热的问题：稀有类别的独热列在训练折内几乎全 0，学不到东西，")
    print("    还会因“一城一列”放大方差，长尾场景下泛化更脆弱；")
    print("  - 目标编码只产出 1 列、速度最快，平滑让稀有类别向全局均值收缩、")
    print("    避免把 1~2 个样本的标签均值当成规律，长尾场景通常最稳；")
    print("  - 但目标编码必须靠“平滑 + K 折交叉”防泄露，否则训练集准确率虚高、测试集崩盘。")
    print("  - 低于 1 个百分点内的差距可能落在标准差范围内，工程上应固定种子、重复多次取均值再下结论。")


if __name__ == "__main__":
    main()