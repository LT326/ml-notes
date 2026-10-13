# 机器学习学习笔记

《机器学习》课程的学习笔记与章节习题代码，按章节组织。讲义为 Markdown，习题为可独立运行的 Python 脚本。

## 目录结构

```
ML/
├── ML_LEARNING_PROFILE.md                     # 学习档案
├── 第1章 机器学习导论.md
├── 第2章 机器学习专用数学基础.md
├── 第3章 机器学习工程环境与数据预处理.md
├── 第4章 线性回归与最小二乘法.md
└── 第三章实验/                                # 第 3 章「精通级」习题代码
    ├── q1_clean_split_scale.py                # 清洗 + 分层划分 + 标准化流水线
    ├── q2_standard_vs_minmax.py               # 标准化 vs 归一化分布对比
    ├── q3_breast_cancer_rfe.py                # 相关性筛选 + RFE + 交叉验证
    ├── q4_encoding_comparison.py              # 独热 / 合并稀有 / 目标编码对比
    ├── q5_data_leakage_demo.py                # 数据泄露演示实验
    ├── q6_column_transformer_pipeline.py      # ColumnTransformer + 自定义目标编码 Transformer
    └── q7_bank_income_mnar.py                 # 收入缺失（MNAR）处理方案
```

## 环境准备

依赖用 [uv](https://docs.astral.sh/uv/) 管理：

```powershell
uv venv
uv pip install -r requirements.txt
```

## 运行

`第三章实验/` 下的每个脚本都可独立运行，随机种子已固定，结果可复现：

```powershell
cd 第三章实验
python q1_clean_split_scale.py
```

每个脚本顶部都有【阅读指南】，正文按 `★★★ 必掌握` / `★★ 工程规范` / `★ 脚手架` 三级标注，便于分清"该精读的"与"可跳过的"。

`q2` 与 `q3` 运行后会把图像保存到 `第三章实验/figures/`。

## 第 3 章「精通级」习题要点

| 脚本 | 核心考点 |
| --- | --- |
| q1 | 处理顺序：去重 → IQR 异常值截断 → `stratify` 分层划分 → 仅在训练集 `fit` 标准化 |
| q2 | 标准化保均值/方差、归一化保范围；缩放不改变分布形状，但对极值敏感度不同 |
| q3 | `np.triu(k=1)` 取上三角去重计数；RFE 放进 Pipeline 消除特征选择引发选择偏差 |
| q4 | 长尾高基数类别：全量独热 vs 合并稀有 vs 平滑+K 折目标编码，比较维度/耗时/准确率 |
| q5 | 缩放泄露随维度放大；用全量标签构造特征会造成最严重的标签泄露 |
| q6 | 自定义 Transformer 用 `fit_transform`(折内 OOF) + `transform`(全量映射) 防泄露 |
| q7 | MNAR 缺失机制诊断、填充偏差量化、缺失指示特征价值与数据字典记录 |

## 主要结论

- 所有「学参数」的预处理（均值/中位数/分位数/编码映射/特征选择）只能在训练集 `fit`，测试集只 `transform`。
- 把预处理写进 `Pipeline` / `ColumnTransformer`，交给 `cross_val_score`，可以自动隔离各折数据、杜绝泄露。
- 目标编码必须「平滑 + K 折交叉」，且整条链放进交叉验证内部。