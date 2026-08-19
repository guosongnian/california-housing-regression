# 加州房价预测：线性回归与正则化

使用 California Housing 数据完成一套可复现的回归分析流程，包括数据检查、特征工程、统一预处理、交叉验证、模型选择和独立测试集评估。

[查看完整分析 Notebook](notebooks/california_housing_regression.ipynb)

## 项目内容

- 使用 Python 自动下载并校验原始数据
- 将测试集提前隔离，避免模型选择阶段使用测试结果
- 通过 `Pipeline` 和 `ColumnTransformer` 统一处理缺失值、数值缩放和类别编码
- 比较中位数基线、普通线性回归、Ridge 与 Lasso
- 使用 5 折交叉验证选择模型和正则化参数
- 只在模型确定后进行一次测试集评估
- 分析预测误差和线性模型的重要系数

## 最终结果

5 折交叉验证结果：

| 模型 | 最优 alpha | CV RMSE |
|---|---:|---:|
| Lasso | 100 | 67,473 |
| Ridge | 10 | 67,483 |
| LinearRegression | — | 67,486 |
| 中位数基线 | — | 118,937 |

交叉验证确定 `Lasso(alpha=100)` 后，使用全部开发集重新训练，并在独立测试集上评估一次：

| 指标 | 测试集结果 |
|---|---:|
| RMSE | 66,349 |
| MAE | 48,820 |
| R² | 0.662 |

三种线性模型的交叉验证表现接近，说明主要收益来自有效特征、统一预处理和规范验证流程，而不是某一个正则化方法的压倒性优势。

## 项目结构

```text
california-housing-regression/
├── notebooks/
│   └── california_housing_regression.ipynb
├── .gitignore
├── NOTICE.md
├── README.md
└── requirements.txt
```

数据会在运行时下载到本地 `data/` 目录，该目录不会提交到 GitHub。

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

打开 `notebooks/california_housing_regression.ipynb`，从上到下运行全部单元格即可。

## 方法边界

- 该数据集的房价目标存在上限截断，线性模型难以完整描述极高房价区域。
- 经纬度距离仅作为简化地理特征，不代表真实道路距离或通勤时间。
- 本项目重点是规范的线性回归实验流程，而不是追求所有模型中的最低误差。

来源与使用说明见 [NOTICE.md](NOTICE.md)。
