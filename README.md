# 加州房价预测：统一比较线性模型与树模型

使用 California Housing 数据完成一套可复现的回归分析流程，包括数据检查、特征工程、统一预处理、交叉验证、七类候选模型比较和独立测试集评估。

[查看完整分析 Notebook](notebooks/california_housing_regression.ipynb) · [查看 Python 版本](housing_regression.py)

## 项目内容

- 下载地址固定到上游提交，并使用 SHA-256 校验原始数据。
- 将测试集提前隔离，避免模型选择阶段使用测试结果。
- 通过 `Pipeline` 和 `ColumnTransformer` 在验证折内处理缺失值、缩放和类别编码。
- 在同一次 5 折交叉验证中比较中位数基线、线性回归、Ridge、Lasso、决策树、随机森林和直方图梯度提升。
- 模型与参数全部确定后，只进行一次测试集评估。
- 保存模型比较表和最终测试指标。

## 最终结果

统一 5 折交叉验证结果：

| 模型 | CV RMSE |
|---|---:|
| HistGradientBoosting | **45,607** |
| Random Forest | 48,255 |
| Decision Tree | 58,050 |
| Lasso | 67,536 |
| Ridge | 67,540 |
| LinearRegression | 67,576 |
| 中位数基线 | 118,937 |

选定 HistGradientBoosting 后，才对独立测试集进行一次评估：

| RMSE | MAE | R² |
|---:|---:|---:|
| **43,129** | **28,726** | **0.857** |

交叉验证已经表明树模型明显优于线性模型，不需要利用测试结果进行二次选型。结果也说明收入、位置和住房变量之间存在明显的非线性与交互关系。

## 项目结构

```text
.
├── notebooks/
│   └── california_housing_regression.ipynb
├── .github/workflows/ci.yml
├── housing_regression.py
├── tree_models.py
├── requirements.txt
├── NOTICE.md
└── README.md
```

`tree_models.py` 作为旧命令的兼容入口保留，实际调用 `housing_regression.py` 中同一套统一流程。`data/` 与 `outputs/` 在本地生成，不提交到 GitHub。

## 运行方式

建议使用 Python 3.11 或更高版本；GitHub Actions 使用 Python 3.12 自动检查依赖、模块导入和语法。

```bash
git clone https://github.com/guosongnian/california-housing-regression.git
cd california-housing-regression
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python housing_regression.py
```

快速检查完整流程：

```bash
python housing_regression.py --quick
```

也可以启动 `jupyter lab`，打开 `notebooks/california_housing_regression.ipynb` 阅读完整分析。

## 方法边界

- 房价目标存在上限截断，极高房价区域仍会产生系统误差。
- 经纬度距离仅作为简化地理特征，不代表真实道路距离或通勤时间。
- 随机划分不能完全模拟跨地区或跨时间部署，后续可增加地理分组验证。

来源与使用说明见 [NOTICE.md](NOTICE.md)。
