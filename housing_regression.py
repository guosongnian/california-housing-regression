"""统一比较线性模型与树模型，预测 California Housing 房价。

运行方式：
    python housing_regression.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_COMMIT = "b7051594afbb1d835f5e031af8664c4363878050"
DATA_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml/"
    f"{SOURCE_COMMIT}/datasets/housing/housing.csv"
)
DATA_SHA256 = "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e"


def file_sha256(path: Path) -> str:
    """计算文件哈希，确保实验始终使用同一份原始数据。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """下载、读取并校验原始数据。"""
    data_path = data_dir / "housing.csv"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print("正在下载 California Housing 数据……")
        urllib.request.urlretrieve(DATA_URL, data_path)

    actual_hash = file_sha256(data_path)
    if actual_hash != DATA_SHA256:
        raise ValueError(
            f"数据校验失败：期望 {DATA_SHA256}，实际 {actual_hash}"
        )

    frame = pd.read_csv(data_path)
    required_columns = {
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    }
    if not required_columns.issubset(frame.columns) or len(frame) != 20_640:
        raise ValueError("数据结构与预期不符")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """添加不依赖全体样本统计量的派生特征。"""
    result = frame.copy()
    result["rooms_per_household"] = result["total_rooms"] / result["households"]
    result["bedrooms_per_room"] = result["total_bedrooms"] / result["total_rooms"]
    result["population_per_household"] = (
        result["population"] / result["households"]
    )
    result["median_income_squared"] = result["median_income"] ** 2
    result["distance_to_sf"] = np.hypot(
        result["longitude"] + 122.4,
        result["latitude"] - 37.8,
    )
    result["distance_to_la"] = np.hypot(
        result["longitude"] + 118.2,
        result["latitude"] - 34.1,
    )
    return result


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按收入区间分层，隔离 20% 独立测试集。"""
    income_group = pd.cut(
        frame["median_income"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=False,
    )
    development, test = train_test_split(
        frame,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=income_group,
    )

    X_development = add_features(
        development.drop(columns="median_house_value")
    )
    y_development = development["median_house_value"].copy()
    X_test = add_features(test.drop(columns="median_house_value"))
    y_test = test["median_house_value"].copy()
    return X_development, X_test, y_development, y_test


def build_search(X_development: pd.DataFrame, quick: bool = False) -> GridSearchCV:
    """在同一次交叉验证中比较全部线性模型和树模型。"""
    numeric_columns = X_development.select_dtypes(include="number").columns
    categorical_columns = X_development.select_dtypes(exclude="number").columns

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    if quick:
        parameter_grid = [
            {"model": [DummyRegressor(strategy="median")]},
            {"model": [Ridge()], "model__alpha": [1.0, 100.0]},
            {
                "model": [DecisionTreeRegressor(random_state=RANDOM_STATE)],
                "model__max_depth": [10],
                "model__min_samples_leaf": [10],
            },
            {
                "model": [HistGradientBoostingRegressor(random_state=RANDOM_STATE)],
                "model__max_iter": [100],
            },
        ]
    else:
        parameter_grid = [
            {"model": [DummyRegressor(strategy="median")]},
            {"model": [LinearRegression()]},
            {"model": [Ridge()], "model__alpha": np.logspace(-2, 4, 13)},
            {
                "model": [Lasso(max_iter=20_000)],
                "model__alpha": [1.0, 10.0, 100.0, 1_000.0],
            },
            {
                "model": [DecisionTreeRegressor(random_state=RANDOM_STATE)],
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [2, 10],
            },
            {
                "model": [
                    RandomForestRegressor(
                        n_estimators=250,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    )
                ],
                "model__max_features": [0.6, 1.0],
                "model__min_samples_leaf": [1, 3],
            },
            {
                "model": [
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        random_state=RANDOM_STATE,
                    )
                ],
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [15, 31],
                "model__l2_regularization": [0.0, 1.0],
            },
        ]
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )


def summarize_search(search: GridSearchCV) -> pd.DataFrame:
    """返回各模型的最佳交叉验证结果。"""
    results = pd.DataFrame(search.cv_results_)
    results["model"] = results["param_model"].map(
        lambda model: type(model).__name__
    )
    results["alpha"] = [
        parameters.get("model__alpha", np.nan)
        for parameters in results["params"]
    ]
    results["CV_RMSE"] = -results["mean_test_score"]
    return (
        results.sort_values("CV_RMSE")
        .groupby("model", as_index=False)
        .first()[["model", "alpha", "CV_RMSE"]]
        .sort_values("CV_RMSE")
        .reset_index(drop=True)
    )


def run(project_root: Path = PROJECT_ROOT, quick: bool = False) -> dict[str, object]:
    """运行完整训练、选择和最终评估流程。"""
    data = load_data(project_root / "data")
    X_development, X_test, y_development, y_test = split_data(data)

    print(f"开发集：{len(X_development):,} 条")
    print(f"测试集：{len(X_test):,} 条")
    print("正在执行 5 折交叉验证……")

    search = build_search(X_development, quick=quick)
    search.fit(X_development, y_development)

    comparison = summarize_search(search)
    print("\n各模型最佳交叉验证结果：")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.0f}"))

    selected_model = search.best_estimator_.named_steps["model"]
    selected_name = type(selected_model).__name__
    selected_alpha = getattr(selected_model, "alpha", None)
    print(f"\n最终模型：{selected_name}")
    if selected_alpha is not None:
        print(f"alpha：{selected_alpha:g}")

    # 模型和参数确定后，只在这里使用一次独立测试集。
    prediction = search.best_estimator_.predict(X_test)
    rmse = mean_squared_error(y_test, prediction) ** 0.5
    mae = mean_absolute_error(y_test, prediction)
    r_squared = r2_score(y_test, prediction)

    print("\n独立测试集结果：")
    print(f"RMSE：{rmse:,.0f}")
    print(f"MAE ：{mae:,.0f}")
    print(f"R²  ：{r_squared:.3f}")

    result = {
        "selected_model": selected_name,
        "best_params": {
            key: (type(value).__name__ if key == "model" else value)
            for key, value in search.best_params_.items()
        },
        "cross_validation_rmse": float(-search.best_score_),
        "test_metrics": {
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r_squared),
        },
    }
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用精简参数网格快速检查完整流程。",
    )
    arguments = parser.parse_args()
    run(PROJECT_ROOT, quick=arguments.quick)


if __name__ == "__main__":
    main()
