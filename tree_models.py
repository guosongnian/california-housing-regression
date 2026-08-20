"""California Housing 树模型基准：决策树、随机森林与梯度提升。"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from housing_regression import RANDOM_STATE, load_data, split_data

def build_search(x:pd.DataFrame,quick:bool=False)->GridSearchCV:
    numeric=x.select_dtypes(include="number").columns; categorical=x.select_dtypes(exclude="number").columns
    preprocess=ColumnTransformer([
        ("numeric",SimpleImputer(strategy="median"),numeric),
        ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical),
    ])
    pipeline=Pipeline([("preprocess",preprocess),("model",DecisionTreeRegressor(random_state=RANDOM_STATE))])
    if quick:
        grid=[{"model":[DecisionTreeRegressor(random_state=RANDOM_STATE)],"model__max_depth":[8,16]}]
    else:
        grid=[
            {"model":[DecisionTreeRegressor(random_state=RANDOM_STATE)],"model__max_depth":[6,10,16,None],"model__min_samples_leaf":[2,10]},
            {"model":[RandomForestRegressor(n_estimators=250,n_jobs=1,random_state=RANDOM_STATE)],"model__max_features":[0.6,1.0],"model__min_samples_leaf":[1,3]},
            {"model":[HistGradientBoostingRegressor(max_iter=300,random_state=RANDOM_STATE)],"model__learning_rate":[0.05,0.1],"model__max_leaf_nodes":[15,31],"model__l2_regularization":[0.0,1.0]},
        ]
    return GridSearchCV(pipeline,grid,scoring="neg_root_mean_squared_error",cv=KFold(5,shuffle=True,random_state=RANDOM_STATE),n_jobs=-1,refit=True)

def summarize(search:GridSearchCV)->pd.DataFrame:
    table=pd.DataFrame(search.cv_results_); table["model"]=table.param_model.map(lambda m:type(m).__name__); table["CV_RMSE"]=-table.mean_test_score
    return table.sort_values("CV_RMSE").groupby("model",as_index=False).first()[["model","CV_RMSE","params"]].sort_values("CV_RMSE")

def run(root:Path,quick:bool=False)->dict:
    sns.set_theme(style="whitegrid"); output=root/"outputs"; output.mkdir(exist_ok=True)
    x_train,x_test,y_train,y_test=split_data(load_data()); search=build_search(x_train,quick); search.fit(x_train,y_train); comparison=summarize(search); print(comparison[["model","CV_RMSE"]].to_string(index=False))
    prediction=search.best_estimator_.predict(x_test)
    result={"selected_model":type(search.best_estimator_.named_steps["model"]).__name__,"best_params":{k:(type(v).__name__ if k=="model" else v) for k,v in search.best_params_.items()},"cross_validation_rmse":float(-search.best_score_),"test_metrics":{"RMSE":float(mean_squared_error(y_test,prediction)**.5),"MAE":float(mean_absolute_error(y_test,prediction)),"R2":float(r2_score(y_test,prediction))}}
    print(json.dumps(result,ensure_ascii=False,indent=2)); comparison.assign(params=comparison.params.astype(str)).to_csv(output/"tree_model_comparison.csv",index=False)
    with (output/"tree_metrics.json").open("w",encoding="utf-8") as f: json.dump(result,f,ensure_ascii=False,indent=2)
    residual=y_test.to_numpy()-prediction; fig,axes=plt.subplots(1,2,figsize=(11,4.5)); sns.scatterplot(x=y_test,y=prediction,alpha=.4,ax=axes[0]); lim=[min(y_test.min(),prediction.min()),max(y_test.max(),prediction.max())]; axes[0].plot(lim,lim,"--",color="black"); axes[0].set(xlabel="Observed price",ylabel="Predicted price",title="Observed vs predicted"); sns.scatterplot(x=prediction,y=residual,alpha=.4,ax=axes[1]); axes[1].axhline(0,ls="--",color="black"); axes[1].set(xlabel="Predicted price",ylabel="Residual",title="Residual diagnostics"); fig.tight_layout(); fig.savefig(output/"tree_diagnostics.png",dpi=160,bbox_inches="tight"); plt.close(fig); return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--quick",action="store_true"); a=p.parse_args(); run(Path(__file__).resolve().parent,a.quick)
