import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb

from classifier import create_train_data

train, test = create_train_data()

# Creating datasets for training
x_train = train.drop(columns=["product_id", "rank"])
y_train = train["rank"]
id_train = train["product_id"]

# Converting category variants to category type
for col in x_train.columns:
    if x_train[col].dtype == "0":
        print( "Category variant is found")
        #x_train[col] = x_train[col].astype("category")

params_base = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    #"learning_rate": 0.03,
    #"n_estimators": 1000, # Default: 100
    #"bagging_freq": 1,
}

# 目的関数の定義
def objective(trial):
    # 探索するハイパーパラメータ
    params_tuning = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1 ),
        "n_estimators": trial.suggest_int("n_estimators", 100, 10000 ),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        "min_sum_hessian_in_leaf": trial.suggest_float("min_sum_hessian_in_leaf", 1e-5, 1e-2, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 1e+2, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 1e+2, log=True),
    }
    params_tuning.update(params_base)
    
    # モデル学習・評価
    list_metrics = []
    cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x_train, y_train))
    #list_fold = [0]  # 処理高速化のために1つめのfoldのみとする。
    list_fold = [0, 1, 2, 3, 4]  # 処理高速化のために1つめのfoldのみとする。
    for nfold in list_fold:
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr = x_train.loc[idx_tr, :], y_train[idx_tr]
        x_va, y_va = x_train.loc[idx_va, :], y_train[idx_va]
        model = lgb.LGBMClassifier(**params_tuning)
        #model.fit(x_tr,
        #          y_tr,
        #          eval_set=[(x_tr,y_tr), (x_va,y_va)],
        #          early_stopping_rounds=100,
        #          verbose=0,
        #         )
        
#         # 2024/02/14環境で動かしたい場合はこのコードを利用してください。
        model.fit(x_tr,
                   y_tr,
                   eval_set=[(x_tr,y_tr), (x_va,y_va)],
                   callbacks=[
                       lgb.early_stopping(stopping_rounds=100, verbose=True),
                       lgb.log_evaluation(0),
                   ],
                  )
        
        y_va_pred = model.predict_proba(x_va)[:,1]
        metric_va = roc_auc_score(y_va, y_va_pred) # 評価指標をAUCにする
        list_metrics.append(metric_va)
    
    # 評価指標の算出
    metrics = np.mean(list_metrics)
    
    return metrics

sampler = optuna.samplers.TPESampler(seed=123)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=5)

trial = study.best_trial
print("="*40)
print("-"*20, f"acc(best)={trial.value:.4f}", "-"*20)
print(trial.params)

params_best = trial.params
params_best.update(params_base)
print("="*20, f"Param with base params", "="*20)
print(params_best)