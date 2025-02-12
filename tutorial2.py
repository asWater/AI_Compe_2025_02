import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"
MACHINE_LOG_CSV = "data/machine_log.csv"
SUB_CSV = "data/sample_submission.csv"

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
log = pd.read_csv(MACHINE_LOG_CSV)
sub = pd.read_csv(SUB_CSV)

# === My Pre-Procesing ==================================================
# Count Encording.
#CE_RANK_COL_NAME = "ce_Rank"
#ce_Rank = train["rank"].value_counts().to_dict()
#train[CE_RANK_COL_NAME] = train["rank"].map(ce_Rank)

#Predict with replacing "D" with 1
label_dict = {"A": 0, "B": 0, "C": 0, "D": 1}
train["rank"] = train["rank"].map(label_dict)

# Maintenace in every 1000 batches, so converting remaining by dividing by 1000
train["batch_count_men"] = train["batch_count"] % 1000
test["batch_count_men"] = test["batch_count"] % 1000

log_col = [ 
    "maintenance_count", 
    "temperature_1", 
    "temperature_2", 
    "temperature_3", 
    "pressure" 
]

stats_col = [ "mean", "std", "max", "min" ]


for _col in log_col:
    # static vlume for line, batch_count
    # Gropu by "line" & "batch_count", Aggregated by "mean", "std", "max", "min" for columns listed in log_col
    # i.e. Only the data of "temperature_1", "temperature_2", "temperature_3", "pressure" are stored in log_stats
    # ref: https://pycarnival.com/agg_pandas/
    # groupbyメソッドにより、データフレームはグループ化した列の値に基づいて新しいインデックスが設定されます。
    # しかし、この新しいインデックスは、後続のデータ操作や分析にとって必ずしも便利ではありません。
    # ここでreset_indexメソッドが役立ちます。reset_indexメソッドにより、インデックスは連続した整数にリセットされ、
    # グループ化した列は通常の列に戻ります。
    # この組み合わせにより、データの集約とインデックスの管理が一度に行え、データ分析がより効率的になります。
    # また、結果のデータフレームは、他のPandasのメソッドや機能と組み合わせてさらに操作や分析を行うことが可能です。
    # このように、groupbyとreset_indexの組み合わせは、Pandasを使用したデータ分析において非常に強力なツールとなります。

    #log_stats = log.groupby(["line", "batch_count"])[_col].agg(stats_col).reset_index()
    
    # changing the column name to merge
    #log_stats.columns = ["line", "batch_count"] + [ f"{_col}_{_stat}" for _stat in stats_col ]

    
    if _col == "maintenance_count":
        log_stats = log.groupby(["line", "batch_count"])[_col].agg("sum").reset_index()
        log_stats.columns = ["line", "batch_count", f"{_col}_sum"]
    else:
        log_stats = log.groupby(["line", "batch_count"])[_col].agg(stats_col).reset_index()
        log_stats.columns = ["line", "batch_count"] + [ f"{_col}_{_stat}" for _stat in stats_col ]
     
    
    # Adding train, test to the static value (log_stats) in order to connect with the rank
    # Merging datasets "train" and "logs_stats" by "line" and "batch_count"
    train = pd.merge(train, log_stats, on=["line", "batch_count"])
    test = pd.merge(test, log_stats, on=["line", "batch_count"])


# Deleting columns
# > axis=1 = axis="columns"
feat_col = train.drop(["product_id", "rank"], axis=1).columns

# Splitting the data into the one for training and the other for testing.
# - test_size [0.0~1.0]: 0.2 = 20% for Testing, 80% for Training
# - random_state [float or int, default=None]: Controls the shuffling applied to the data before applying the split. 
#                                Pass an int for reproducible output across multiple function calls. See Glossary.
#   > 機械学習のモデルの性能を比較するような場合、どのように分割されるかによって結果が異なってしまうため、
#     乱数シードを固定して常に同じように分割されるようにする必要がある。
x_train, x_valid, y_train, y_valid = train_test_split(train[feat_col], train["rank"], test_size=0.2, random_state=42)

# Classifiers
'''
class sklearn.ensemble.GradientBoostingClassifier
(*, loss='log_loss', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, 
max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, 
tol=0.0001, ccp_alpha=0.0)

class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, 
solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, 
power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
'''
#model = GradientBoostingClassifier(max_depth=5, random_state=42) # >>> Score: 0.826
#model = GradientBoostingClassifier() # >>> Score: 0.81
#model = RandomForestClassifier(n_estimators=100, class_weight='balanced', criterion='log_loss', random_state=42) # >>> Score: 0.76
#model = RandomForestClassifier(class_weight='balanced') # >>> Score: 0.76
#model = DecisionTreeClassifier() # >>> Score: 0.61
#model = MLPClassifier() # >>> Score: 0.63
#model = MLPClassifier(solver="sgd", hidden_layer_sizes=(200,)) # >>> Score: 0.531
lgbParams = {
    'objective': 'binary',
    'num_iterations': 1000,
    'learning_rate': 0.03,
    #'num_leaves': 31, #Default: 31
    #'n_estimators': 100, #Default: 100
    #'early_stopping_round': 100,
    'seed': 42,
}
model = lgb.LGBMClassifier(**lgbParams) # >>> Score: 0.837

# Checking data and label
print(f"=== x_train ===")
x_train.info()
print(f"=== y_train ===")
y_train.info()

# Training the model
model.fit(x_train, y_train)

# "np.array[:, n]" = m*n行列のnumpy行列(ndarray)から n + 1 列目の要素すべてを取得します
# Inputting the data of 2nd column of model.predict_proba(x_valid) to y_pred 
'''
import numpy as np
mylist = np.array( 
           [["1st Col", "2nd Col", "3rd Col"],
           ["A", "B", "C"] , 
           ["D", "E", "F"]]
         )

print(mylist[:,1])
=> ['2nd Col' 'B' 'E']
'''
y_pred = model.predict_proba(x_valid)[:,1]
#y_pred = model.predict(x_valid)[:,1]

# ROCは Receiver operating characteristic（受信者操作特性）、
# AUCは Area under the curve の略で、Area under an ROC curve（ROC曲線下の面積）をROC-AUC
# 第一引数に正解クラス、第二引数に予測スコアのリストや配列をそれぞれ指定する。
score = roc_auc_score(y_valid, y_pred)

print(f">>> Prediction: {y_pred}")
print(f">>> Score: {score}")

#'''
# FPR（False Positive Rate: 偽陽性率）: 陰性を間違って陽性と判定した割合 (小さい方が良い)
# TPR（True Positive Rate: 真陽性率）: 陽性を正しく陽性と判定した割合 (大きいほうが良い)
# 閾値: Decreasing thresholds on the decision function used to compute fpr and tpr. 
#      thresholds[0] represents no instances being predicted and is arbitrarily set to np.inf.
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)

plt.plot(fpr, tpr, label="ROC curve (area = xxx)")
plt.show()

# Creating a submission file
y_test = model.predict_proba(test[feat_col])[:,1]
sub["prediction"] = y_test
sub.to_csv(Path("submission") / "submission_tutorial_suyama.csv")
#'''


