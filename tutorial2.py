import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"
MACHINE_LOG_CSV = "data/machine_log.csv"
SUB_CSV = "data/sample_submission.csv"

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
log = pd.read_csv(MACHINE_LOG_CSV)
sub = pd.read_csv(SUB_CSV)

#Predict with replacing "D" with 1
label_dict = {"A": 0, "B": 0, "C": 0, "D": 1}
train["rank"] = train["rank"].map(label_dict)

# Maintenace in every 1000 batches, so converting remaining by dividing by 1000
train["batch_count_men"] = train["batch_count"] % 1000
test["batch_count_men"] = test["batch_count"] % 1000

log_col = ["temperature_1", "temperature_2", "temperature_3", "pressure"]
stats_col = ["mean", "std", "max", "min"]

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
    log_stats = log.groupby(["line", "batch_count"])[_col].agg(stats_col).reset_index()
    # changing the column name to merge
    log_stats.columns = ["line", "batch_count"] + [f"{_col}_{_stat}" for _stat in stats_col ]
    
    # Adding train, test to the static value in order to connect with the rank
    # Merging datasets "train" and "logs_stats" by "line" and "batch_count"
    train = pd.merge(train, log_stats, on=["line", "batch_count"])
    test = pd.merge(test, log_stats, on=["line", "batch_count"])


# Deleting columns
feat_col = train.drop(["product_id", "rank"], axis=1).columns

# Splitting the data into the one for training and the other for testing.
# - test_size [0.0~1.0]: 0.2 = 20% for Testing, 80% for Training
# - random_state [float or int, default=None]: Controls the shuffling applied to the data before applying the split. 
#                                Pass an int for reproducible output across multiple function calls. See Glossary.
#   > 機械学習のモデルの性能を比較するような場合、どのように分割されるかによって結果が異なってしまうため、
#     乱数シードを固定して常に同じように分割されるようにする必要がある。
x_train, x_valid, y_train, y_valid = train_test_split(train[feat_col], train["rank"], test_size=0.2, random_state=42)

#
model = RandomForestClassifier()
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

# ROCは Receiver operating characteristic（受信者操作特性）、
# AUCは Area under the curve の略で、Area under an ROC curve（ROC曲線下の面積）をROC-AUC
# 第一引数に正解クラス、第二引数に予測スコアのリストや配列をそれぞれ指定する。
score = roc_auc_score(y_valid, y_pred)

print(f">>> Prediction: {y_pred}")
print(f">>> Score: {score}")

# FPR（False Positive Rate: 偽陽性率）: 陰性を間違って陽性と判定した割合 (小さい方が良い)
# TPR（True Positive Rate: 真陽性率）: 陽性を正しく陽性と判定した割合 (大きいほうが良い)
# 閾値: Decreasing thresholds on the decision function used to compute fpr and tpr. 
#      thresholds[0] represents no instances being predicted and is arbitrarily set to np.inf.
fpr, tpr, thresholds = roc_curve(y_valid, y_pred)

plt.plot(fpr, tpr, label="ROC curve (area = xxx)")
plt.show()

# Creating a submission file
#y_test = model.predict_proba(test[feat_col])[:,1]
#sub["prediction"] = y_test
#sub.to_csv(Path("submission") / "submission_tutorial_suyama.csv")



