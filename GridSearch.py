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

from sklearn.model_selection import GridSearchCV

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
    
    # Adding train, test to the static value (log_stats) in order to connect with the rank
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

# Classifier
model = RandomForestClassifier() 
#model = RandomForestClassifier(class_weight='balanced') # >>> Score: 0.76
#model = DecisionTreeClassifier() # >>> Score: 0.61
#model = MLPClassifier() # >>> Score: 0.63

'''
REF: https://qiita.com/hara_tatsu/items/581db994ec8866afe8f8

[ n_estimators ]
決定木モデルの本数
整数を指定(デフォルト:100)

[ criterion ]
決定木モデルにデータを分割するための指標
'gini'：ジニ係数（デフォルト）
'entropy'：交差エントロピー
'log_loss'

[ max_depth ]
それぞれの決定木モデルの深さ
整数またはNoneを指定(デフォルト: None)
過学習を抑制するために重要となるパラメータ
一般的に、
小さい値：精度低い
大きい値：精度は高いが過学習になりやすい

[ min_samples_split ]
ノードを分割するために必要となってくるサンプル数
（ノードの中にあるサンプル数が指定した値以下になると決定木の分割が止まる）
整数または小数を指定 (デフォルト: None)
一般的に値が小さすぎるとモデルが過剰適合しやすくなる

[ max_leaf_nodes ]
決定木モデルの葉の数
整数または None を指定 (デフォルト: None)

[ min_samples_leaf ]
決定木の分割後に葉に必要となってくるサンプル数
整数または小数を指定 (デフォルト: 1)

'''

# 検証したいパラメータの指定
search_gs = {
  "n_estimators": [None, 150, 200],
  "criterion": ["gini", "entropy", "log_loss"],
  "class_weight": [None, "balanced"]
}

gs = GridSearchCV(model, search_gs, cv=5)

gs.fit(x_train, y_train)

print(gs.best_params_)



