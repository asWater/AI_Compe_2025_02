import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import re
import gc
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb


# Constants
TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"
MACHINE_LOG_CSV = "data/machine_log.csv"
SUB_CSV = "data/sample_submission.csv"

import math

#===================================================================================================
# FUNC: get_row_num_matrix
#===================================================================================================
def get_row_num_matrix( maxSlotsInRow, slotId ):
  if (slotId % maxSlotsInRow) == 0:
    row_num = math.floor( (slotId / maxSlotsInRow) )
  else:
    row_num = math.floor(slotId / maxSlotsInRow) + 1 

  return row_num

#===================================================================================================
# FUNC: get_col_num_matrix
#===================================================================================================
def get_col_num_matrix( maxSlotsInRow, slotId ):

  if (slotId % maxSlotsInRow) == 0:
    col_num = maxSlotsInRow
  else:
    row_num = math.floor(slotId / maxSlotsInRow)
    col_num = slotId - (row_num * maxSlotsInRow)
  
  return col_num


#===================================================================================================
# FUNC: reduce_mem_usage
#===================================================================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            pass

        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

        return df
    
#==== END of FUNC: reduce_mem_usage ================================================================



#===================================================================================================
# FUNC: train_data
#===================================================================================================
def train_data( input_x,
                input_y,
                input_id,
                params,
                list_nfolds=[0, 1, 2, 3, 4],
                n_splits=5,
                ):
    train_oof = np.zeros( len(input_x) )
    metrics = []
    imp = pd.DataFrame()

    # Cross-Validation
    cv = list( StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(input_x, input_y) )

    for nfold in list_nfolds:
        print("-"*20, nfold, "-"*20)

        # Make Dataset
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        # !!! Following lines are failed due to the error "pandas.errors.IndexingError: Too many indexers" at inpu_id.loc[idx_tr, :]
        #x_tr, y_tr, id_tr = input_x.loc[idx_tr, :], input_y[idx_tr], input_id.loc[idx_tr, :]
        #x_va, y_va, id_va = input_x.loc[idx_va, :], input_y[idx_va], input_id.loc[idx_va, :]
        x_tr, y_tr = input_x.loc[idx_tr, :], input_y[idx_tr]
        x_va, y_va = input_x.loc[idx_va, :], input_y[idx_va]

        print( x_tr.shape, x_va.shape )

        # Training
        model = lgb.LGBMClassifier( **params )
        model.fit( x_tr,
                   y_tr,
                   eval_set=[(x_tr, y_tr), (x_va, y_va)],
                   #early_stopping_rounds=100,
                   #verbose=100,
                   )
        
        fname_lgb = "model_lgb_fold{}.pickle".format(nfold)
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)

        # Evaluation
        y_tr_pred = model.predict_proba(x_tr)[:, 1]
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append( [nfold, metric_tr, metric_va] )
        print( f"[AUC] tr:{metric_tr:.4f} va:{metric_va:.4f}" )

        # OOF (Out Of Fold): Data that is not used for training
        train_oof[idx_va] = y_va_pred

        # Feature Importance
        _imp = pd.DataFrame( {"col":input_x.columns, "imp":model.feature_importances_, "nfold":nfold} )
        imp = pd.concat( [imp, _imp] )

    print( "-"*20, "Result", "-"*20 )

    # Metric
    metrics = np.array( metrics )
    print(metrics)
    print( "[CV] tr:{:.4f}{:.4f}, va:{:.4f}+-{:.4f}".format(
        metrics[:,1].mean(), metrics[:,1].std(),
        metrics[:,2].mean(), metrics[:,2].std()
    ) )

    print(f"[OOF] {roc_auc_score( input_y, train_oof ):.4f}")

    # oof
    train_oof = pd.concat([
        input_id,
        pd.DataFrame({"pred": train_oof})
    ], axis=1)

    # Importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    return train_oof, imp, metrics

#==== END of FUNC: train_data ===========================================================================

#===================================================================================================
# FUNC: predict
#===================================================================================================
def predict(
        input_x,
        input_id,
        list_nfold=[0,1,2,3,4],
):
    pred = np.zeros( (len(input_x), len(list_nfold)) )

    for nfold in list_nfold:
        print( "-"*20, nfold, "-"*20 )
        fname_lgb = f"model_lgb_fold{nfold}.pickle"
        
        with open( fname_lgb, "rb" ) as f:
            model = pickle.load(f)
        
        pred[:, nfold] = model.predict_proba(input_x)[:,1]

    pred = pd.concat([
        input_id,
        pd.DataFrame( {"prediction": pred.mean( axis=1 )} ),
    ], axis=1)

    print(">>> Prediction is finished <<<")

    return pred
#==== END of FUNC: predict =========================================================================

#===================================================================================================
# FUNC: ope_chara
#===================================================================================================
def ope_chara(train_df, test_df, log_df):
    log_col = [ 
        "maintenance_count",
        #"batch_count",
        "temperature_1", 
        "temperature_2", 
        "temperature_3", 
        "pressure" 
    ]

    stats_col = [
        #"sum", 
        #"mean", 
        #"std", 
        #"max", 
        #"min", 
        "median", 
        "first",
    ]

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
        
        if _col == "pressure":
            log_stats = log_df.groupby(["line", "batch_count"])[_col].agg(stats_col).reset_index()
            log_stats.columns = ["line", "batch_count"] + [ f"{_col}_{_stat}" for _stat in stats_col ]
        else:
            log_stats = log_df.groupby(["line", "batch_count"])[_col].agg("sum").reset_index()
            log_stats.columns = ["line", "batch_count", f"{_col}_sum"]
        
        # Adding train, test to the static value (log_stats) in order to connect with the rank
        # Merging datasets "train" and "logs_stats" by "line" and "batch_count"
        train_df = pd.merge(train_df, log_stats, on=["line", "batch_count"])
        test_df = pd.merge(test_df, log_stats, on=["line", "batch_count"])

    train_df = add_more_charas( train_df, log_col )
    test_df = add_more_charas( test_df, log_col )

    return train_df, test_df

#==== END of FUNC: ope_chara =========================================================================

#===================================================================================================
# FUNC: add_more_charas
#===================================================================================================
def add_more_charas( df, log_cols):
    MAX_SLOTS_IN_ROW = 4
    #=== TRAINING DATA ====================
    df["pos_row"] = df.apply( lambda x: get_row_num_matrix(MAX_SLOTS_IN_ROW, x["position"]), axis=1 )
    df["pos_col"] = df.apply( lambda x: get_col_num_matrix(MAX_SLOTS_IN_ROW, x["position"]), axis=1 )
    
    # +
    #df["line_row"] = df["line"] + df["pos_row"]
    #df["line_col"] = df["line"] + df["pos_col"]
    #df["line_tray_row"] = df["line"] + df["pos_row"] + df["tray_no"]
    #df["line_tray_col"] = df["line"] + df["pos_col"] + df["tray_no"]
    #df["tray_row"] = df["tray_no"] + df["pos_row"]
    #df["tray_col"] = df["tray_no"] + df["pos_col"]

    # X
    df["line_row"] = df["line"] * df["pos_row"]
    df["line_col"] = df["line"] * df["pos_col"]
    df["line_tray_row"] = df["line"] * df["pos_row"] * df["tray_no"]
    df["line_tray_col"] = df["line"] * df["pos_col"] * df["tray_no"]
    #df["tray_row"] = df["tray_no"] * df["pos_row"]
    #df["tray_col"] = df["tray_no"] * df["pos_col"]

    df["line_tray"] = df["line"] * df["tray_no"]
    #df["row_x_col"] = df["pos_row"] * df["pos_col"]

    #df["press_median_first"] = df["pressure_median"] - df["pressure_first"]

    df["tray_no_even"] = df.apply( lambda x: 1 if x["tray_no"] % 2 == 0 else 0, axis=1 )
    #df["col_edge"] = df.apply( lambda x: 1 if x["pos_col"] == 1 or x["pos_col"] == 4 else 0, axis=1 )
    #df["position"] = df.apply( lambda x: 1 if x["position"] % 2 == 0 else 0, axis=1 )
    #df["batch_cnt_event"] = df.apply( lambda x: 1 if x["batch_count_men"] % 2 == 0 else 0, axis=1 )

    #df["temp_all_sum_avg"] = ( df["temperature_1_sum"] + df["temperature_2_sum"] + df["temperature_3_sum"] ) / 3
    
    #for _col in log_cols:
    #    df[f"{_col}_max_min_diff"] = df[f"{_col}_max"] - df[f"{_col}_min"]

    #df.drop( columns=["batch_count", ], inplace=True )

    return df

#==== END of FUNC: add_more_charas =================================================================


#===================================================================================================
# FUNC: create_train_data
#===================================================================================================
def create_train_data():
    # Reading data
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    log = pd.read_csv(MACHINE_LOG_CSV)
    #sub = pd.read_csv(SUB_CSV)

    # Reducing the memory, if possible    
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    log = reduce_mem_usage(log)

    # Predict with replacing "D" with 1
    label_dict = {"A": 0, "B": 0, "C": 0, "D": 1}
    train["rank"] = train["rank"].map(label_dict)

    # Maintenace in every 1000 batches, so converting remaining by dividing by 1000
    train["batch_count_men"] = train["batch_count"] % 1000
    test["batch_count_men"] = test["batch_count"] % 1000

    # Characteristic Operation
    train, test = ope_chara(train, test, log)

    # Checking the final train data
    print("-"*20, "Final Train Data", "-"*20)
    print(train.info())
    print(train.head().T)
    print( train.shape )

    return train, test

#==== END of FUNC: create_train_data =======================================================================

############################################################################################################
# Main Procedure
############################################################################################################
# Creating the training / test data
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

# Creating datasets for prediction
x_test = test.drop( columns=["product_id"])
id_test = test[ ["product_id"] ]

print( f"Mean: {y_train.mean():.4f}" )
print( y_train.value_counts() )


params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "importance_type": "gain",
    "random_state": 42,
    "learning_rate": 0.02, # Default: 0.1
    "num_leaves": 52, # Default: 31
    "n_estimators": 850, # Default: 100
    "min_child_samples": 280, # Default: 20,
    #"min_child_weight": 0.1, # Default: 0.001 
    #'subsample': 0.9046304961782754, # Deafault: 1.0 (0~1)
    #'subsample_freq': 4, # Default: 0
    #'reg_alpha': 1.1891647991566514, # Default: 0.0
    #'reg_lambda': 1.6025630427886866, # Default: 0.0
    #"class_weight": "balanced", # Default: None 
}

'''
params = {
  'num_leaves': 31, 
  'learning_rate': 0.05508630645991973, 
  'n_estimators': 1176, 
  'min_child_samples': 60, 
  'subsample': 0.9046304961782754, 
  'subsample_freq': 4, 
  'colsample_bytree': 0.7035257576603995, 
  'reg_alpha': 1.1891647991566514, 
  'reg_lambda': 1.6025630427886866,
  'boosting_type': 'gbdt', 
  'objective': 'binary', 
  'metric': 'auc'
}
#'''

# Training
train_oof, imp, metrics = train_data ( 
                            x_train,
                            y_train,
                            id_train,
                            params,
                            list_nfolds=[0, 1, 2, 3, 4],
                            n_splits=5,
                        )

# Showing the importance of the feature
print( imp.sort_values("imp", ascending=False)[:10] )

# Prediction
test_pred = predict(
    x_test,
    id_test,
    list_nfold=[0, 1, 2, 3, 4],
)

# ROCは Receiver operating characteristic（受信者操作特性）、
# AUCは Area under the curve の略で、Area under an ROC curve（ROC曲線下の面積）をROC-AUC
# 第一引数に正解クラス、第二引数に予測スコアのリストや配列をそれぞれ指定する。
#score = roc_auc_score(x_train, test_pred)

# Submission of prediction results
test_pred.to_csv("submission/submission_new.csv")

'''
# Creating a submission file
y_test = model.predict_proba(test[feat_col])[:,1]
sub["prediction"] = y_test
sub.to_csv(Path("submission") / "submission_tutorial_suyama.csv")
'''
