import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pickle
import re
import gc

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

TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"
MACHINE_LOG_CSV = "data/machine_log.csv"
SUB_CSV = "data/sample_submission.csv"


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

    print("Done")

    return pred
#==== END of FUNC: predict =========================================================================


#################################################
# Main Procedure
#################################################
# Reading data
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
log = pd.read_csv(MACHINE_LOG_CSV)
#sub = pd.read_csv(SUB_CSV)

# Reducing the memory, if possible    
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

#Predict with replacing "D" with 1
label_dict = {"A": 0, "B": 0, "C": 0, "D": 1}
train["rank"] = train["rank"].map(label_dict)

# Maintenace in every 1000 batches, so converting remaining by dividing by 1000
train["batch_count_men"] = train["batch_count"] % 1000
test["batch_count_men"] = test["batch_count"] % 1000

# Creating datasets for training
x_train = train.drop(columns=["product_id", "rank"])
y_train = train["rank"]
id_train = train["product_id"]

# Creating datasets for prediction
x_test = test.drop( columns=["product_id"])
id_test = test[ ["product_id"] ]


train.info()
train.head()
print( train.shape )

print( f"Mean: {y_train.mean():.4f}" )
print( y_train.value_counts() )

'''
cv = list( StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y_train) )

# Confirmation of index: Learning data for fold=0
print("index(train): ", cv[0][0])

# Confirmation of index: Veridation data for fold=0
print("index(valid): ", cv[0][1])
'''

params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 31, # Default: 31
    "n_estimators": 1000, # Default: 100
    "random_state": 42,
    "importance_type": "gain",
}

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
#test_pred.to_csv("submission/submission_new.csv")

'''
# Creating a submission file
y_test = model.predict_proba(test[feat_col])[:,1]
sub["prediction"] = y_test
sub.to_csv(Path("submission") / "submission_tutorial_suyama.csv")
'''
