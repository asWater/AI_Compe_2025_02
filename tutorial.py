import numpy as np
import pandas as pd
from sklearn import datasets

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

TEST_CSV = "data/test.csv"
TRAIN_CSV = "data/train.csv"
MACHINE_LOG_CSV = "data/machine_log.csv"
SUB_CSV = "data/sample_submission.csv"

train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
log = pd.read_csv(MACHINE_LOG_CSV)
sub = pd.read_csv(SUB_CSV)

#plt.figure(figsize=[20, 5])
#sns.countplot(data=train, x="tray_no", hue="rank")
train.head()
test.head()
log.head()
sub.head()

# Compare the time differencies of Temp and Pressure for rank A and D 
import random
a_idx = train[ train["rank"] == "A" ].index
d_idx = train[ train["rank"] == "D" ].index
a_idx = random.sample(list(a_idx), 10)
d_idx = random.sample(list(d_idx), 10)

fig, ax = plt.subplots(2,2, figsize=[12, 7])

ax_f = ax.flatten()

for idx in a_idx:
    line = train.loc[idx, "line"]
    batch = train.loc[idx, "batch_count"]
    _log = log[(log["line"] == line) & (log["batch_count"] == batch)]
    ax_f[0].plot(_log["process_time"], _log["temperature_1"], color="red")
    ax_f[1].plot(_log["process_time"], _log["temperature_2"], color="red")
    ax_f[2].plot(_log["process_time"], _log["temperature_3"], color="red")
    ax_f[3].plot(_log["process_time"], _log["pressure"], color="red")

for idx in d_idx:
    line = train.loc[idx, "line"]
    batch = train.loc[idx, "batch_count"]
    _log = log[(log["line"] == line) & (log["batch_count"] == batch)]
    ax_f[0].plot(_log["process_time"], _log["temperature_1"], color="blue")
    ax_f[1].plot(_log["process_time"], _log["temperature_2"], color="blue")
    ax_f[2].plot(_log["process_time"], _log["temperature_3"], color="blue")
    ax_f[3].plot(_log["process_time"], _log["pressure"], color="blue")




plt.show()
