

import gc
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Literal

import logzero
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup)

sys.path.append('./')

class CFG(BaseModel):
    competition: str = "ai_comp"
    seed: int = 0
    debug: bool = False
    output_dir: Path = Path('./ai_comp/results/expNN007')
    input_dir: Path = Path('./ai_comp/input/')

    note: str = 'gru_fullfold_aux'
    model: str = "gru"

    apex: bool = False
    print_freq: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1000.0
    batch_size: int = 128
    epochs: int = 10
    num_workers: int = 4

    warmup_ratio: int = 0.0
    
    scheduler: Literal['linear', 'cosine'] = 'linear'
    batch_scheduler: bool = True
    encoder_lr: float = 1.0e-3
    decoder_lr: float =  1.0e-3
    min_lr: float = 1.0e-6
    eps: float = 1.0e-6
    betas: list[float] = [0.9, 0.999]

    target_col: str = "target"

    n_folds: int = 5
    trn_fold: list[int] = [0, 1, 2, 3, 4]


cfg = CFG()

cfg.output_dir.mkdir(parents=True, exist_ok=True)

LOGGER = logzero.setup_logger(
    logfile=cfg.output_dir / 'train.log', level=20, fileLoglevel=20)

exp_name = exp_name = cfg.output_dir.parts[-1]



# ====================================================
# Utils
# ====================================================

class COLS:
    product_id = 'product_id'
    line = 'line'
    batch_count = 'batch_count'
    tray_no = 'tray_no'
    position = 'position'
    rank = 'rank'

class LOG_COLS:
    datetime = 'datetime'
    line = 'line'
    maintenance_count = 'maintenance_count'
    batch_count = 'batch_count'
    process_time = 'process_time'
    temperature_1 = 'temperature_1'
    temperature_2 = 'temperature_2'
    temperature_3 = 'temperature_3'
    pressure = 'pressure'


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def get_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return roc_auc_score(y_true, y_pred)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



# ====================================================
# Dataset
# ====================================================

def process_df(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns( 
        ((pl.col(COLS.position) - 1) // 4 - 2).alias('position_y'),
        ((pl.col(COLS.position) - 1) % 4 -2).alias('position_x'),

    ).with_columns(
        (pl.col('position_x') ** 2 + pl.col('position_y') ** 2).alias('position_r2')
    ).with_columns(
        [(pl.col(COLS.tray_no) == i).alias(f'tray_no_is_{i}') for i in range(30)]
    ).with_columns(
         [(pl.col(COLS.line) == i).alias(f'line_is_{i}') for i in [1, 2, 3, 4]]
    ).with_columns(
        (pl.col(COLS.batch_count) % 1000 / 1000).alias('since_maintenance')
    )

    features = ['position_x', 'position_y', 'position_r2']
    features += [f'tray_no_is_{i}' for i in range(30)]
    features += [f'line_is_{i}' for i in [1, 2, 3, 4]]
    features += ['since_maintenance']


    features += [ f'{COLS.line}_te_{i}' for i in range(4)]
    features +=  [ f'{COLS.position}_te_{i}' for i in range(4)]
    features +=  [ f'{COLS.tray_no}_te_{i}' for i in range(4)]

    return df, features

def process_machine_log_df(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col(LOG_COLS.temperature_1) + 273.15) / pl.col(LOG_COLS.pressure)).alias('temp1/P'),
        ((pl.col(LOG_COLS.temperature_2) + 273.15) / pl.col(LOG_COLS.pressure)).alias('temp2/P'),
        ((pl.col(LOG_COLS.temperature_3) + 273.15)/ pl.col(LOG_COLS.pressure)).alias('temp3/P'),

        ((pl.col(LOG_COLS.temperature_1) + 273.15) * pl.col(LOG_COLS.pressure)).alias('temp1xP'),
        ((pl.col(LOG_COLS.temperature_2) + 273.15) * pl.col(LOG_COLS.pressure)).alias('temp2xP'),
        ((pl.col(LOG_COLS.temperature_3) + 273.15) * pl.col(LOG_COLS.pressure)).alias('temp3xP'),

        ((pl.col(LOG_COLS.temperature_3) - pl.col(LOG_COLS.temperature_1)) ).alias('temp3-temp1'),
    )


def load_data(input_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    train_df = pl.read_csv(input_dir / 'train.csv')
    test_df = pl.read_csv(input_dir / 'test.csv')
    sample_submission_df = pl.read_csv(input_dir / 'sample_submission.csv')
    machine_log_df = pl.read_csv(input_dir / 'machine_log.csv')
    return train_df, test_df, sample_submission_df, machine_log_df


def process_train(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(1).alias('is_train'),
    )

    target_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df = df.with_columns(
        pl.col(COLS.rank).replace_strict(target_mapping).alias('target')
    )

    return df

def process_test(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.lit(0).alias('is_train'),
    )

    df = df.with_columns(
        pl.lit(-1).alias('target').cast(pl.Int64),
        pl.lit('None').alias('rank')
    )

    return df

def create_folds(df: pl.DataFrame, n_folds: int, seed: int) -> pl.DataFrame:
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df = df.with_columns(pl.lit(0).alias('fold'))

    # # 行番号を示す列を追加
    df = df.with_row_count(name='row_nr')

    # # 各フォールドごとにループ
    for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df[[COLS.line]].to_numpy(), df[[COLS.rank]].to_numpy(), groups=df[COLS.batch_count].to_numpy())
        ):
        # 'fold'列を更新
        df = df.with_columns(
            pl.when(
                    pl.col('row_nr').is_in(val_idx)
                ).then(
                    fold
                ).otherwise(
                    pl.col('fold')
                ).alias('fold')
        )

    return df


class TrainDataset(Dataset):
    def __init__(self, df: pl.DataFrame, machine_log_df: pl.DataFrame, target_cols=None):

        self.df, self.features = process_df(df)
        self.df = self.df.with_columns(
            pl.Series('id', range(len(self.df)))
        )
        self.machine_log_df = machine_log_df
        self.target_cols = target_cols
        self.labels = df.select(target_cols).to_numpy().reshape(-1, )

        

    def __len__(self):
        return len(self.df)

    def get_features(self):
        return self.features

    def __getitem__(self, item):

        line, batch_count, position = self.df.filter(pl.col('id') == item).select([COLS.line, COLS.batch_count, COLS.position]).to_numpy().reshape(-1, )
    

        features = self.df.filter(
                (pl.col(COLS.line) == line) & (pl.col(COLS.batch_count) == batch_count) & (pl.col(COLS.position) == position)
            ).select(
                self.features 
            ).to_numpy().reshape(-1, ).astype(np.float32)

        log_features = self.machine_log_df.filter(
                pl.col('line') == line
            ).filter(
                pl.col('batch_count') == batch_count
            ).select(
                [LOG_COLS.temperature_1, LOG_COLS.temperature_2, LOG_COLS.temperature_3, LOG_COLS.pressure,
                 'temp1/P', 'temp2/P', 'temp3/P', 'temp1xP', 'temp2xP', 'temp3xP', 'temp3-temp1']
            ).to_numpy().reshape(-1, 4+7).astype(np.float32)
        
        inputs = {
            'log_features': log_features,
            'features': features
        }

        label = torch.tensor([self.labels[item]], dtype=torch.float)
        return inputs, label


# ====================================================
# Model
# ====================================================
    

class CustomModel(nn.Module):
    def __init__(self, n_features: int, dim: int = 128):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(11, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(dim),
            
            nn.Linear(dim, dim),
            nn.ReLU(),

            )

        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.gru2 = nn.GRU(input_size=dim*2, hidden_size=dim, num_layers=1, batch_first=True, bidirectional=True)

        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(dim),

            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(dim)
        )

        self.fc = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(dim//2),
            nn.Linear(dim//2, 1)
        )

        self.aux_head = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(dim//2),

            nn.Linear(dim//2, 1)
        )

    def forward(self, inputs: dict[str, torch.Tensor]):
        log_features = inputs['log_features']
        features = inputs['features']

        features = self.feature_fc(features)
        
        log_features = self.input_projection(log_features)
        out, hidden = self.gru(log_features)
        _feat = torch.cat([
            out[:, :, :out.shape[2]//2] + features.unsqueeze(1),
            out[:, :, out.shape[2]//2:] + features.unsqueeze(1)
             ], dim=2)
        _, hidden = self.gru2(_feat)
        hidden = hidden[-1]

        out = self.fc(hidden)

        aux = self.aux_head(hidden)

        return out, aux

# ====================================================
# Loss
# ====================================================

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, y_preds, labels):
        out, aux = y_preds

        mse_loss = self.mse(aux, labels)

        bce_loss = self.bce(out, (labels == 3).float())

        return mse_loss + bce_loss

# ====================================================
# train loop
# ====================================================

def train_fn(fold: int, train_loader, model, criterion, optimizer, epoch, scheduler, device, cfg: CFG):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0


    for step, (inputs, labels) in enumerate(train_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if cfg.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, cfg: CFG):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.inference_mode():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds[0].to('cpu').numpy())
        end = time.time()
        if step % cfg.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions


# ====================================================
# train loop
# ====================================================
def train_loop(folds: pl.DataFrame, fold: int, machine_log_df: pl.DataFrame, device='cuda:0', cfg: CFG = CFG()):

    print(f"========== fold: {fold} training ==========")

    train_pl_ = folds.filter(pl.col("fold") != fold)

    val_pl = folds.filter(pl.col("fold") == fold)

    for col in [COLS.line, COLS.position, COLS.tray_no]:
        te = TargetEncoder()
        te.fit(train_pl_.select([col]), train_pl_.select(cfg.target_col))

        transformed = te.transform(train_pl_.select([col]).to_numpy())
        train_pl_ = train_pl_.with_columns(
            pl.Series(f"{col}_te_{i}", transformed[:, i]) for i in range(4)
        )


        transformed = te.transform(val_pl.select([col]).to_numpy())
        val_pl = val_pl.with_columns(
            pl.Series(f"{col}_te_{i}", transformed[:, i]) for i in range(4)
            )
        

        with open(cfg.output_dir / f"{col}_te.pkl", 'wb') as f:
            pickle.dump(te, f)


    y_valid = val_pl.select(cfg.target_col).to_numpy()

    train_folds_ = train_pl_


    train_dataset = TrainDataset(train_folds_, machine_log_df, target_cols='target')
    valid_dataset = TrainDataset(val_pl, machine_log_df, target_cols='target')

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg.batch_size ,
                              shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(n_features = train_dataset.get_features().__len__())
    model.to(device)

    optimizer_parameters = model.parameters()
    
    optimizer = AdamW(optimizer_parameters, lr=cfg.encoder_lr,
                      eps=cfg.eps, betas=cfg.betas, weight_decay=1e-2)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.warmup_ratio*num_train_steps), num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler

    num_train_steps = int(len(train_folds_) / cfg.batch_size * cfg.epochs)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    criterion = CustomLoss()

    best_score = 0

    for epoch in range(cfg.epochs):

        start_time = time.time()

        # train
        avg_loss = train_fn(fold, train_loader, model,
                            criterion, optimizer, epoch, scheduler, device, cfg)

        # eval
        avg_val_loss, predictions = valid_fn(
            valid_loader, model, criterion, device, cfg)

        # scoring
        score = get_score(y_valid.reshape(-1, ) == 3, predictions.reshape(-1, ))

        elapsed = time.time() - start_time

        print(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - Score: {score:.4f}')

        if best_score < score:
            best_score = score
            print(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions.reshape(-1, )},
                       cfg.output_dir / f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth")

    predictions = torch.load(cfg.output_dir / f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']

    val_pl = val_pl.with_columns(
        pl.Series(f"pred_{cfg.target_col}", predictions)
    )

    torch.cuda.empty_cache()
    gc.collect()

    return val_pl



# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs, _ in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds[0].to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions



def main():

    seed_everything(cfg.seed)
    train_pl, test_pl, sample_sub_pl, machine_log_pl = load_data(cfg.input_dir)
    
    if cfg.debug:
        train_pl = train_pl.head(1000)

    train_pl = process_train(train_pl)
    test_pl = process_test(test_pl)


    df = pl.concat([train_pl, test_pl.select(train_pl.columns)])

    machine_log_pl = process_machine_log_df(machine_log_pl)

    scaler = StandardScaler()
    scaler.fit(machine_log_pl.select([
        LOG_COLS.temperature_1, LOG_COLS.temperature_2, LOG_COLS.temperature_3, LOG_COLS.pressure,
        'temp1/P', 'temp2/P', 'temp3/P', 'temp1xP', 'temp2xP', 'temp3xP', 'temp3-temp1'
    ]).to_numpy())

    transformed = scaler.transform(machine_log_pl.select([
        LOG_COLS.temperature_1, LOG_COLS.temperature_2, LOG_COLS.temperature_3, LOG_COLS.pressure,
        'temp1/P', 'temp2/P', 'temp3/P', 'temp1xP', 'temp2xP', 'temp3xP', 'temp3-temp1'
    ]).to_numpy())

    machine_log_pl = machine_log_pl.with_columns(
        pl.Series('temperature_1', transformed[:, 0]),
        pl.Series('temperature_2', transformed[:, 1]),
        pl.Series('temperature_3', transformed[:, 2]),
        pl.Series('pressure', transformed[:, 3]),
        pl.Series('temp1/P', transformed[:, 4]),
        pl.Series('temp2/P', transformed[:, 5]),
        pl.Series('temp3/P', transformed[:, 6]),
        pl.Series('temp1xP', transformed[:, 7]),
        pl.Series('temp2xP', transformed[:, 8]),
        pl.Series('temp3xP', transformed[:, 9]),
        pl.Series('temp3-temp1', transformed[:, 10])

    )

    train_pl = df.filter(pl.col('is_train') == 1)
    test_pl = df.filter(pl.col('is_train') == 0)

    train_pl = create_folds(train_pl, cfg.n_folds, cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_result(oof_df: pl.DataFrame):
        labels = oof_df.select(cfg.target_col).to_numpy().reshape(-1, ) == 3
        preds = oof_df.select(f"pred_{cfg.target_col}").to_numpy().reshape(-1, )
        score = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}')

        return score
    

    oof_df = pl.DataFrame()
    for fold in range(cfg.n_folds):
        if fold in cfg.trn_fold:

            seed_everything(cfg.seed)
            _oof_df = train_loop(train_pl, fold, machine_log_pl, device, cfg)
            oof_df = pl.concat([oof_df, _oof_df], how='vertical')
            LOGGER.info(f'=== fold {fold} ====')
            get_result(_oof_df)

    LOGGER.info('====== CV ======')
    get_result(oof_df)
    oof_df.write_csv(cfg.output_dir / 'oof_df.csv')

    # ====================================================
    # inference
    # ====================================================
    test_preds = []
    test_pl = test_pl.with_columns(
        pl.lit(-1).alias('target')
    )

    test_ds = TrainDataset(test_pl, machine_log_df=machine_log_pl, target_cols='target')


    model = CustomModel(n_features=test_ds.get_features().__len__())
    for fold in range(cfg.n_folds):
        if fold in cfg.trn_fold:

            for col in [COLS.line, COLS.position, COLS.tray_no]:
                with open(cfg.output_dir / f"{col}_te.pkl", 'rb') as f:
                    te = pickle.load(f)


                transformed = te.transform(test_pl.select([col]).to_numpy())
                test_pl = test_pl.with_columns(
                    pl.Series(f"{col}_te_{i}", transformed[:, i]) for i in range(4)
                )

            test_ds = TrainDataset(test_pl, machine_log_df=machine_log_pl, target_cols='target')
            test_loader = DataLoader(test_ds, batch_size=cfg.batch_size ,
                                    shuffle=False,
                                    num_workers=cfg.num_workers,)

            model.to(device)
            model = CustomModel(n_features=test_ds.get_features().__len__())

            model.load_state_dict(torch.load(cfg.output_dir / f"{cfg.model.replace('/', '-')}_fold{fold}_best.pth")['model'])
            predictions = inference_fn(test_loader, model, device)
            test_preds.append(predictions.reshape(-1, ))

    test_preds = np.mean(test_preds, axis=0)
    test_pl = test_pl.with_columns(
        pl.Series(f'prediction', test_preds)
    )
    test_pl.select([COLS.product_id, 'prediction']).write_csv(cfg.output_dir / 'submission.csv')


if __name__ == '__main__':
    main()
