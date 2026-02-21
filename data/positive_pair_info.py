import os, sys
import numpy as np
import pandas as pd
from os.path import dirname, abspath

sys.path.append(dirname(dirname(abspath(__file__))))
from const import POS_PAIR_FILES, EDGE_FILES


# 阳性对数据
POS_D2C_PAIRS = pd.read_csv(POS_PAIR_FILES['D2C'],index_col=0).drop_duplicates().loc[:,['DisGENet_id','Compound_id']]
POS_D2H_PAIRS = pd.read_csv(POS_PAIR_FILES['D2H'],index_col=0).drop_duplicates().loc[:,['DisGENet_id','Herb_id']]
POS_DISEASES = list(set(pd.concat([POS_D2C_PAIRS['DisGENet_id'],POS_D2H_PAIRS['DisGENet_id']])))
POS_D2C_DISEASES = list(set(POS_D2C_PAIRS['DisGENet_id']))
POS_D2H_DISEASES = list(set(POS_D2H_PAIRS['DisGENet_id']))
POS_COMPOUNDS = list(set(POS_D2C_PAIRS['Compound_id']))
POS_HERBS = list(set(POS_D2H_PAIRS['Herb_id']))
POS_PAIRS = pd.concat([POS_D2C_PAIRS,POS_D2H_PAIRS])
POS_PAIRS['positive'] = 1


# 有连接的疾病
CONNECTED_DISEASES = set(pd.read_csv(EDGE_FILES['Target_Disease'])['DisGENet_id'])


def get_fold_data(pos_pairs, n_split=5, seed=42):
    df = pos_pairs.query("DisGENet_id in @CONNECTED_DISEASES")
    # 1. 统计疾病频率
    counts = df['DisGENet_id'].value_counts()
    single_idx = df[df['DisGENet_id'].isin(counts[counts == 1].index)].index
    multi_idx = df[df['DisGENet_id'].isin(counts[counts > 1].index)].index

    # 2. 对样本数 >= 2 的疾病进行手动切分
    # 因为样本太少无法用 StratifiedKFold，我们用随机打乱后取余数的方法手动分桶
    df_multi = df.loc[multi_idx].copy()
    indices = df_multi.index.tolist()
    np.random.seed(seed)
    np.random.shuffle(indices)

    # 将索引分成 5 份
    fold_indices = np.array_split(indices, n_split)

    folds = []
    for i in range(n_split):
        val_idx = list(fold_indices[i])
        # 训练集 = 其他 4 份 + 所有的孤儿数据
        train_idx = [idx for idx in indices if idx not in val_idx] + list(single_idx)
        
        folds.append({
            'train': df.loc[train_idx],
            'val': df.loc[val_idx]
        })
    return folds


POS_D2C_FOLDS = get_fold_data(pos_pairs=POS_D2C_PAIRS, n_split=5, seed=42)
POS_D2H_FOLDS = get_fold_data(pos_pairs=POS_D2H_PAIRS, n_split=10, seed=42)


if __name__ == '__main__':
    # print(POS_D2C_PAIRS.head())
    # print(POS_D2H_PAIRS.head())
    # print(POS_PAIRS.head())
    # print(len(POS_D2C_PAIRS), 'positive D2C pairs')
    # print(len(POS_D2H_PAIRS), 'positive D2H pairs')
    # print(len(POS_PAIRS), 'positive pairs')
    print(POS_D2C_FOLDS)