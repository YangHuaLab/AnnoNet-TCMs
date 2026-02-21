import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from glob import glob
import logging
import multiprocessing as mp
from math import ceil

from os.path import abspath, dirname, join

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, cosine, correlation, chebyshev, canberra

from data.positive_pair_info import POS_D2C_DISEASES, POS_D2H_DISEASES, POS_DISEASES, POS_PAIRS
from data.ppi_info import PPI_DIST_MAT, PPI_DIST_MAT_NODE2ID
from MHNetwork import MHNetwork
from const import ROOT_PATH, OUT_PATH
from logger import run_log, worker_log
from util_funcs import getNeighborNodes, getParamList, getParamStr, loadMHNet, loadPmap
from vectorized_distances import vectorized_distance_matrix, distance_1d

import warnings
warnings.filterwarnings("ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


NS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 400]
METHODS = [
    # 随机距离
    'random',
    # 基于网络结构的距离
    'jaccard', 'molecular',
    # 传播图概率分布距离
    'cosine', 'correlation', 'l1', 'l2', 'chebyshev',
    'canberra', 'sqeuclidean', 
    'js', 'pearson', 'bhattacharyya',
    'hellinger', 'wasserstein', 'total_variation',
    # 不对的距离
    'chi-square',
    # 'braycurtis',  # 计算时，会内存报错，但不是内存不足
    # 'minkowski',  # p=1时，曼哈顿距离；p=2时，欧氏距离；p=无穷大时，切比雪夫距离
]


def get_progress(i, total) -> str:
    if isinstance(total, (list, np.ndarray)):
        digits = len(str(len(total)))
        progress = f"{i:0{digits}}/{len(total)}"
    elif isinstance(total, int):
        digits = len(str(total))
        progress = f"{i:0{digits}}/{total}"
    else:
        progress = ""
    return progress


# 计算相似度、相似距离
# 余弦相似度
def get_cosine_similarity(x: NDArray, y: NDArray) -> float:
    if np.all(x == 0) or np.all(y == 0):
        return 0
    return np.sum(x * y) / np.sqrt(np.sum(np.square(x)) * np.sum(np.square(y)))


# l2相似度
def get_l2_similarity(x, y) -> float:
    return np.sqrt(np.sum(np.abs(x - y) ** 2))


# l1相似度
def get_l1_similarity(x, y) -> float:
    return np.sum(np.abs(x - y))


def get_distance(x, y, method='cosine'):
    assert method in METHODS
    
    if method == 'jaccard':
        return 1 - len(np.intersect1d(x, y)) / len(np.union1d(x, y))

    if np.all(x == 0) or np.all(y == 0):
        return 0

    # 不记得当初为什么要减均值了
    # x1 = x - x.mean()
    # y1 = y - y.mean()

    # 加一个小值防止除零
    epsilon = 1e-15
    x = x + epsilon
    y = y + epsilon
    # 如果将向量和放大至1，由于精度差异会导致实际和大于1
    # 会导致np.sqrt(1-np.sum())之类计算时出错
    # 对某些距离函数（jensenshannon, hellinger）有影响
    # 这种情况下，直接令结果为0
    x = x / np.sum(x)
    y = y / np.sum(y)

    if method == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        # cos_sim = 1 - cosine(x, y)
        # return 1 - (cos_sim + 1) / 2
        # return cosine(x, y) / 2
    elif method == 'correlation':
        # return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return correlation(x, y)
    elif method == 'l1':
        return np.sum(np.abs(x - y))
    elif method == 'l2':
        return np.sqrt(np.sum(np.abs(x - y) ** 2))
    elif method == 'canberra':
        # return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))
        return canberra(x, y)
    elif method == 'kl':
        return entropy(x, y)  # distance
    elif method == 'js':
        js = jensenshannon(x, y)
        return 0 if np.isnan(js) else js
    elif method == 'bhattacharyya':
        return -np.log(np.sum(np.sqrt(x * y)))
    elif method == 'hellinger':
        hlg = np.sqrt(1 - np.sum(np.sqrt(x * y)))
        return 0 if np.isnan(hlg) else hlg
    elif method == 'wasserstein':
        return np.sum(np.abs(np.cumsum(x) - np.cumsum(y)))
    elif method == 'total_variation':
        return 0.5 * np.sum(np.abs(x - y))
    elif method == 'pearson':
        pearson_corr = np.corrcoef(x, y)[0, 1]
        return 1 - (pearson_corr + 1) / 2
    elif method == 'chi-square':
        return np.sum((x - y) ** 2 / (x + y + 1e-10))  # 避免除以零
    elif method == 'energy':
        return 2 * np.sum(x * y) - np.sum(x ** 2) - np.sum(y ** 2)
    elif method == 'chebyshev':
        return chebyshev(x, y)
    elif method == 'sorensen_dice':
        sorensen_dice = 2 * np.sum(x * y) / (np.sum(x ** 2) + np.sum(y ** 2))
        return 1 - sorensen_dice


def molecular_distance(comp_targs, dise_targs):
    node2set_SPLs = []  # node to set shortest path lengths
    for t in comp_targs:
        node2node_SPLs = []
        for s in dise_targs:
            node2node_SPLs.append(PPI_DIST_MAT[PPI_DIST_MAT_NODE2ID[t]][PPI_DIST_MAT_NODE2ID[s]])
        node2set_SPLs.append(min(node2node_SPLs))
    distance = np.mean(node2set_SPLs)
    return distance


def node_sets_distance(comp_targs, dise_targs, dist_method: str=None):
    if dist_method == 'jaccard':
        return 1 - len(np.intersect1d(comp_targs, dise_targs)) / len(np.union1d(comp_targs, dise_targs))
    elif dist_method == 'molecular':
        return molecular_distance(comp_targs, dise_targs)
    else:
        raise ValueError(f"不支持的节点集距离：{dist_method}")


################## 计算D2C和D2H的ranks和metrics ##################
# 全局声明，用于子进程存放数据
global_net = None
global_pmap = None

def cal_D2C_rank_init(net, pmap):
    global global_net, global_pmap
    global_net = net
    global_pmap = pmap

def cal_D2C_rank_func1(sub_d, worker, dist_method, masks, cols, verbose):
    worker_log(verbose=verbose, worker=worker, msg=f"running {len(sub_d)} diseases...")
    df = pd.DataFrame(columns=cols)
    
    dise_list = [d for d in sub_d]
    comp_list = [c for c in global_net.nodes_of_type('Compound') if c in global_net.node_info['Compound'].index]
    if dist_method == 'gnn':
        dise_pmaps = [global_pmap[d] for d in dise_list]
        comp_pmaps = [global_pmap[c] for c in comp_list]
    elif dist_method in ['random', 'molecular', 'jaccard']:
        dise_pmaps = None
        comp_pmaps = None
    else:
        dise_pmaps = [global_pmap[d][masks['Target_&_GO']] for d in dise_list]
        comp_pmaps = [global_pmap[c][masks['Target_&_GO']] for c in comp_list]
    
    # heat diffusion时用这个
    # dise_pmaps = [pmap[d] for d in dise_list]
    # comp_pmaps = [pmap[c] for c in comp_list]

    if dist_method == 'random':
        distances = np.random.random((len(dise_list), len(comp_list)))
        for i in range(len(dise_list)):
            for j in range(len(comp_list)):
                df.loc[len(df)] = [dise_list[i], comp_list[j], distances[i, j]]
            worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
    elif dist_method in ['jaccard', 'molecular']:
        for i in range(len(dise_list)) if not verbose else tqdm(range(len(dise_list)), desc=f'worker {worker}'):
            for j in range(len(comp_list)):
                d_targs = getNeighborNodes(graph=global_net.effect_graph, node=dise_list[i], neighbor_type='Target')
                c_targs = getNeighborNodes(graph=global_net.effect_graph, node=comp_list[j], neighbor_type='Target')
                df.loc[len(df)] = [
                    dise_list[i], comp_list[j],
                    node_sets_distance(d_targs, c_targs, dist_method)
                ]
            worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
    elif dist_method == 'gnn':
        for i in range(len(dise_list)) if not verbose else tqdm(range(len(dise_list)), desc=f'worker {worker}'):
            for j in range(len(comp_list)):
                df.loc[len(df)] = [
                    dise_list[i], comp_list[j], 1 - (dise_pmaps[i] * comp_pmaps[j]).sum(axis=-1)
                ]
    elif dist_method in ['l1', 'chebyshev', 'canberra', 'js', 'chi-square', 'bhattacharyya', 'hellinger', 'wasserstein', 'total_variation']:
        for i in range(len(dise_pmaps)) if not verbose else tqdm(range(len(dise_pmaps)), desc=f'worker {worker}'):
            for j in range(len(comp_pmaps)):
                # x=dise_pmaps[i]
                # y=comp_pmaps[j]
                # mask = np.logical_and(x != 0, y != 0)
                # df.loc[len(df)] = [
                #     dise_list[i], comp_list[j], get_distance(x[mask], y[mask], dist_method)
                # ]
                df.loc[len(df)] = [
                    dise_list[i], comp_list[j], distance_1d(dise_pmaps[i], comp_pmaps[j], dist_method)
                ]
            worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
    else:
        dist_mat = vectorized_distance_matrix(np.array(dise_pmaps), np.array(comp_pmaps), dist_method)
        for i in range(len(dise_pmaps)):
            for j in range(len(comp_pmaps)):
                df.loc[len(df)] = [dise_list[i], comp_list[j], dist_mat[i, j]]
            worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
    worker_log(verbose=verbose, worker=worker, msg="Done.")
    
    # q.put(df)
    return df

def cal_D2C_rank(
        net: MHNetwork,
        pmap: dict = None,
        dise_list: list = None,
        dist_method: str = 'cosine',
        n_proc: int = 32,
        verbose: bool = False,
    ):
    """_summary_

    Args:
        pmap (dict): _description_
        net (MHNetwork): _description_
        n_proc (int, optional): _description_. Defaults to 64.
        verbose (bool, optional): _description_. Defaults to False.
        dist_method (str, optional): _description_. Defaults to 'cosine'.

    Returns:
        dict: D2C_rank
                                        |  distance  |  rank  |  positive  |
        |  DisGENet_id  |  Compound_id  |            |        |            |
        |    C000001    |    COMP0001   |   0.1234   |   12   |     1.0    |
        |      ...      |       ...     |     ...    |   ...  |     ...    |
    """
    run_log(log_to_file=False)

    if dise_list is None:
        dise_list = POS_DISEASES
    elif isinstance(dise_list, str):
        dise_list = [dise_list]
    diseases = np.array([d for d in dise_list if d in net.node_info['Disease'].index])
    # pbar = tqdm(diseases)
    masks = net.eg_node_type_masks
    cols = ['DisGENet_id', 'Compound_id', 'distance']


    def func(diseases: NDArray, func):
        tasks = []
        for i in range(min(n_proc, len(diseases))):
            sub_diseases = diseases[range(i, len(diseases), n_proc)]
            worker = f"{i:0{len(str(n_proc))}}"  # 子进程编号
            tasks.append((sub_diseases, worker, dist_method, masks, cols, verbose))
        
        with mp.Pool(processes=n_proc, initializer=cal_D2C_rank_init, initargs=(net, pmap), maxtasksperchild=1) as pool:
            try:
                df_list = pool.starmap(func, tasks)
            except KeyboardInterrupt:
                print("检测到中断，正在终止进程池...")
                raise
            finally:
                pool.terminate()
                pool.join()

        
        if not df_list:
            return pd.DataFrame()
        
        rank = pd.concat(df_list, axis=0)

        rank['rank'] = rank.groupby('DisGENet_id')['distance'].rank()  # 距离数值越小，排名越高，rank越小
        rank = pd.merge(rank, POS_PAIRS, on=['DisGENet_id', 'Compound_id'], how='left').drop(["Herb_id"], axis=1)
        rank = rank.set_index(['DisGENet_id', 'Compound_id'])
        rank.loc[rank["positive"] != 1, 'positive'] = 0
        return rank

    logging.info("generating %s D2C_rank dataframe..." % dist_method)
    # D2C_rank = func2(diseases=diseases, func=func1)
    D2C_rank = func(diseases=diseases, func=cal_D2C_rank_func1)
    logging.info("%s D2C_rank dataframe generated..." % dist_method)
    return D2C_rank

def get_D2C_rank(
        net: MHNetwork,
        dist_method: str,
        param: str,
        pmap: dict = None,
    ):
    """
        1. 先根据dist_method确定如何获取param_str\n
        2. 再尝试根据param_str读取保存好的D2C_rank,若不存在,则计算\n
        3. 若dist_method为pmap距离函数,则param必须要有,用于从指定路径读取D2C_rank或保存至指定路径\n
        4. pmap和net可以输入,如果没输入,会自动读取
    """
    if dist_method in ['random', 'molecular', 'jaccard']:
        param_str = 'none_pmap_methods'
    else:
        param_str = getParamStr(param)
    print(f"getting D2C_rank of param '{param_str}', dist_method '{dist_method}'...", end=' ')
    fp = OUT_PATH + f'ranks/{len(net.all_nodes)}/{param_str}/D2C/'
    fn = f'{dist_method}.csv'
    if os.path.exists(fp+fn):
        print("Reading Complete!")
        return pd.read_csv(fp+fn).set_index(['DisGENet_id', 'Compound_id'])
    if dist_method not in ['random', 'molecular', 'jaccard']:
        print('dist_method')
        if pmap is None:
            pmap = loadPmap(net, param)
    D2C_rank = cal_D2C_rank(
        pmap=pmap, net=net, dist_method=dist_method,
        n_proc=64, verbose=False,
    )
    os.makedirs(fp, exist_ok=True)
    D2C_rank.to_csv(fp+fn)
    print("Calculating and Saving Complete!")
    return D2C_rank


def top_Ms_in_random_Ns(
        all_dists, Ns: int | list[int], Ms: int | list[int]
    ) -> dict[int, dict[int, tuple[float, float]]]:
    """随机采样n个距离1000次，统计前m个距离的均值，计算均值的均值与方差

    Args:
        all_dists (_type_): 用于随机采样的距离集合
        Ns (int | list[int]): 随机采样的距离数量，n
        Ms (int | list[int]): n个采样距离中的前m个，计算平均距离

    Returns:
        dict: avgs_stds. avgs_stds[n][m] = (avg, std)
    """
    if isinstance(Ns, int):
        Ns = [Ns]
    if isinstance(Ms, int):
        Ms = [Ms]
    
    avgs_stds = {n: {} for n in Ns}
    
    # 在（特定疾病的）所有距离中随机采样max(n_random)个距离，1000次
    # 
    random_sampled = np.reshape(np.random.choice(all_dists, 1000*max(Ns)), (1000, -1))
    for n in Ns:
        random_n = random_sampled[:, :n]  # (1000, n)
        for m in Ms:
            top_m_in_random_n = np.sort(random_n, axis=1)[:, :m]  # (1000, m)
            top_m_in_random_n = np.mean(top_m_in_random_n, axis=1)  # (1000,)
            avg = float(np.mean(top_m_in_random_n))
            std = float(np.std(top_m_in_random_n))
            avgs_stds[n][m] = (avg, std)
    return avgs_stds


def H2D_MinN(D2C_rank, net, herbs=None, dises=None, n_comps=1) -> dict[str, dict[int, dict[int, tuple[float, float]]]]:
    D2C_rank = D2C_rank.reset_index()
    
    if herbs is None:
        herbs = [h for h in net.nodes_of_type('Herb')]
    elif isinstance(dises, str):
        dises = [dises]
    
    if dises is None:
        dises = set(D2C_rank['DisGENet_id'])
    elif isinstance(dises, str):
        dises = [dises]
    
    n_herb_comps = {h: len(list(net.herb_graph.neighbors(h))) for h in herbs}
    
    result = {}
    for d in dises:
        result.update({
            d: top_Ms_in_random_Ns(
                D2C_rank[D2C_rank['DisGENet_id']==d]['distance'],
                n_herb_comps.values(),
                n_comps
            )
        })
    return result


def get_herb_random_sampled_top_n_mean_distances(D2C_rank: pd.DataFrame) -> dict:
    """随机采样统计均值与方差，用于计算z_score

    Args:
        D2C_rank (_type_): _description_
        disease_id (_type_): _description_

    Returns:
        dict: _description_
    """
    dises = list(set(D2C_rank.reset_index()['DisGENet_id']))
    
    # result[disease_id][n_herb_neighbors][n_top_comps][0]
    # result[disease_id][n_herb_neighbors][n_top_comps][1]


def cal_avgs_stds(D2C_rank: pd.DataFrame, net: MHNetwork, Ms: list=[1,4,16,64,256]):
    D2C_rank = D2C_rank.reset_index().set_index(['DisGENet_id', 'Compound_id'])
    d2c = D2C_rank.index
    avgs_stds = {}
    
    for d in tqdm(set(D2C_rank.reset_index()['DisGENet_id']), desc='generating avgs_stds'):
        dise_dists = D2C_rank.loc[d]['distance']
        Ns = set()
        for h in net.nodes_of_type('Herb'):
            n_herb_comps = len([c for c in net.herb_graph.neighbors(h) if (d,c) in d2c])
            Ns.add(n_herb_comps)
            # avgs_stds.update({d: top_Ms_in_random_Ns(
            #         all_dists=dise_dists,
            #         Ns=[n_herb_comps],
            #         Ms=[ceil(0.1*n_herb_comps)]
            # )})
        avgs_stds.update({d: top_Ms_in_random_Ns(all_dists=dise_dists, Ns=Ns, Ms=Ms)})
    return avgs_stds  # avgs_stds[dise_id][n][m] = (avg, std)


def cal_dist_avgs_stds(D2C_rank: pd.DataFrame, dise_id: str, N: int, M: int):
    D2C_rank = D2C_rank.reset_index().set_index(['DisGENet_id', 'Compound_id'])
    null_top_m_means = []
    for _ in range(1000):
        df = D2C_rank.loc[dise_id].sample(N).sort_values('distance')
        null_top_m_means.append(np.mean(df['distance'][:M]))
    return np.mean(null_top_m_means), np.std(null_top_m_means)


def cal_pmap_avgs_stds(pmap: dict, net: MHNetwork, dist_method: str, D2C_rank: pd.DataFrame, dise_id: str, N: int, M: int):
    TG_mask = net.eg_node_type_masks['Target_&_GO']
    dise_masked_pmap = pmap[dise_id][TG_mask]
    df = D2C_rank.reset_index().set_index(['DisGENet_id', 'Compound_id']).loc[dise_id]
    comps = np.array(list(df.index))
    all_indices = np.random.randint(0, len(comps), size=(1000, N))  # 随机采样的索引(1000, N)
    
    null_top_m_means = []
    if M >= N:
        random_sampled_comps = comps[all_indices]
        for i in range(1000):
            row_comps = random_sampled_comps[i]
            pmap_mean = np.mean([pmap[c] for c in row_comps], axis=0)
            dist = distance_1d(dise_masked_pmap, pmap_mean[TG_mask], dist_method)
            null_top_m_means.append(dist)
    else:
        dists = np.array(list(df['distance']))
        all_dists = dists[all_indices]  # 随机采样的距离(1000, N)
        min_indices = np.argpartition(all_dists, M, axis=1)[:, :M]
        for i in range(1000):
            row_indices = all_indices[i]
            row_min_indices = min_indices[i]
            min_comps = [comps[row_indices[j]] for j in row_min_indices]
            pmap_mean = np.mean([pmap[c] for c in min_comps], axis=0)
            dist = distance_1d(dise_masked_pmap, pmap_mean[TG_mask], dist_method)
            null_top_m_means.append(dist)
    return np.mean(null_top_m_means), np.std(null_top_m_means)

# mp.pool()计算D2H_ranks
global_d2c_rank = None

def cal_D2H_rank_init(net, d2c_rank):
    global global_net, global_d2c_rank
    global_net = net
    global_d2c_rank = d2c_rank

def cal_D2H_rank_func1(sub_d, worker, dist_method, cal_unzscored, Ms, Ks, verbose):
    worker_log(verbose=verbose, worker=worker, msg=f"running {len(sub_d)} diseases...")
    
    # 结果收集列表
    results_list = []
    
    # 提前获取 Herb 列表
    herb_list = global_net.nodes_of_type('Herb')
    
    for i, d in enumerate(sub_d):
        try:
            # 从全局加载的 D2C_rank 中提取当前疾病的数据
            d_rank_data = global_d2c_rank.loc[d]
        except KeyError:
            continue
            
        d_dists = np.array(d_rank_data['distance'])
        
        # 遍历药材
        for h in herb_list:
            # 找到在该疾病中有距离信息的成分
            herb_comps = [c for c in global_net.herb_graph.neighbors(h) if c in d_rank_data.index]
            
            # 分支 1: Jaccard / Molecular (基于原始定义)
            if dist_method in ['jaccard']:  # ['jaccard', 'molecular']
                d_targs = getNeighborNodes(graph=global_net.effect_graph, node=d, neighbor_type='Target')
                h_targs = []
                for c in herb_comps:
                    h_targs += getNeighborNodes(graph=global_net.effect_graph, node=c, neighbor_type='Target')
                
                d2h_dist = node_sets_distance(d_targs, h_targs, dist_method)
                results_list.append({
                    'DisGENet_id': d, 'Herb_id': h, 'n_nb_comps': len(herb_comps),
                    '100pct_dist': d2h_dist
                })
                continue

            # 分支 2: 基于成分距离聚合 (Z-Score 方法)
            if not herb_comps:
                continue

            # 提取并排序当前药材成分的预测距离
            sorted_herb_comp_dists = np.sort([d_rank_data.loc[c]['distance'] for c in herb_comps])
            n_comps = len(herb_comps)
            
            # 构造结果字典
            row = {'DisGENet_id': d, 'Herb_id': h, 'n_nb_comps': n_comps}
            
            # 预计算采样背景 (加速计算)
            indices = np.random.randint(0, len(d_dists), size=(1000, n_comps))
            null_dists_sorted = np.sort(d_dists[indices], axis=1)

            # 计算 Ms (Top-M)
            for m in Ms:
                m_val = min(m, n_comps)
                real_mean = np.mean(sorted_herb_comp_dists[:m_val])
                null_means = np.mean(null_dists_sorted[:, :m_val], axis=1)
                dist_avg, dist_std = np.mean(null_means), np.std(null_means)
                
                if cal_unzscored: row[f'{m}_dist'] = real_mean
                row[f'{m}_zscr'] = (real_mean - dist_avg) / (dist_std + 1e-15)

            # 计算 Ks (Top-K%)
            for k in Ks:
                n_top_k = ceil(k / 100 * n_comps)
                real_mean = np.mean(sorted_herb_comp_dists[:n_top_k])
                null_means = np.mean(null_dists_sorted[:, :n_top_k], axis=1)
                dist_avg, dist_std = np.mean(null_means), np.std(null_means)
                
                if cal_unzscored: row[f'{k}pct_dist'] = real_mean
                row[f'{k}pct_zscr'] = (real_mean - dist_avg) / (dist_std + 1e-15)
            
            results_list.append(row)
            
        worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
    
    worker_log(verbose=verbose, worker=worker, msg="Done.")
    return pd.DataFrame(results_list)


def cal_D2H_ranks(
        net: MHNetwork,
        D2C_rank: pd.DataFrame,
        dist_method: str,
        cal_unzscored: bool = False,
        dise_list: list = None,
        n_proc: int = 32,
        verbose: bool = False,
        Ms: list = [],
        Ks: list = [],
    ) -> pd.DataFrame:
    # run_log(log_to_file=False)
    D2C_rank = D2C_rank.reset_index().set_index(['DisGENet_id', 'Compound_id'])

    if dise_list is None:
        # dise_list = POS_DISEASES  # 计算所有阳性对的疾病
        dise_list = POS_D2H_DISEASES  # 计算有药材阳性对的疾病，用于计算metrics
    elif isinstance(dise_list, str):
        dise_list = list(set(POS_D2H_DISEASES) | set([dise_list]))
    elif isinstance(dise_list, list):
        dise_list = list(set(POS_D2H_DISEASES) | set(dise_list))
    diseases = np.array([d for d in dise_list if d in net.node_info['Disease'].index])  # 同时存在于POS_DISEASES和pmap中的疾病

    cols = ['DisGENet_id', 'Herb_id', 'n_nb_comps']  # 'all_mean'其实就是top_100%，不用单独列出了
    if cal_unzscored:
        cols += [f'{n}_dist' for n in Ms] + [f'{k}pct_dist' for k in Ks]
    cols += [f'{n}_zscr' for n in Ms] + [f'{k}pct_zscr' for k in Ks]

    def func1(q, sub_d: NDArray, worker: str):
        worker_log(verbose=verbose, worker=worker, msg=f"running {len(sub_d)} diseases...")

        if dist_method in ['jaccard', 'molecular']:
            # ['DisGENet_id', 'Herb_id', 'n_nb_comps', 'all_herb_targ']
            df = pd.DataFrame(columns=cols[:3]+['100pct_dist'])
        else:
            df = pd.DataFrame(columns=cols)

        for i, d in enumerate(sub_d):
            d_rank: pd.DataFrame = D2C_rank.loc[d]
            d_dists = np.array(d_rank['distance'])
            
            for h in tqdm(net.nodes_of_type('Herb'), desc=f'worker {worker}, disease {i}') if verbose else net.nodes_of_type('Herb'):
                herb_comps = [c for c in net.herb_graph.neighbors(h) if c in d_rank.index]
                
                # ?: molecular方法和jaccard方法可不可以以原始定义扩展到药材层面？
                # ?: 如果可以，那就收集所有成分的靶点表示该药材，然后计算距离
                # ?: 如果不可以，那就与其他距离函数的方法一样，先算每个成分的距离，再算距离的平均值

                # NOTE: molecular方法是对于每个疾病靶点，取其对所有药材靶点的最短距离，然后取平均
                # NOTE: 这意味着，药材成分数量越多，药材靶点越多，molecular方法算出的距离就越小
                # NOTE: 极限情况就是药材成分非常多，选取到的成分的靶点就包含了所有疾病靶点，最终距离就是0
                # NOTE: 这种算法回导致，药材-疾病距离和
                # NOTE: 所以这个方法应用于两个靶点集样本数相差不大的时候，才比较准

                # ?: jaccard方法可以拓展到药材层面吗？
                # NOTE: 可以，计算选取到的所有化合物对应的靶点，计算这些靶点与疾病靶点的交并比
                if dist_method in ['molecular', 'jaccard']:
                    d_targs = getNeighborNodes(graph=net.effect_graph, node=d, neighbor_type='Target')
                    h_targs = []
                    for c in herb_comps:
                        h_targs += getNeighborNodes(graph=net.effect_graph, node=c, neighbor_type='Target')
                    d2h_dist = node_sets_distance(d_targs, h_targs, dist_method)
                    df.loc[len(df)] = [d, h, len(herb_comps), d2h_dist]
                    continue
                
                if herb_comps == []:  # 若药材中所有成分 对当前疾病均无距离信息，则跳过该药材
                    continue

                sorted_herb_comp_dists = np.sort([d_rank.loc[c]['distance'] for c in herb_comps])
                # NOTE: 药材距离有多种计算方法，哪种合适？
                # NOTE: 所有化合物取平均，等同于前100%个化合物，合并到前k%中，不再单独计算
                # NOTE: 前m个化合物，若不足，则取该药材所有化合物，距离平均、距离平均z-score，平均pmap距离（舍弃，会出现(1,-1)和(-1,1)取平均为(0,0)的情况，算起来也麻烦）
                # NOTE: 前k%个化合物，若不足，则取该药材所有化合物，距离平均、距离平均z-score，平均pmap距离（舍弃，理由同上）
                # 1.所有化合物距离取平均
                # all_mean = np.mean(sorted_herb_comp_dists)
                # 2.前m个化合物（若不足，则取所有len(herb_comps)个），距离平均，距离平均zscore，平均pmap距离
                top_dist_means = []  # 前m个距离的均值
                top_dist_zscrs = []  # 前m个距离的均值的Z-Score
                for m in Ms:
                    indices = np.random.randint(0, len(d_dists), size=(1000, len(herb_comps)))
                    null_dists = d_dists[indices]
                    null_top_dists = np.sort(null_dists, axis=1)[:, :m]
                    null_dist_means = np.mean(null_top_dists, axis=1)  # (1000, )
                    dist_avg, dist_std = np.mean(null_dist_means), np.std(null_dist_means)
                    
                    top_dist_mean = np.mean(sorted_herb_comp_dists[:m])
                    top_dist_zscr = (top_dist_mean - dist_avg) / dist_std
                    top_dist_means.append(top_dist_mean)
                    top_dist_zscrs.append(top_dist_zscr)
                # 3.前k%个化合物，距离平均，距离平均zscore，平均pmap距离
                top_pct_dist_means = []
                top_pct_dist_zscrs = []
                for k in Ks:
                    n_top_k_pct = ceil(k/100*len(herb_comps))  # k%药材化合物数量
                    indices = np.random.randint(0, len(d_dists), size=(1000, len(herb_comps)))
                    null_dists = d_dists[indices]
                    null_top_dists = np.sort(null_dists, axis=1)[:, :n_top_k_pct]
                    null_dist_means = np.mean(null_top_dists, axis=1)  # (1000, )
                    dist_avg, dist_std = np.mean(null_dist_means), np.std(null_dist_means)
                    
                    top_pct_dist_mean = np.mean(sorted_herb_comp_dists[:n_top_k_pct])
                    top_pct_dist_zscr = (top_pct_dist_mean - dist_avg) / dist_std
                    top_pct_dist_means.append(top_pct_dist_mean)
                    top_pct_dist_zscrs.append(top_pct_dist_zscr)
                
                # 将所有距离记录到一行中
                results = [d, h, len(herb_comps)]  # all_mean就是100%，不用单独列出了
                if cal_unzscored:
                    results += top_dist_means + top_pct_dist_means
                results += top_dist_zscrs + top_pct_dist_zscrs
                df.loc[len(df)] = results
            worker_log(verbose=verbose, worker=worker, msg=f"({i + 1}/{len(sub_d)})")
        worker_log(verbose=verbose, worker=worker, msg="Done.")
        q.put(df)

    def func2(diseases: NDArray, func):
        processes = []
        queue = mp.Manager().Queue()

        for i in range(min(n_proc, len(diseases))):
            sub_diseases = diseases[range(i, len(diseases), n_proc)]
            worker = f"{i:0{len(str(n_proc))}}"  # 子进程编号
            p = mp.Process(
                target=func, args=(queue, sub_diseases, worker))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        df_list = []
        while not queue.empty():
            df_list.append(queue.get())
        dists = pd.concat(df_list, axis=0)
        ranks = dists.copy()

        for c in dists.columns[3:]:  # 前三列是['DisGENet_id', 'Herb_id', 'n_nb_comps']
            ranks[c+'_rank'] = dists.groupby('DisGENet_id')[c].rank()  # 距离越小，排名越高，rank越小，越应该是阳性
        # ranks = ranks[list(ranks.columns[:3])+sorted(list(ranks.columns[3:]))]
        # dists = pd.merge(dists, POS_PAIRS, on=['DisGENet_id', 'Herb_id'], how='left').drop(["Compound_id"], axis=1)
        ranks = pd.merge(ranks, POS_PAIRS, on=['DisGENet_id', 'Herb_id'], how='left').drop(["Compound_id"], axis=1)
        ranks = ranks.set_index(['DisGENet_id', 'Herb_id'])
        ranks.loc[ranks["positive"] != 1, 'positive'] = 0

        return dists, ranks

    def func_pool(diseases: NDArray, func):
        tasks = []
        for i in range(min(n_proc, len(diseases))):
            sub_diseases = diseases[range(i, len(diseases), n_proc)]
            worker = f"{i:0{len(str(n_proc))}}"
            tasks.append((sub_diseases, worker, dist_method, cal_unzscored, Ms, Ks, verbose))
        
        # 使用 Pool 并初始化全局变量
        with mp.Pool(processes=n_proc, initializer=cal_D2H_rank_init, initargs=(net, D2C_rank), maxtasksperchild=1) as pool:
            try:
                df_list = pool.starmap(func, tasks)
            except KeyboardInterrupt:
                print("检测到中断，正在终止进程池...")
                pool.terminate()
                raise
            finally:
                pool.close()
                pool.join()

        if not df_list or all(df.empty for df in df_list):
            return pd.DataFrame()
        
        dists = pd.concat(df_list, axis=0)
        
        # 计算排名
        ranks = dists.copy()
        meta_cols = ['DisGENet_id', 'Herb_id', 'n_nb_comps']
        score_cols = [c for c in dists.columns if c not in meta_cols]

        for c in score_cols:
            # 距离越小，rank越小，排名越靠前
            ranks[c+'_rank'] = dists.groupby('DisGENet_id')[c].rank(ascending=True)

        # 合并阳性标签
        ranks = pd.merge(ranks, POS_PAIRS, on=['DisGENet_id', 'Herb_id'], how='left').drop(["Compound_id"], axis=1, errors='ignore')
        ranks = ranks.set_index(['DisGENet_id', 'Herb_id'])
        ranks['positive'] = ranks['positive'].fillna(0).astype(int)
        
        return ranks

    logging.info("generating D2H_ranks dataframe...")
    # D2H_dists, D2H_ranks = func2(diseases=diseases, func=func1)
    D2H_ranks = func_pool(diseases=diseases, func=cal_D2H_rank_func1)
    logging.info("D2H_ranks dataframe generated...")
    return D2H_ranks


def get_D2H_ranks(
        D2C_rank: pd.DataFrame,
        net: MHNetwork,
        dist_method: str,
        param: str,
    ):
    if dist_method in ['random', 'molecular', 'jaccard']:
        param_str = 'none_pmap_methods'
        Ks = [100]
    else:
        param_str = getParamStr(param)
        Ks = [10,20,30,40,50,60,70,80,90,100]
    print(f"getting D2H_ranks of param '{param_str}', dist_method '{dist_method}'...", end=' ')
    fp = OUT_PATH + f'ranks/{len(net.all_nodes)}/{param_str}/D2H/'
    fn = f"{dist_method}.csv"
    if os.path.exists(fp+fn):
        print(f"Reading Complete!")
        return pd.read_csv(fp+fn).set_index(['DisGENet_id', 'Herb_id'])
    D2H_ranks = cal_D2H_ranks(
        net=net, D2C_rank=D2C_rank,
        dist_method=dist_method, cal_unzscored=True,
        n_proc=64, verbose=False, Ks=Ks
    )  # 函数输出的df的index已经定义好了
    os.makedirs(fp, exist_ok=True)
    D2H_ranks.to_csv(fp+fn)
    print("Calculating and Saving Complete!")
    return D2H_ranks


def get_ranks(
        pmap: dict,
        net: MHNetwork,
        n_proc: int = 32,
        verbose: bool = False,
        n_comp: int = 8,
        dist_method: str = 'cosine',
):
    """
    药材-疾病的距离计算为：该药材对应的所有成分中，与该疾病距离最近的n_comp个成分的距离平均值
    按疾病分组计算rank，rank越大，药材/成分-疾病的距离越近

    return: 返回两个pd.DataFrame，形如：\n
        |             |             | distance | rank | positive |\n
        | DisGENet_id | Compound_id |          |      |          |\n
        |   C0001144  |   COMP3043  |  0.2345  |   1  |     0    |\n
        |   C0001144  |   COMP3044  |  0.2345  |   2  |     1    |\n
        ...

        |             |         |   distances   | mDist@topn | rank | positive |\n
        | DisGENet_id | Herb_id |               |            |      |          |\n
        |   C0001144  | HERB001 | [0.1,0.2,...] |   0.5524   |   1  |     0    |\n
        |   C0001144  | HERB002 | [0.1,0.2,...] |   0.6593   |   2  |     1    |\n
        ...
    """

    D2C_rank = cal_D2C_rank(pmap, net, n_proc=n_proc, verbose=verbose, dist_method=dist_method)
    D2H_rank = cal_D2H_ranks(pmap, net, D2C_rank, n_proc=n_proc, verbose=verbose, n_comp=n_comp)
    logging.info("ranks generated!")

    return D2C_rank, D2H_rank


################## 计算MeAUC、mPrecision、mRecall ##################

def cal_D2C_rank_metrics(D2C_rank: pd.DataFrame, show_tqdm=False) -> tuple:
    """
        mAUC: 随机抽取一对真实正负样本，正样本的预测排序高于负样本的概率
        mAP: 平均精确率
        mR@1%: 前1%排序中正样本占所有正样本
        mR@5%: 前5%排序中正样本占所有正样本
        mR@10%: 前10%排序中正样本占所有正样本
        mR@20%: 前20%排序中正样本占所有正样本
    """
    # 因为会使用分割后的训练、验证、测试集，需要重排rank，否则metric计算会出错
    D2C_rank = D2C_rank.reset_index()
    D2C_rank['rank'] = D2C_rank.groupby('DisGENet_id')['distance'].rank()
    D2C_rank = D2C_rank.set_index('DisGENet_id')
    aucs, precisions = [], []
    recall_at_1, recall_at_5, recall_at_10, recall_at_20 = [], [], [], []
    pos_pair_diseases = list(set(D2C_rank.query("positive == 1").index.get_level_values('DisGENet_id')))
    for disease in tqdm(pos_pair_diseases) if show_tqdm else pos_pair_diseases:
        data = D2C_rank.loc[disease]
        n_all = len(data)
        pos_rank_sum = (data.query("positive == 1")['rank']).sum()
        n_pos = len(data.query("positive == 1"))

        # AUC
        aucs.append(
            1 - (pos_rank_sum - (n_pos * (1 + n_pos) / 2)) / (n_pos * (n_all - n_pos)))
        # aucs.append(
        #     (n_all*n_pos - pos_rank_sum - n_pos*(n_pos-1)/2) / (n_pos * (n_all - n_pos)))

        # PRECISION
        # rank要取负值，因为average_precision_score()默认数值越大约阳性
        precisions.append(sklearn.metrics.average_precision_score(data['positive'], -data['rank']))

        # RECALL
        # recalls.append(len(data.query("positive==1 and rank>=@num-99")) / positive_num)
        recall_at_1.append(len(data[(data['positive']==1) & (data['rank']<n_all*0.01)]) / n_pos)
        recall_at_5.append(len(data[(data['positive']==1) & (data['rank']<n_all*0.05)]) / n_pos)
        recall_at_10.append(len(data[(data['positive']==1) & (data['rank']<n_all*0.1)]) / n_pos)
        recall_at_20.append(len(data[(data['positive']==1) & (data['rank']<n_all*0.2)]) / n_pos)
        
    # return data
    return {
        'mAUC': round(float(np.array(aucs).mean()), 4),
        'mAP': round(float(np.array(precisions).mean()), 4),
        # 'mR@1%': round(float(np.array(recall_at_1).mean()), 4),
        # 'mR@5%': round(float(np.array(recall_at_5).mean()), 4),
        # 'mR@10%': round(float(np.array(recall_at_10).mean()), 4),
        'mR@20%': round(float(np.array(recall_at_20).mean()), 4),
    }


def cal_D2H_ranks_metrics(D2H_rank, dist_method, show_tqdm=False):
    D2H_rank = D2H_rank.reset_index().set_index('DisGENet_id')
    pos_pair_diseases = list(set(D2H_rank.query("positive == 1").index.get_level_values('DisGENet_id')))
    metrics = {}
    for col in D2H_rank.columns[1:-1]:
        aucs, precisions = [], []
        recall_at_1, recall_at_5, recall_at_10, recall_at_20 = [], [], [], []
        # 仅统计rank列
        if not col.endswith('_rank'):
            continue
        # 若方法为无pmap的方法，则pmap列的metrics为nan
        if dist_method in ['random', 'molecular', 'jaccard'] and 'pmap' in col:
            metrics.update({col[:-5]: (np.nan,)*6})
            continue
        for disease in tqdm(pos_pair_diseases) if show_tqdm else pos_pair_diseases:
            data = D2H_rank.loc[disease]

            # AUC
            n_all = len(data)
            pos_rank_sum = np.array(data.query("positive == 1")[col]).sum()
            n_pos = len(data.query("positive == 1"))
            aucs.append(
                1 - (pos_rank_sum - (n_pos * (1 + n_pos) / 2)) / (n_pos * (n_all - n_pos)))
            # aucs.append(
            #     (n_all*n_pos - pos_rank_sum - n_pos*(n_pos-1)/2) / (n_pos * (n_all - n_pos)))

            # PRECISION
            # rank要取负值，因为average_precision_score()默认数值越大约阳性
            precisions.append(sklearn.metrics.average_precision_score(data['positive'], -data[col]))

            # RECALL
            # recalls.append(len(data[(data['positive']==1) & data[c]>=num-49]) / positive_num)
            recall_at_1.append(len(data[(data['positive']==1) & (data[col]<n_all*0.01)]) / n_pos)
            recall_at_5.append(len(data[(data['positive']==1) & (data[col]<n_all*0.05)]) / n_pos)
            recall_at_10.append(len(data[(data['positive']==1) & (data[col]<n_all*0.1)]) / n_pos)
            recall_at_20.append(len(data[(data['positive']==1) & (data[col]<n_all*0.2)]) / n_pos)
            
        metrics.update({col[:-5]: {
            'mAUC': round(float(np.array(aucs).mean()), 4),
            'mAP': round(float(np.array(precisions).mean()), 4),
            # 'mR@1%': round(float(np.array(recall_at_1).mean()), 4),
            # 'mR@5%': round(float(np.array(recall_at_5).mean()), 4),
            # 'mR@10%': round(float(np.array(recall_at_10).mean()), 4),
            'mR@20%': round(float(np.array(recall_at_20).mean()), 4),
        }})

    return metrics


def cal_fold_D2C_rank_metrics(D2C_rank, folds):
    """
        利用提前分好的fold计算metrics
        D2C_rank: pd.DataFrame(index=['DisGENet_id', 'Compound_id'], columns=['distance', 'rank', 'positive'])
        folds: [{'train': pd.DataFrame(columns=['DisGENet_id', 'Compound_id']), 'val': ...}, ...]
    """
    print(f'Calculating {len(folds)} fold D2C_rank metrics...', end=' ')
    sets = ['train', 'val']
    # metrics = ['mAUC', 'mAP', 'mR@1%', 'mR@5%','mR@10%','mR@20%']
    metrics = ['mAUC', 'mAP','mR@20%']
    results = {s: {m:[] for m in metrics} for s in sets}
    D2C_rank = D2C_rank.query('DisGENet_id in @POS_D2C_DISEASES')
    negs = D2C_rank[D2C_rank['positive']==0].index
    for i, fold in enumerate(folds):
        # print(f"fold {i+1}")

        for _set in ['train', 'val']:
            df = D2C_rank.loc[negs.union(fold[_set].set_index(['DisGENet_id', 'Compound_id']).index)]
            rank_metrics = cal_D2C_rank_metrics(df, show_tqdm=False)

            for m in metrics:
                results[_set][m].append(rank_metrics[m])
    
    CIs = {s: {m:[] for m in metrics} for s in sets}
    for s in sets:
        for m in metrics:
            values = results[s][m]
            CIs[s][m] = f"{np.mean(values):.4f}±{1.96*np.std(values)/np.sqrt(len(values)):.4f}"
    
    print('Complete!')

    return results, CIs



if __name__ == '__main__':
    get_D2H_ranks(dist_method='random')
    
    
