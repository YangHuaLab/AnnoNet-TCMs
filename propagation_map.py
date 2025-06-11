import logging
import os
import sys
from os.path import dirname, abspath

import networkx as nx
import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike
from scipy import sparse

sys.path.append(dirname(dirname(abspath(__file__))))
from const import NODE_WEIGHTS
from PMRWRH import PMRWRH
from logger import worker_log


def reconstruct_refine_M(adj_matrix, S, J):
    adj_new = adj_matrix.copy()
    adj_new[:, S] = 0
    adj_new[J, :] = 0
    adj_new = sparse.csr_matrix(adj_new)
    D = scipy.array(adj_new.sum(axis=1)).flatten()
    D[D != 0] = 1.0 / D[D != 0]
    Q = sparse.spdiags(D.T, 0, *adj_new.shape, format='csr')
    adj_new = Q * adj_new
    return adj_new


# 以单个节点为起始，获取传播图
def get_propagation_map_from_node_id(
    adj_matrix: np.matrix,
    start_node: str,
    all_nodes: ArrayLike,
    mask: ArrayLike,
    herb_graph: nx.Graph,
    LAMBDA: float=0.1,
    EPSILON: float=1e-6,
    worker: int=None
) -> dict[str, ArrayLike]:
    """
    获取以node_id为起始点的传播图
    起始节点必须为药物或疾病节点

    :param adj_matrix: 所有节点的邻接矩阵
    :param LAMBDA: 参数λ
    :param EPSILON: 参数ε
    :param start_node: 起始节点id
    :param all_nodes: 所有节点id
    :param mask: 所有起始节点的掩码
    :param herb_graph: 药材-成分网络，当输入为药材节点时，以此网络获取该药材的关联成分，并以这些成分为起点计算传播图
    :return: 返回一个包含一个键值对的字典，键为起始点id，值为对应传播图
    """
    worker_log(worker=worker, msg=f"propagating from node {start_node}...")
    
    def calculate_metrics(id_list, ID):
        S = scipy.array(id_list) == ID  # 起始点编码，起始点为1，其余为0
        P0 = S  # 初始传播图，起始点为1，其余为0
        J = mask ^ S  # 除当前起始点外，其他候选起始点编码，符合条件的为True
        M = reconstruct_refine_M(adj_matrix, S, J)  # 更新后的
        return P0, S, J, M

    P0, S, J, M = calculate_metrics(all_nodes, start_node)
    propagation_map = PMRWRH(M, LAMBDA, EPSILON, P0, S, J)
    res = {start_node: propagation_map}
    
    return res


# 以若干节点为起始，获取若干传播图
def get_propagation_maps_from_nodes(
    adj_matrix: np.matrix,
    start_nodes: ArrayLike,
    all_nodes: ArrayLike,
    mask: ArrayLike,
    herb_graph: nx.Graph,
    LAMBDA: float=0.1,
    EPSILON: float=1e-6,
    worker: int=None
) -> dict[str, ArrayLike]:
    """
    输入起始节点id列表，串行

    :param adj_matrix:
    :param LAMBDA:
    :param EPSILON:
    :param start_nodes:
    :param all_nodes:
    :param mask:
    :param herb_graph:
    :return: 返回一个字典，键为输入的所有起始点id，值为对应传播图
    """
    propagation_maps = {}
    for i,node_id in enumerate(start_nodes):
        propagation_maps.update(
            get_propagation_map_from_node_id(
                adj_matrix=adj_matrix,
                start_node=node_id,
                all_nodes=all_nodes,
                mask=mask,
                herb_graph=herb_graph,
                LAMBDA=LAMBDA,
                EPSILON=EPSILON,
                worker=worker))
        digits = len(str(len(start_nodes)))
        progress = f"{(i+1):0{digits}}/{len(start_nodes)}"  # 如1/9，01/20，001/318
        worker_log(worker=worker, msg=f"node {node_id} propagation complete! ({progress})")
    worker_log(worker=worker, msg="nodes propagation complete!")
    return propagation_maps


# 以若干节点为起始，多进程获取若干传播图
def parallel_get_propagation_maps_from_nodes(
    adj_matrix: np.matrix,
    start_nodes: ArrayLike,
    all_nodes: ArrayLike,
    mask: ArrayLike,
    herb_graph: nx.Graph,
    LAMBDA: float=0.1,
    EPSILON: float=1e-6,
    n_procs: int=1
) -> dict[str, ArrayLike]:
    """
    并行计算以start_nodes为起始点，n_procs定义进程数量，默认为1

    :param adj_matrix:
    :param LAMBDA:
    :param EPSILON:
    :param start_nodes:
    :param all_nodes:
    :param mask:
    :param herb_graph:
    :param n_procs:
    :return: 返回一个字典，键为所有起始点，值为以某一起始点开始的传播图
    """
    # 若输入的起始点只有一个，则直接调用get_propagation_map_from_node_id
    if len(start_nodes) == 1:
        result = get_propagation_map_from_node_id(
            adj_matrix=adj_matrix,
            start_node=start_nodes[0],
            all_nodes=all_nodes,
            mask=mask,
            herb_graph=herb_graph,
            LAMBDA=LAMBDA,
            EPSILON=EPSILON,
            worker=None)
        return result

    # 起始点有多个
    # 若子进程数被设为1，则直接调用get_propagation_map_from_node_ids
    if n_procs == 1:
        pm = get_propagation_maps_from_nodes(
            adj_matrix=adj_matrix,
            start_nodes=start_nodes,
            all_nodes=all_nodes,
            mask=mask,
            herb_graph=herb_graph,
            LAMBDA=LAMBDA,
            EPSILON=EPSILON,
            worker=None)
        return pm
    
    # 若子进程数为大于1的整数，则将start_node_ids等分为n_procs份，每份一个进程进行计算
    elif n_procs > 1:
        def func(queue,node_ids,worker):
            """
                queue用于输出结果，node_ids为输入的起始节点，worker为当前进程的worker序号
            """
            worker_log(worker=worker, msg=f"pid:{os.getpid()}")
            try:
                result = get_propagation_maps_from_nodes(
                    adj_matrix=adj_matrix,
                    start_nodes=node_ids,
                    all_nodes=all_nodes,
                    mask=mask,
                    herb_graph=herb_graph,
                    LAMBDA=LAMBDA,
                    EPSILON=EPSILON,
                    worker=worker)
                queue.put(result)
                # os.kill(os.getpid(), signal.SIGTERM)
            except Exception as e:
                import traceback
                worker_log(level=logging.ERROR, worker=worker, msg=traceback.format_exc())

        import multiprocess as mp

        n_procs = min(n_procs, len(start_nodes))
        processes = []
        queue = mp.Manager().Queue()  # multiprocessing.Queue()会因未知原因导致子进程无法退出，必须用multiprocessing.Manager().Queue()
        propagation_maps = {}
        
        for i in range(n_procs):  # 循环生成子进程
            sub_start_node_ids = start_nodes[range(i, len(start_nodes), n_procs)]
            worker = f"{i:0{len(str(n_procs))}}"  # 子进程编号
            p = mp.Process(
                target=func,args=(queue,sub_start_node_ids,worker))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        
        while not queue.empty():
            res = queue.get()
            propagation_maps.update(res)

        logging.info("parallel propagation complete!")
        
        if len(propagation_maps.keys())<len(start_nodes):
            logging.warning(f"Generating less start nodes! ({len(propagation_maps.keys())} out of {len(start_nodes)})")
        
        return propagation_maps


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from MHNetwork import MHNetwork
    net = MHNetwork()
    param = {
        "node_weights": {
            "Compound": 1,
            "Target": 3,
            "Disease": 6,
            "GO": 0.5
        },
        "LAMBDA": 0.1,
    }
    net.set_effect_graph_adj_matrix(force=False, node_weights=param["node_weights"])
    net.set_random_walk_params(LAMBDA=param['LAMBDA'], EPSILON=1e-6)
    net.random_walk_on_effect_graph()