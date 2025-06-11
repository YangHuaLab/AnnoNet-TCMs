import os
import re
import sys
import networkx as nx
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from MHNetwork import MHNetwork
from const import NETWORK_FP, OPTIMAL_PMAP


# 返回指定单个节点的指定若干类型的邻居节点
def getNeighborNodes(graph: nx.Graph, node: str, neighbor_type: str | list[str] = ["Target", "GO"]):
    """return neighbor nodes of specific node type(s)"""
    if type(neighbor_type) == str:
        return [n for n in graph[node] if graph.nodes[n]["type"] == neighbor_type]
    elif type(neighbor_type) == list:
        return [n for n in graph[node] if graph.nodes[n]["type"] in neighbor_type]


# 输入节点类型字符串，返回节点类型缩写字符
def getTypeAbbr(t: str) -> str:
    node_type_abbr = {
        "Herb": "H",
        "Compound": "C",
        "Target": "T",
        "GO": "G",
        "Disease": "D"
    }
    return node_type_abbr[t]


# 输入节点列表，返回节点类型缩写字符串
def getPathAbbr(path: list[str]) -> str:
    res = ""
    for node in path:
        res = res + getTypeAbbr(MHNetwork.getNodeType(node))
    return res


def getParamStr(i: str | list) -> str:
    if isinstance(i, str):
        p = ".*(C.*_T.*_D.*_G.*_L.{4}).*"
        g = re.match(p, i).groups()
        return g[0]
    if isinstance(i, list):
        assert len(i) == 5
        return f"C{i[0]:.2f}_T{i[1]:.2f}_D{i[2]:.2f}_G{i[3]:.2f}_L{i[4]:.2f}"


def getParamList(i: str | dict) -> list:
    if type(i) == str:
        p = ".*C(.*)_T(.*)_D(.*)_G(.*)_L(.{4}).*"
        g = re.match(p, i).groups()
        return list(np.float64(g))
    if type(i) == dict:
        return [list(i['node_weights'].values()), i['LAMBDA']]


def getParamDict(i: str | list) -> dict:
    if type(i) == str:
        l = getParamList(i)
    if type(i) == list:
        l = i
    return {
        'node_weights': {
            'Compound': l[0],
            'Target': l[1],
            'Disease': l[2],
            'GO': l[3],
        },
        'LAMBDA': l[4],
    }


def loadPmap(pmap_fp: str = OPTIMAL_PMAP) -> dict:
    import pickle
    print(f"Loading pmap {pmap_fp}...", end='', flush=True)
    with open(pmap_fp, 'rb') as f:
        pmap = pickle.load(f)
    if len(pmap.keys()) != 5084:
        os.remove(pmap_fp)
        raise ValueError(f"Warning: Pmap has {len(pmap.keys())} keys, less than expected! Deleted!")
    print("Complete!", flush=True)
    return pmap


def loadMHNet(network_fp: str = NETWORK_FP, force_generate: bool = False) -> MHNetwork:
    import pickle
    if os.path.exists(network_fp) and not force_generate:
        print(f"Loading MHNetwork {network_fp}...", end='', flush=True)
        with open(network_fp, 'rb') as f:
            net = pickle.load(f)
        print("Complete!", flush=True)
    else:
        net = MHNetwork()
        with open(network_fp, 'wb') as f:
            pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)

    return net


if __name__ == "__main__":
    loadMHNet()
    print(getParamStr([1, 2, 3, 4, 5]))
