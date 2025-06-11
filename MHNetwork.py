import logging
import sys
import os
import pickle
import re

from datetime import datetime

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from const import NODE_FILES, NODE_TYPES, NODE_ID, NODE_WEIGHTS
from const import EDGE_FILES, EDGE_TYPES
from const import OUT_PATH, LOG_PATH
from propagation_map import parallel_get_propagation_maps_from_nodes
from logger import run_log


class MHNetwork:
    """
    初始化一个MHNetwork实例，实例包含节点信息、边信息，以及药材网络、药效网络。\n
    其中药效网络中包含节点类型、掩码。\n
    这之后，依次调用：
    1. self.set_effect_graph_adj_matrix()
       计算药效网络的邻接矩阵
    2. self.set_random_walk_params()
       设定随机游走的参数
    3. self.random_walk_on_effect_graph()
       在药效网络上运行随机游走\n
    

    """

    def __init__(self, force_cal_adj_matrix=False) -> None:

        self.force_cal_adj_matrix = force_cal_adj_matrix

        print("**************************************************")

        print("**  reading all node **.csv files...            **")
        self.node_info = self.load_node_info()
        self.all_nodes = np.array([idx for df in self.node_info.values() for idx in df.iloc[:, 0]])

        print("**  reading all edge **.csv files...            **")
        self.all_edges = self.load_edge_info()

        print("**************************************************")

        # 生成effect_graph, herb_graph，定义节点类型，生成effect_graph节点掩码
        print("**  generating graphs of each edge_type...      **")
        self.edge_type_graphs = self._generate_edge_type_graphs()

        print("**  generating complete graph...                **")
        self.complete_graph = self._generate_complete_graph()

        print("**  generating herb_graph...                    **")
        self.herb_graph = self._generate_herb_graph()

        print("**  generating effect_graph...                  **")
        self.effect_graph = self._generate_effect_graph()

        print("**************************************************")

        print("**  defining node attributes...                 **")
        self._define_node_attributes()

        print("**  defining edge attributes...                 **")
        self._define_edge_attributes()

        print("**  setting node types and masks...             **")
        self._set_node_types_and_masks()

        print("**************************************************")
        print("**  MHNetwork initialized!                      **")
        print("**************************************************")

    @staticmethod
    def getNodeType(node: str) -> str:
        node_type_dict = {
            "HERB": "Herb",
            "COMP": "Compound",
            "TARG": "Target",
            "GO:": "GO",
            "C": "Disease",
        }
        return node_type_dict[re.sub(r"\d", "", node)]

    def getNodeName(self, node: str) -> str:
        return self.complete_graph.nodes[node]['name']

    def getNodeID(self, name: str) -> str:
        for node_id, node_name in self.complete_graph.nodes('name'):
            if node_name == name:
                return node_id

    # ----------- 生成网络 ----------- #

    # 读取node数据
    @staticmethod
    def load_node_info() -> dict:
        info = dict()
        for node_type in NODE_TYPES:
            if NODE_FILES[node_type]:
                info[node_type] = pd.read_csv(NODE_FILES[node_type], index_col=0)
        return info

    # 读取edge数据
    def load_edge_info(self, examine: bool = False):
        """定义所有类型的edges并合成一个字典

        :param examine: 运行时每生成一类edge，是否将前若干行打印至控制台以检视
        :return: 一个edges字典，keys为所有edge_types，values
        """
        edges = {}
        for edge_type, edge in EDGE_TYPES.items():
            start = edge[0]  # 起点类型
            end = edge[-1]  # 终点类型
            edges[edge_type] = pd.read_csv(EDGE_FILES[edge_type], index_col=0)  # 读取
            if [start, end] in [["Herb", "Compound"], ["Compound", "Target"]]:
                edges[edge_type] = pd.merge(edges[edge_type], self.node_info[start], on=NODE_TYPES[start], how='left')
                edges[edge_type] = pd.merge(edges[edge_type], self.node_info[end], on=NODE_TYPES[end], how='left')
            elif [start, end] in [["Target", "GO"], ["Target", "Disease"]]:
                edges[edge_type] = pd.merge(edges[edge_type], self.node_info["Target"], on='Gene_name', how='left')
            elif start == end == "Target":
                edges[edge_type] = pd.merge(edges[edge_type], self.node_info["Target"],
                                            left_on='Gene_name1', right_on='Gene_name', how='left').iloc[:, :-1]
                edges[edge_type] = pd.merge(edges[edge_type], self.node_info["Target"],
                                            left_on='Gene_name2', right_on='Gene_name', how='left').iloc[:, :-1]
                edges[edge_type].rename(columns={'Target_id_x': 'Target_id1', 'Target_id_y': 'Target_id2'},
                                        inplace=True)
            elif start == end == "GO":
                pass
            else:
                raise ValueError(f"Unidentified edge type: [{start}, {end}]!")

            # 检视当前edge_type的dataframe的生成情况
            if examine:
                print("\n" + edge_type)
                print(edges[edge_type].head())

        return edges

    def _generate_edge_type_graphs(self) -> dict:
        """
        Generate graphs of each edge_type with initial weights.\n
            Target_Disease edge weights are inherited from HERB database.\n
            Target_Target edge weights = (experiment/1000).\n
            Herb_Compound edge weights are set as content percentages,\n
            \tedges without content info are set as the mean value of the remaining percentage\n
            The rest edge type weights = 1.\n
        
        :return: a dict with values of weighted network corresponding to the edge_type of its key\n
            networks["GO_GO"] is a network composed of all GO_GO edges, with weights of 1\n
            
        """
        edge_type_graphs = {}

        for edge_type, edge in EDGE_TYPES.items():
            if edge_type == ["Target_Disease"]:
                pass
            elif edge_type == ["Target_Target"]:
                self.all_edges[edge_type]["weight"] = self.all_edges[edge_type]["experimental"] / 1000
            elif edge_type == ["Herb_Compound"]:
                self.all_edges[edge_type]["weight"] = self.all_edges[edge_type]["prop"]
            else:
                self.all_edges[edge_type]['weight'] = 1

            # 定义graphs
            start = edge[0]  # 起点类型
            end = edge[-1]  # 终点类型
            suffix = {True: ['1', '2'], False: ['', '']}  # source/target后缀，start==end时，需加后缀1/2
            edge_type_graphs[edge_type] = nx.from_pandas_edgelist(
                self.all_edges[edge_type],
                source=NODE_ID[start] + suffix[start == end][0],
                target=NODE_ID[end] + suffix[start == end][1],
                edge_attr="weight")

        return edge_type_graphs

    def _generate_complete_graph(self) -> nx.Graph:
        """
        generating a complete graph using graphs of all edge_types
        """
        if hasattr(self, "edge_type_graphs"):
            return nx.compose_all(list(self.edge_type_graphs.values()))
        else:
            raise Exception("graphs not defined!")

    def _generate_effect_graph(self) -> nx.Graph:
        if hasattr(self, "edge_type_graphs"):
            sub = [self.edge_type_graphs[type_] for type_ in EDGE_TYPES if
                   type_ not in ["Prescription_Herb", "Herb_Compound"]]
            return nx.compose_all(sub)

    def _generate_herb_graph(self) -> nx.Graph:
        """herb graph contains prescriptions, herbs and compounds"""
        if hasattr(self, "edge_type_graphs"):
            sub = [self.edge_type_graphs[type_] for type_ in EDGE_TYPES if
                   type_ in ["Prescription_Herb", "Herb_Compound"]]
            return nx.compose_all(sub)

    def _define_node_attributes(self):
        # 将id列设为索引列，加速后续筛选
        for v in self.node_info.values():
            v.set_index(v.columns[0], inplace=True)

        node_type_dict = {
            'PRES': 'Prescription',
            'HERB': 'Herb',
            'COMP': 'Compound',
            'TARG': 'Target',
            'C': 'Disease',
            'GO': 'GO',
        }
        node_name_dict = {
            'Prescription': 'Prescription',
            'Herb': 'Herb',
            'Compound': 'name',
            'Target': 'Gene_name',
            'Disease': 'Disease_name',
            'GO': 'GO_name',
        }
        node_level_dict = {
            'Prescription': 0,
            'Herb': 0,
            'Compound': 0,
            'Target': 0,
            'Disease': 0,
            'GO': -1,
        }

        def func(graph):
            for node in graph.nodes:
                node_type = self.getNodeType(node)
                node_level = node_level_dict[node_type]
                graph.nodes[node]['type'] = node_type
                graph.nodes[node]['level'] = node_level
                if node_type == 'Compound':
                    graph.nodes[node]['SMILES'] = self.node_info[node_type].loc[node, 'SMILES']
                try:
                    graph.nodes[node]['name'] = self.node_info[node_type].loc[node, node_name_dict[node_type]]
                except KeyError:
                    graph.nodes[node]['name'] = np.nan

        func(self.complete_graph)
        func(self.herb_graph)
        func(self.effect_graph)

    def _define_edge_attributes(self):
        edge_types = [
            ('Herb', 'Compound'),
            ('Compound', 'Target'),
            ('Target', 'Target'),
            ('Target', 'GO'),
            ('GO', 'GO'),
            ('Target', 'Disease'),
        ]

        def func(graph):
            for e in graph.edges:
                for et in edge_types:
                    if set(et) == {graph.nodes[e[0]]['type'], graph.nodes[e[1]]['type']}:
                        graph.edges[e]['type'] = et
                        continue
                try:
                    graph.edges[e]['type']
                except KeyError:
                    print(e)

        func(self.complete_graph)
        func(self.herb_graph)
        func(self.effect_graph)

    def _set_node_types_and_masks(self):
        self.cg_node_types = np.array(self.complete_graph.nodes.data('type'))[:, 1]
        self.cg_node_type_masks = self._get_node_type_masks(self.cg_node_types)

        self.hg_node_types = np.array(self.herb_graph.nodes.data('type'))[:, 1]
        self.hg_node_type_masks = self._get_node_type_masks(self.hg_node_types)

        self.eg_node_types = np.array(self.effect_graph.nodes.data('type'))[:, 1]
        self.eg_node_type_masks = self._get_node_type_masks(self.eg_node_types)

    def nodes_of_type(self, type_: str):
        """get nodes of specific type in complete graph

        Args:
            type_ (str): node type

        Returns:
            nodes (np.array): _description_
        """
        return np.array(self.complete_graph.nodes)[self.cg_node_type_masks[type_]]

    # ------- 计算effect_graph传播参数 -------- #

    # 基于effect_graph的nodes生成各类型mask
    def _get_node_type_masks(self, node_types: ArrayLike) -> dict:
        """
        基于effect_graph的nodes生成各类型mask
        :return: 返回一个字典，键为类型，值为该类型的掩码
        """

        node_type_masks = {
            "Prescription": (node_types == 'Prescription'),
            "Herb": (node_types == 'Herb'),
            "Disease": (node_types == 'Disease'),
            "Compound": (node_types == 'Compound'),
            "Target": (node_types == 'Target'),
            "GO": (node_types == 'GO')
        }

        node_type_masks["Target_&_GO"] = node_type_masks["Target"] | node_type_masks["GO"]

        return node_type_masks

    # 自定义节点权重node_weights
    def set_node_weights(self, node_weights):
        if node_weights is None:
            self.node_weights = NODE_WEIGHTS
        else:
            self.node_weights = node_weights

        self.node_weight_info = (
            f"C{self.node_weights['Compound']:.2f}"
            f"_T{self.node_weights['Target']:.2f}"
            f"_D{self.node_weights['Disease']:.2f}"
            f"_G{self.node_weights['GO']:.2f}"
        )

    # 基于节点权重node_weights计算节点的传播概率
    def cal_transition_probability_of_node(self, matrix, node) -> None:
        """

        Args:
            node:
            matrix:

        Returns:
            无返回值，直接修改输入网络的权重
        """
        neighbor_nodes = list(matrix[node])
        neighbor_node_types = list(
            map(lambda x: matrix.nodes[x]['type'], neighbor_nodes)
        )
        neighbor_node_weights = list(
            map(lambda x: matrix.edges[node, x]['weight'], neighbor_nodes)
        )
        neighbor_node_type_weights = pd.DataFrame([
            neighbor_node_types, neighbor_node_weights
        ]).T.groupby(0).sum()

        weight_sum = np.array(
            list(
                map(lambda x: self.node_weights[x], set(neighbor_node_types))
            )
        ).sum()

        # 若此节点所有邻接节点初始权重均为0，则将所有edge权重设为0，否则以权重计算概率
        if weight_sum == 0:
            for neighbor_node in neighbor_nodes:
                matrix.edges[node, neighbor_node]['weight'] = 0
        else:
            for neighbor_node, neighbor_node_type in zip(neighbor_nodes, neighbor_node_types):
                old_weight = matrix.edges[node, neighbor_node]['weight']  # node和当前neighbor_node的边权重
                w = self.node_weights[neighbor_node_type] / weight_sum  # 某节点类型权重/所有相邻节点类型权重和
                n = neighbor_node_type_weights.loc[neighbor_node_type, 1]  # 当前节点类型的所有节点个数
                if n == 0:
                    matrix.edges[node, neighbor_node]['weight'] = 0
                else:
                    matrix.edges[node, neighbor_node]['weight'] = w * old_weight / n

    # 基于节点传播概率生成邻接矩阵
    def get_adj_with_different_weight(self):
        print("generating adj_matrix...")
        adj_matrix = self.effect_graph.copy()
        for node in adj_matrix.nodes():
            self.cal_transition_probability_of_node(adj_matrix, node)
        adj_matrix = nx.to_numpy_array(adj_matrix)
        # adj = nx.to_scipy_sparse_matrix(multi_network_copy)
        print("adj matrix generated!")
        return adj_matrix

    # 载入或计算邻接矩阵
    def get_adj_matrix(self):
        if not self.force_cal_adj_matrix and os.path.exists(
                OUT_PATH + f"adj_matrices/{self.node_weight_info}.npy"):
            print(f"loading adjacent matrix {self.node_weight_info}.npy...")
            adj = np.load(OUT_PATH + f"adj_matrices/{self.node_weight_info}.npy")
        else:
            adj = self.get_adj_with_different_weight()
            date = datetime.now().strftime('%Y%m%d')
            np.save(OUT_PATH + f"adj_matrices/{date}_{self.node_weight_info}.npy", adj)
        return adj

    def set_effect_graph_adj_matrix(self, force=False, node_weights: dict = None):
        """
        force_cal_adj_matrix: 强制计算邻接矩阵，或者直接读取adj_matrix.npy
        node_weights: 自定义节点权重，若不设置，则读取const中的默认值
        """
        self.force_cal_adj_matrix = force
        self.set_node_weights(node_weights)
        self.effect_graph_adj_matrix = self.get_adj_matrix()

    # ----------- 随机游走 ----------- #
    def set_random_walk_params(self, LAMBDA=0.1, EPSILON=1e-6, n_procs=16):
        """
        LAMBDA: 重新游走的概率
        EPSILON: 随机游走收敛阈值
        n_procs: 随机游走的并行线程数
        """
        self.LAMBDA = LAMBDA
        self.EPSILON = EPSILON
        self.n_procs = n_procs

    def random_walk_from_node(self, node: str) -> dict:

        propagation_map = parallel_get_propagation_maps_from_nodes(
            adj_matrix=self.effect_graph_adj_matrix,
            start_nodes=np.array([node]),
            all_nodes=np.array(self.effect_graph.nodes),
            mask=self.eg_node_type_masks["Compound"] | self.eg_node_type_masks["Disease"],
            herb_graph=self.herb_graph,
            LAMBDA=self.LAMBDA,
            EPSILON=self.EPSILON,
            n_procs=1,
        )

        return propagation_map

    def random_walk_on_effect_graph(
            self,
            log_to_file: bool = True,
            save_pmap: bool = True,
            log_path: str = LOG_PATH,
            return_pmap_fp: bool = False,
    ) -> dict:

        suffix = f'_{self.node_weight_info}_L{self.LAMBDA:.2f}'
        run_log(log_to_file=log_to_file, prefix='gen_pmap_', suffix=suffix, log_path=log_path)

        # 记录随机游走的参数
        self.log_rw_params()

        all_nodes = np.array(self.effect_graph.nodes)
        start_nodes = np.concatenate([
            all_nodes[self.eg_node_type_masks['Compound']],
            all_nodes[self.eg_node_type_masks['Disease']]
        ])
        mask = self.eg_node_type_masks["Compound"] | self.eg_node_type_masks["Disease"]

        propagation_map: dict = {}
        n = 1
        while len(propagation_map) != len(start_nodes) and n < 3:
            propagation_map = parallel_get_propagation_maps_from_nodes(
                adj_matrix=self.effect_graph_adj_matrix,
                start_nodes=start_nodes,
                all_nodes=all_nodes,
                mask=mask,
                herb_graph=self.herb_graph,
                LAMBDA=self.LAMBDA,
                EPSILON=self.EPSILON,
                n_procs=self.n_procs)
            n += 1

        if save_pmap:
            date = datetime.now().strftime('%Y%m%d')
            pmap_fp = OUT_PATH + f'pmaps/{date}_{self.node_weight_info}_L{self.LAMBDA:.2f}.pkl'
            with open(pmap_fp, 'wb') as f:
                pickle.dump(propagation_map, f, pickle.HIGHEST_PROTOCOL)

            if return_pmap_fp:
                return {'pmap': propagation_map, 'pmap_fp': pmap_fp}

        return propagation_map

    def log_rw_params(self):
        logging.info("Random Walk Parameters:")
        logging.info("Compound:  %s", self.node_weights['Compound'])
        logging.info("Target:    %s", self.node_weights['Target'])
        logging.info("Disease:   %s", self.node_weights['Disease'])
        logging.info("GO:        %s", self.node_weights['GO'])
        logging.info("LAMBDA:    %s", self.LAMBDA)
        logging.info("EPSILON:   %s", self.EPSILON)

