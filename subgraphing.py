from ctypes import ArgumentError
import logging
import platform
import re
import sys
from collections import Counter
from os.path import dirname, abspath
from typing import Literal

import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import rankdata

from MHNetwork import MHNetwork
from const import OPTIMAL_PMAP, OUT_PATH
from logger import run_log, worker_log
from ranks_and_metrics import get_distance
from util_funcs import getParamDict, getParamList, getParamStr, loadMHNet
from util_funcs import getNeighborNodes, getPathAbbr, loadPmap
from plotting import truncate_colormap, line_wrap_label


class SubGraphConfigs:
    def __init__(
            self,
            net: MHNetwork = None,
            param: str = OPTIMAL_PMAP,
    ) -> None:
        if net is not None:
            self.net = net
        else:
            self.net = loadMHNet()
        self.param = param
        self.param_str = getParamStr(param)
        self.param_list = getParamList(self.param_str)
        self.param_dict = getParamDict(self.param_list)

        self.pmap = loadPmap(net, self.param)

        self.set_generating_configs(
            parallel_log=True,
            n_total_comps=40,
            n_H2D_steps=5,
            dist_method='canberra',
            n_procs=60,
            n_TG_nodes=100,
            get_TG_node_by='alpha',
            show_comp_node_by='prob',
        )

        self.set_drawing_configs(
            cal_pmap_prob_by='herb',
            n_shown_nodes={},
            save_plot=False,
        )

    def set_generating_configs(
            self,
            parallel_log: bool = None,
            n_total_comps: int = None,
            n_H2D_steps: int = None,
            dist_method: str = None,
            n_procs: int = None,
            n_TG_nodes: int = None,
            get_TG_node_by: Literal['path', 'alpha'] = None,  # 以什么为标准获取子图节点
            show_comp_node_by: Literal['prob', 'dist'] = None,
    ) -> None:
        """
        Args:
            get_TG_node_by (str): choose in ['path', 'alpha'].
            show_comp_node_by (str): choose in ['prob', 'dist'].
            parallel_log (bool): whether to take log while parallel getting path nodes.
            n_total_comps (int): total number of calculated compounds.
            n_H2D_steps (int): number of herb to disease path nodes. only functional when geth_TG_node_by set to 'path'.
            dist_method (str): distance method.
            n_procs (int): number of procedures.
            n_TG_nodes (int): number of target and GO nodes of the subgraph. only functional when geth_TG_node_by set to 'alpha'.

        Raises:
            KeyError: get_TG_node_by not in ['path', 'alpha']
            KeyError: get_comp_node_by not in ['prob', 'dist']
        """
        if parallel_log is not None: self.parallel_log = parallel_log
        if dist_method is not None: self.dist_method = dist_method
        if n_total_comps is not None: self.n_total_comps = n_total_comps
        if n_H2D_steps is not None: self.n_H2D_steps = n_H2D_steps
        if n_H2D_steps is not None: self.n_C2D_steps = n_H2D_steps - 1
        if n_procs is not None: self.n_procs = n_procs

        if n_TG_nodes is not None and n_TG_nodes > 0:
            self.n_TG_nodes = n_TG_nodes
        if n_TG_nodes is not None and n_TG_nodes <= 0 or (n_TG_nodes is None and not hasattr(self, 'n_TG_nodes')):
            self.n_TG_nodes = len(self.net.node_info['Target']) + len(self.net.node_info['GO'])

        if get_TG_node_by is not None:
            if get_TG_node_by in ['path', 'alpha']:
                self.get_TG_node_by = get_TG_node_by
            else:
                raise KeyError(f"Incorrect input of get_TG_node_by: '{get_TG_node_by}'")

        if show_comp_node_by is not None:
            if show_comp_node_by in ['prob', 'dist']:
                self.show_comp_node_by = show_comp_node_by
            else:
                raise KeyError(f"Incorrect input of set_comp_alpha_by: '{show_comp_node_by}'")

    def set_drawing_configs(
            self,
            cal_pmap_prob_by: str = None,  # 以什么为单位计算节点概率
            n_shown_nodes: dict = None,
            save_plot: bool = None,
    ):
        if cal_pmap_prob_by is not None:
            if cal_pmap_prob_by in ['herb', 'comp']:
                self.cal_pmap_prob_by = cal_pmap_prob_by
            else:
                raise KeyError(f"Incorrect input cal_prob_by: {cal_pmap_prob_by}")
        if not hasattr(self, 'n_shown_nodes'):
            self.n_shown_nodes = {}
        if n_shown_nodes is not None:
            self.n_shown_nodes = n_shown_nodes
        if save_plot is not None: self.save_plot = save_plot

    def set_nodes(
            self,
            compounds: str | list[str] = None,
            herbs: str | list[str] = None,
            herb_weights: float | list[float] = None,
            dise: str = None,
    ) -> None:
        self.set_compounds(compounds=compounds)
        self.set_herbs(herbs=herbs, weights=herb_weights)
        self.set_dise(dise=dise)

    def set_compounds(self, compounds: str | list[str]) -> None:
        if isinstance(compounds, list) or compounds is None:
            self.compounds = compounds
        elif isinstance(compounds, str):
            self.compounds = [compounds]
        else:
            raise TypeError(f"Incorrect type of argument 'compounds', expected str or list, got {type(compounds)}")

    def set_herbs(self, herbs: str | list[str] | None = None, weights: float | list[float] | None = None):
        if isinstance(herbs, str):
            self.herbs = [herbs]
        elif isinstance(herbs, list) and len(herbs)>0:
            self.herbs = herbs
        elif herbs is None or herbs == []:
            self.herbs = []
            self.n_herb_comps = []
            self.herb_weights = []
            self.total_herb_weight = []
            return
        else:
            raise TypeError(f"Incorrect type of argument herbs, expected str or list or None, got {type(herbs)}")
        self.n_herb_comps = self.n_total_comps // len(self.herbs)
        if weights is None:
            self.herb_weights = [1 for _ in herbs]
            self.total_herb_weight = sum(self.herb_weights)
        else:
            self.set_herb_weights(weights)

    def set_herb_weights(self, weights: list):
        if len(weights) != len(self.herbs):
            raise ValueError(
                f"Number of weights {len(weights)} is not the same with the number of herbs {len(self.herbs)}!")
        self.herb_weights = weights
        self.total_herb_weight = sum(self.herb_weights)

    def set_dise(self, dise: str):
        self.dise = dise
        # self.check_if_node_in_pmap(node=dise)

    def set_n_H2D_steps(self, n_H2D_steps):
        self.n_H2D_steps = n_H2D_steps
        self.n_C2D_steps = n_H2D_steps - 1

    def save_plot(self, b: bool = True):
        self.save_plot = b

    def keep_parallel_log(self, b: bool = True):
        self.parallel_log = b

    def get_subgraph_name(self):
        string = ''
        for h in self.herbs:
            string = string + self.net.getNodeName(h) + ','
        string = string[:-1]
        return f"{string}-{self.net.getNodeName(self.dise)}"

    def load_ranks(self):
        self.D2C_rank = pd.read_csv(
            OUT_PATH + f"ranks_and_metrics/{self.param_str}/D2C_rank_{self.dist_method}.csv"
        )
        self.D2H_rank = pd.read_csv(
            OUT_PATH + f"ranks_and_metrics/{self.param_str}/D2H_ranks_{self.dist_method}/D2H_rank_{self.dist_method}_40.csv"
        )

    def set_n_comps(self, n_comps: int) -> None:
        self.n_herb_comps = n_comps

    def check_if_node_in_pmap(self, node) -> None:
        if node not in self.pmap.keys():
            self.net.set_effect_graph_adj_matrix(node_weights=self.param_dict)
            self.net.set_random_walk_params(LAMBDA=self.param_dict["LAMBDA"], n_proc=self.n_procs)
            self.pmap.update(self.net.random_walk_from_node(node))


class SubGraphAnalyzer:
    def __init__(
            self,
            configs: SubGraphConfigs,
            compounds: str | list = None,
            herbs: str | list = None,
            dise: str = None,
            exclude_nodes: list = None,
    ) -> None:
        self.drawing_params()
        self.cfgs = configs
        self.pmap = self.cfgs.pmap
        self.cg = self.cfgs.net.complete_graph
        self.hg = self.cfgs.net.herb_graph
        self.eg = self.cfgs.net.effect_graph
        self.TG_mask = self.cfgs.net.eg_node_type_masks['Target_&_GO']
        self.cfgs.set_nodes(compounds=compounds, herbs=herbs, dise=dise)
        self.top_comps = self._get_top_comps()
        self.generate_subgraph()

    def _get_C2D_nodes(self, start: str, end: str, ) -> set:
        """获取指定compound-disease对之间短于指定步长的路径
        
        删除多余的Compound、Disease和Target节点
        从start开始游走，不重复地记录经过的点和路径
        当游走至与end相连的target节点时，记录该条路径上的所有节点
        
        """
        eg = self.eg.copy()
        s, stype = start, eg.nodes[start]['type']
        e, etype = end, eg.nodes[end]['type']
        # print(f"{s}-{e} paths less than {n_steps+1} steps")

        # print('removing excessive nodes...')
        s_targs = [n for n in eg[s] if eg.nodes[n]['type'] == 'Target']
        e_targs = [n for n in eg[e] if eg.nodes[n]['type'] == 'Target']
        targs = set(s_targs + e_targs)
        rm_targs = [n for n in eg if eg.nodes[n]['type'] == 'Target' and n not in targs]
        eg.remove_nodes_from(rm_targs)  # 移除与起始与结束点无关的Target点
        eg.remove_nodes_from(
            [n for n in eg if eg.nodes[n]['type'] == stype and n != s])  # 移除除其自身外与起始点相同类型的点
        eg.remove_nodes_from(
            [n for n in eg if eg.nodes[n]['type'] == etype and n != e])  # 移除除其自身外与结束点相同类型的点

        # 开始游走
        # print('walking...')
        inpath_nodes = {s, e}  # 确定存在于C2D路径上的点
        bypass_nodes = set([])  # 经过但不确定是否存在于C2D路径上的点
        n_step_paths = [[[n] for n in s_targs]]  # 根据步数记录符合其他条件但尚未连通始末点的路径，
        # 从s_targs游走至e_targs，path须为T(G)T形式，记录inpath_nodes
        for _ in range(self.cfgs.n_C2D_steps - 2):
            n_step_paths.append([])
            for path in n_step_paths[-2]:  # 基于上一步的路径path
                for node in getNeighborNodes(eg, path[-1]):  # 从上一步的终点前进一步
                    if node in inpath_nodes or node in bypass_nodes:  # 若此点已经由更短的路径出现过，则记录经过的点，但不记录路径
                        inpath_nodes = inpath_nodes | set(path + [node])  # 将此路径中的点记录进inpath_nodes
                        bypass_nodes = bypass_nodes - inpath_nodes  # 将此路径中的点移出bypass_nodes
                        continue
                    if re.match("^T+G*T*$", getPathAbbr(path + [node])) is None:  # 若路径形成不为^T+G*T*$，则不考虑
                        continue
                    elif node in e_targs:  # 若此点未出现过，且与终点相连
                        inpath_nodes = inpath_nodes | set(path + [node])  # 将此路径中的点记录进inpath_nodes
                        bypass_nodes = bypass_nodes - inpath_nodes  # 将此路径中的点移出bypass_nodes
                    else:  # 若此步点未出现过，且不与终点相连
                        bypass_nodes = bypass_nodes | {node}  # 将此路径中的点记录进bypass_nodes
                        n_step_paths[-1].append(path + [node])  # 将此路径记录进n_step_paths中
            # print(f"{i+1} step paths: {len(n_step_paths[-1])}, {len(inpath_nodes)} in-path nodes")

        return inpath_nodes

    def _get_top_comps(self) -> dict:
        assert hasattr(self.cfgs, 'herbs')
        assert hasattr(self.cfgs, 'dise')

        if self.cfgs.compounds is not None:
            return {'all': np.array(self.cfgs.compounds)}

        comps = [c for h in self.cfgs.herbs for c in self.hg[h]]
        nodes = dict()

        if self.cfgs.cal_pmap_prob_by == 'herb':  # 每个药材的化合物分别获取前若干个
            for h in self.cfgs.herbs:
                df = pd.DataFrame(
                    [[n, get_distance(
                        self.pmap[n][self.TG_mask],
                        self.pmap[self.cfgs.dise][self.TG_mask],
                        self.cfgs.dist_method,
                    )] for n in self.hg[h]],
                    columns=['Compound_id', 'distance']
                )
                nodes[h] = np.array(df.sort_values('distance')['Compound_id'][:self.cfgs.n_herb_comps])
        elif self.cfgs.cal_pmap_prob_by == 'comp':  # 所有药材的化合物汇总获取前若干个
            df = pd.DataFrame(
                [[n, get_distance(
                    self.pmap[n][self.TG_mask],
                    self.pmap[self.cfgs.dise][self.TG_mask],
                    self.cfgs.dist_method,
                )] for n in comps],
                columns=['Compound_id', 'distance']
            )
            nodes['all'] = np.array(df.sort_values('distance')['Compound_id'][:self.cfgs.n_total_comps])

        return nodes

    def get_path_nodes(self) -> set:  # 仅在
        comps = np.concatenate(list(self.top_comps.values()))

        def worker_func(queue, comps: list, dise: str, worker: str):
            worker_log(verbose=True, worker=worker, msg=f"starting {len(comps)} nodes...")
            result = set([])
            for i, c in enumerate(comps):
                result = result | self._get_C2D_nodes(start=c, end=dise)
                worker_log(verbose=True, worker=worker, msg=f"{c} - {dise} done! ({i + 1}/{len(comps)})")
            queue.put(result)

        all_nodes = set(self.cfgs.herbs)

        n_procs = min(self.cfgs.n_procs, len(comps), mp.cpu_count())
        processes = []
        q = mp.Manager().Queue()
        # h2d_nodes = set([h])

        if self.cfgs.parallel_log:
            run_log(log_to_file=False)

        logging.info(f"parallel getting {self.cfgs.herbs} - {self.cfgs.dise} nodes...")
        for i in range(n_procs):  # 循环生成子进程
            sub_comps = comps[range(i, len(comps), n_procs)]
            worker = f"{i:0{len(str(n_procs - 1))}}"  # 子进程编号
            p = mp.Process(
                target=worker_func, args=(q, sub_comps, self.cfgs.dise, worker))
            processes.append(p)
            # pool.apply_async(worker_func, (q, sub_comps, worker))
            p.start()

        for p in processes:
            p.join()

        logging.info(f"parallel getting {self.cfgs.herbs} - {self.cfgs.dise} nodes complete!")
        while not q.empty():
            result = q.get()
            all_nodes = all_nodes | result

        return all_nodes

    def load_prob_to_graphs(self) -> None:
        node_number_dict = dict(zip(self.eg.nodes, range(len(self.eg.nodes))))  # 字典{node_id: serial number}

        if self.cfgs.cal_pmap_prob_by == 'comp':
            # 不考虑药材，将所有化合物排序后取前若干个的pmap计算均值
            comp_pmaps = [self.pmap[c] for c in self.top_comps['all']]
            comp_pmaps_mean = np.mean(np.array(comp_pmaps), axis=0)
        elif self.cfgs.cal_pmap_prob_by == 'herb':
            # 考虑药材，每个药材的化合物排序后取前若干个的pmap计算均值，再将每个药材的平均pmap计算加权平均pmap
            all_herb_pmaps = []
            for h in self.cfgs.herbs:
                df = pd.DataFrame(
                    [[n, get_distance(
                        self.pmap[n][self.TG_mask],
                        self.pmap[self.cfgs.dise][self.TG_mask],
                        self.cfgs.dist_method,
                    )] for n in self.hg[h]],
                    columns=['Compound_id', 'distance']
                )
                top_comps = np.array(df.sort_values('distance')['Compound_id'][:self.cfgs.n_herb_comps])
                herb_mean_pmap = np.mean(np.array([self.pmap[n] for n in top_comps]), axis=0)
                all_herb_pmaps.append(herb_mean_pmap)
            comp_pmaps_mean = np.sum(
                np.array(all_herb_pmaps) * [[w / self.cfgs.total_herb_weight] for w in self.cfgs.herb_weights],
                axis=0
            )
        else:
            raise ArgumentError(f"Incorrect input of by: '{self.cfgs.cal_pmap_prob_by}'")

        # 最终节点和边的probability定义为comp_pmaps_mean与dise_pmap相乘
        dise_pmap = self.pmap[self.cfgs.dise]
        for n in self.eg.nodes:
            if self.cfgs.net.getNodeType(n) in ['Target', 'GO']:
                self.eg.nodes[n]['alpha'] = comp_pmaps_mean[node_number_dict[n]] * dise_pmap[node_number_dict[n]]
            elif self.cfgs.net.getNodeType(n) == 'Disease':
                self.eg.nodes[n]['alpha'] = comp_pmaps_mean[node_number_dict[n]]
            elif self.cfgs.net.getNodeType(n) == 'Compound':
                self.eg.nodes[n]['distance'] = get_distance(
                    x=self.pmap[n][self.TG_mask],
                    y=self.pmap[self.cfgs.dise][self.TG_mask],
                    method=self.cfgs.dist_method
                )
                self.eg.nodes[n]['probability'] = dise_pmap[node_number_dict[n]]
                if self.cfgs.show_comp_node_by == 'dist':
                    self.eg.nodes[n]['alpha'] = 1 / self.eg.nodes[n]['distance']
                elif self.cfgs.show_comp_node_by == 'prob':
                    self.eg.nodes[n]['alpha'] = self.eg.nodes[n]['probability']

        for n in self.cg.nodes:
            if self.cfgs.net.getNodeType(n) == 'Herb':
                self.cg.nodes[n]['alpha'] = 1
            else:
                self.cg.nodes[n]['alpha'] = self.eg.nodes[n]['alpha']
            if self.cfgs.net.getNodeType(n) == 'Compound':
                self.cg.nodes[n]['distance'] = self.eg.nodes[n]['distance']
                self.cg.nodes[n]['probability'] = self.eg.nodes[n]['probability']

        for e in self.cg.edges:
            try:
                self.cg.edges[e]['alpha'] = self.cg.nodes[e[0]]['alpha'] * self.cg.nodes[e[1]]['alpha']
            except KeyError:
                self.cg.edges[e]['alpha'] = 1

    def generate_subgraph(self):
        self.load_prob_to_graphs()

        if self.cfgs.get_TG_node_by == 'path':
            self.subgraph: nx.Graph = nx.subgraph(
                self.cg.copy(),
                self.get_path_nodes()
            )
        elif self.cfgs.get_TG_node_by == 'alpha':
            df = pd.DataFrame(self.cg.nodes('alpha'), columns=['node', 'alpha'])
            df = df[[self.cfgs.net.getNodeType(i) in ['Target', 'GO'] for i in df['node']]]
            df = df.sort_values('alpha', ascending=False)
            nodes = set(df['node'][:self.cfgs.n_TG_nodes])
            if self.cfgs.herbs is not None:
                nodes = nodes | set(self.cfgs.herbs)
            nodes = nodes | set(np.concatenate(list(self.top_comps.values())))
            nodes = nodes | {self.cfgs.dise}
            self.subgraph: nx.Graph = nx.subgraph(
                self.cg.copy(),
                nodes
            )

        self.comp_df = self.get_comp_df()
        self.targ_df = self.get_targ_df()
        self.GO_df = self.get_GO_df()

    def set_n_shown_nodes(self):
        df = pd.DataFrame({
            'node': np.array(self.subgraph.nodes),
            'name': np.array(self.subgraph.nodes.data('name'))[:, 1],
            'type': np.array(self.subgraph.nodes.data('type'))[:, 1],
            'alpha': np.array(self.subgraph.nodes.data('alpha'))[:, 1].astype(np.float64),
        })

        drop_nodes = np.array([])
        for node_type in self.cfgs.n_shown_nodes.keys():
            try:
                sorted_nodes = df.set_index(['type']).loc[node_type].sort_values('alpha', ascending=False)
                drop = sorted_nodes.loc[:, 'node'][self.cfgs.n_shown_nodes[node_type]:]
                drop_nodes = np.concatenate([drop_nodes, drop])
            except KeyError:  # 无此类型节点或只有一个，则不drop
                pass
            except TypeError:
                pass

        res_array = np.setdiff1d(df.loc[:, 'node'], drop_nodes, assume_unique=True)
        self.subgraph: nx.Graph = nx.subgraph(self.subgraph, res_array)

    def rescale_subgraph_alpha(self):
        # 对于Herb, Disease节点，alpha直接设为1
        for n, a in self.subgraph.nodes.items():
            if a['type'] in ['Herb', 'Disease']:
                self.subgraph.nodes[n]['alpha'] = 1

        # 对于其他节点类型
        # 如果该类型只有一个点，则alpha设为1
        # 否则将该类型alpha缩放至0~1
        # 最后将alpha赋值回subgraph
        for node_type in ['Compound', 'Target', 'GO']:
            l = []

            for n, a in self.subgraph.nodes.items():  # node, attributes
                if a['type'] == node_type:
                    l = l + [[n, a['alpha']]]
            df = pd.DataFrame(l, columns=['node', 'alpha'])

            if len(df) == 1:
                df['alpha'] = 1
            else:
                df['alpha'] = (df['alpha'] - df['alpha'].min()) / (df['alpha'].max() - df['alpha'].min())

            for _, (n, a) in df.iterrows():
                self.subgraph.nodes[n]['alpha'] = a

        edge_types = [
            ('Herb', 'Compound'),
            ('Compound', 'Target'),
            ('Target', 'Target'),
            ('Target', 'GO'),
            ('GO', 'GO'),
            ('Target', 'Disease')
        ]

        for edge_type in edge_types:
            df = pd.DataFrame(
                [[e, a['alpha']] for e, a in self.subgraph.edges.items() if a['type'] == edge_type],
                columns=['edge', 'alpha']
            )

            if len(df) == 1:
                df['alpha'] = 1
            else:
                df['alpha'] = (df['alpha'] - df['alpha'].min()) / (df['alpha'].max() - df['alpha'].min())

            for _, (e, a) in df.iterrows():
                self.subgraph.edges[e]['alpha'] = a

    def relevel_GO(self, df: pd.DataFrame):
        # df为GO节点DataFrame，index为node_id
        # 定义GO节点的层级
        # 初始GO节点level为-1
        # 若不与任何节点相连，设为5
        # 若与Target相连，设level为1，其中互相相连的level设为3
        # 若仅与GO相连，设level为5，其中互相相连的level设为7
        for v in df.index:
            try:
                max_neighbor_level = max(map(
                    lambda x: self.subgraph.nodes[x]['level'],
                    self.subgraph[v].keys())
                )
            except ValueError:  # 没有连接
                df.loc[v, 'level'] = 10
                continue
            if max_neighbor_level == 0:  # 与Target有连接
                df.loc[v, 'level'] = 1
            elif max_neighbor_level == -1:  # 只与GO有连接
                df.loc[v, 'level'] = 5
            else:
                print(v, max_neighbor_level)

        for u in df[df['level'] == 1].index:
            for v in df[df['level'] == 1].index:
                if self.subgraph.has_edge(u, v):
                    if df.loc[u, 'alpha'] <= df.loc[v, 'alpha']:
                        df.loc[u, 'level'] = 3
                    else:
                        df.loc[v, 'level'] = 3

        for u in df[df['level'] == 5].index:
            for v in df[df['level'] == 5].index:
                if self.subgraph.has_edge(u, v):
                    if df.loc[u, 'alpha'] <= df.loc[v, 'alpha']:
                        df.loc[u, 'level'] = 7
                    else:
                        df.loc[v, 'level'] = 7

        level_1 = df[df['level']==1].sort_values('alpha', ascending=False)
        if len(level_1) > 10:
            for i in range(1, len(level_1), 2):
                df.loc[level_1.iloc[i].name, 'level'] = 2

        for node_id, data in df.iterrows():
            self.subgraph.nodes[node_id]['level'] = data['level']
        
        # 删去没有任何连接的GO节点
        df = df[df['level']!=10]
        
        return df

    def drawing_params(self):
        # types and cmaps
        nt = ['Herb', 'Compound', 'Target', 'GO', 'Disease']
        self.node_types = nt
        self.edge_types = [
            ('Herb', 'Compound'),
            ('Compound', 'Target'),
            ('Target', 'Target'),
            ('Target', 'GO'),
            ('GO', 'GO'),
            ('Target', 'Disease')
        ]
        self.node_cmaps = dict(zip(nt, [
            truncate_colormap(plt.cm.Greens, 0.99, 1),
            truncate_colormap(plt.cm.Purples, 0.05, 0.7),
            truncate_colormap(plt.cm.Blues, 0.05, 0.7),
            truncate_colormap(plt.cm.Reds, 0.05, 0.7),
            truncate_colormap(plt.cm.Oranges, 0.99, 1),
        ]))
        self.edge_cmaps = dict(zip(self.edge_types, [
            truncate_colormap(plt.cm.Greens, 0.3, 1),
            truncate_colormap(plt.cm.Purples, 0.3, 1),
            truncate_colormap(plt.cm.Blues, 0.3, 1),
            truncate_colormap(plt.cm.Greys, 0.3, 1),
            truncate_colormap(plt.cm.Reds, 0.3, 1),
            truncate_colormap(plt.cm.Oranges, 0.3, 1),
        ]))
        
        # node
        self.node_shapes = dict(zip(nt, ['s', 'h', '^', 'o', 'd']))
        self.node_sizes = dict(zip(nt, [50, 80, 200, 300, 80]))
        
        # layout params
        self.layout_centers = dict(zip(nt, np.array([
            [-2.5, -2.5],
            [-1.5, -1.0],
            [+1.5, -1.0],
            [+1.5, -0.0],  # 纵坐标，横坐标；其余相反
            [+3.0, -2.5]
        ])))
        self.layout_scales = dict(zip(nt, [1/20, 1/15, 1/15, 1, 1]))
        # label params
        self.label_alignments = {
            'Herb': ['center', 'center'],
            'Compound': ['center', 'center'],
            'Target': ['center', 'center'],
            'GO': ['center', 'center'],
            'Disease': ['center', 'center'],
        }  # dict(zip(nt, [['center', 'center']]*5))
        self.label_font_sizes = dict(zip(nt, [4, 3, 3, 2, 3]))
        self.label_font_max_sizes = dict(zip(nt, [5, 5, 5, 2, 5]))
        self.label_pos_offset = dict(zip(nt, [[0,-0.4], [0,0], [0,-0.1], [0,0], [-0.4,-0.5]]))
        self.funcs = {
            'node_size': lambda t,n: self.node_sizes[t] / (n**0.3),  # n: number of t type nodes
            'layout_scale': lambda df,t: (len(df)+6)*self.layout_scales[t],  # df: node type df, t: node type
            'GO_layout_scale': lambda n: self.layout_scales['GO'] * (n**0.5),  # n: number of nodes of the level with the most nodes
            'edge_color': lambda a: np.maximum(a, np.array(0)),
            'label_alpha': lambda a: np.maximum((np.log2(a + 1) + 0.1) * (1 / 1.1), 0.4),
            # 'label_font_size': lambda a,t: np.minimum(np.maximum(self.label_font_sizes[t] * a,
            #                                                      self.label_font_sizes[t] * 0.3, np.array(1.0)),
            #                                           np.array(self.label_font_max_sizes[t])),
            'label_font_color': lambda a: 'k' if a<0.5 else '#dddddd',
            'label_font_size': lambda t: self.label_font_sizes[t],
            'label_pos': lambda i,p: p + self.label_pos_offset[MHNetwork.getNodeType(i)]
        }

    def draw_subgraph(self, dpi: int = 900, figsize: tuple[float, float] = (6.0, 6.0), print_params: bool = True):
        if not hasattr(self, 'subgraph'):
            self.generate_subgraph()

        self.set_n_shown_nodes()
        self.rescale_subgraph_alpha()
        
        nodes: dict[str, pd.DataFrame] = dict()
        edges: dict[tuple[str, str], pd.DataFrame] = dict()
        pos = dict()

        # 按节点类型定义node
        for t in self.node_types:
            df = pd.DataFrame(
                [(n, a['name'], a['alpha'], int(a['level']))
                 for n, a in self.subgraph.nodes.items() if a['type'] == t],
                columns=['node_id', 'name', 'alpha', 'level']
            )

            df['rank'] = np.int64(len(df) - rankdata(df['alpha']) + 1)

            if len(df) > 0:
                if t != 'GO':
                    type_pos = nx.circular_layout(df['node_id'],
                        scale=self.funcs['layout_scale'](df, t),
                        center=self.layout_centers[t]
                    )
                    if t == 'Compound' and len(df) == 2:
                        angle = np.pi / 2
                        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                                    [np.sin(angle), np.cos(angle)]])
                        for node in type_pos:
                            type_pos[node] -= self.layout_centers[t]
                            type_pos[node] = np.dot(rotation_matrix, type_pos[node])
                            type_pos[node] += self.layout_centers[t]
                    pos = dict(**pos, **type_pos)
                nodes[t] = df.set_index('node_id')
            else:
                print(f"There's no {t} nodes!")
        
        # 根据GO节点层级数定义GO节点layout纵坐标
        nodes['GO'] = self.relevel_GO(nodes['GO'])
        n_GO_levels = len(set(nodes['GO']['level']))
        # self.layout_centers['GO'] = np.array([-2.2+n_GO_levels*0.2, 0])
        n = max(Counter(nodes['GO']['level']).values())
        
        G = nx.subgraph(self.subgraph.copy(), nodes['GO'].index)
        H = nx.Graph()
        H.add_nodes_from(sorted(G.nodes(data=True), key=lambda x: x[1]['alpha'], reverse=False))
        H.add_edges_from(G.edges(data=True))

        pos = dict(**pos, **nx.multipartite_layout(
            H,
            subset_key='level',
            align='horizontal',
            scale=self.funcs['GO_layout_scale'](n),
            center=self.layout_centers['GO'],
        ))

        # 定义边
        for t in self.edge_types:
            edges[t] = pd.DataFrame([[e, a['alpha']] for e, a in self.subgraph.edges.items() if a['type'] == t])

        ######################## 绘制 ########################
        # plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        elif "WSL" in platform.release():
            plt.rcParams['font.sans-serif'] = ['.PingFang SC']
        else:
            plt.rcParams['font.sans-serif'] = ['Arial']
            # plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        plt.figure(
            dpi=dpi,
            figsize=figsize,
        )
        plt.margins(0.03, 0.03)
        plt.axis('off')
        
        # 按类型绘制nodes
        for t,df in nodes.items():
            nx.draw_networkx_nodes(
                self.subgraph, pos,
                nodelist=list(df.index),
                node_color=df['alpha'].astype(np.float64),
                cmap=self.node_cmaps[t],
                node_size=self.funcs['node_size'](t, len(df)),
                node_shape=self.node_shapes[t],
                vmin=0, vmax=1,
            )

        label_pos = {k: self.funcs['label_pos'](k,v) for k,v in pos.items()}

        # 按类型绘制labels
        for t,df in nodes.items():
            for v in df.index:
                a = self.subgraph.nodes[v]
                if t in ['Herb', 'Disease']:
                    label = {v: str(a['name'])}
                else:
                    label = {v: line_wrap_label(str(a['name']))}
                nx.draw_networkx_labels(
                    self.subgraph, label_pos,
                    labels=label,
                    alpha=1,  # self.funcs['label_alpha'](a['alpha']),
                    font_size=self.funcs['label_font_size'](t),  # self.funcs['label_font_size'](a['alpha'], t),
                    font_color='k',  # if len(df)==1 else self.funcs['label_font_color'](a['alpha']),
                    horizontalalignment=self.label_alignments[t][0],
                    verticalalignment=self.label_alignments[t][1],
                )

        # 按类型绘制edges
        for t in self.edge_types:
            try:  # continue if there's no t type edges
                nx.draw_networkx_edges(
                    self.subgraph, pos,
                    edgelist=edges[t].iloc[:, 0],
                    edge_color=self.funcs['edge_color'](edges[t].iloc[:, 1].astype(np.float64)),
                    edge_cmap=self.edge_cmaps[t],
                    width=0.1,
                    alpha=1,
                    edge_vmin=0,
                    edge_vmax=1,
                )
            except IndexError:
                print(f"There's no {t[0]}-{t[1]} edges!")

        if print_params:
            ax = plt.gca()
            ax.text(x=-4, y=4, s=self.param_text(), fontsize=2, verticalalignment='top', horizontalalignment='left')

        plt.xlim((-4,4))
        plt.ylim((-4,4))
        # plt.gca().set_axis_on()
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)  # 显示刻度和标签
        
        # 保存subgraph
        if self.cfgs.save_plot:
            plt.savefig(OUT_PATH + f"subgraphs/{self.cfgs.get_subgraph_name()}.svg", format='svg')

        plt.show()

    def param_text(self):
        text = f"configs.param_list = {self.cfgs.param_list}\n"
        text += f"configs.dist_method = {self.cfgs.dist_method}\n"
        text += f"configs.n_total_comps = {self.cfgs.n_total_comps}\n"
        text += f"configs.cal_pmap_prob_by = {self.cfgs.cal_pmap_prob_by}\n"
        text += f"configs.get_subg_node_by = {self.cfgs.get_TG_node_by}\n"
        if self.cfgs.get_TG_node_by == 'path':
            text += f"configs.n_H2D_steps = {self.cfgs.n_H2D_steps}\n"
        elif self.cfgs.get_TG_node_by == 'alpha':
            text += f"configs.n_TG_nodes = {self.cfgs.n_TG_nodes}\n"
        # if self.configs.n_shown_nodes is not None:
        #     text += f"configs.n_shown_nodes = {self.configs.n_shown_nodes}\n"
        return text

    def get_comp_df(self):
        df = pd.DataFrame([[
                c,
                self.cfgs.net.getNodeName(c),
                self.subgraph.nodes[c]['distance'],
                self.subgraph.nodes[c]['probability'],
            ] + [
                c in self.hg[h] for h in self.cfgs.herbs
            ] for c in set(np.concatenate(
                list(self.top_comps.values())))],
            columns=['ID', 'Compound', 'Distance', 'Probability'] + [self.cfgs.net.getNodeName(h) for h in self.cfgs.herbs]
        )
        return df.sort_values('Distance', ignore_index=True)

    def get_targ_df(self):
        df = pd.DataFrame(
            [[
                t,
                self.cfgs.net.getNodeName(t),
                self.subgraph.nodes[t]['alpha'],
            ] for t in self.subgraph.nodes if self.cfgs.net.getNodeType(t) == 'Target'],
            columns=['id', 'target', 'alpha']
        )
        return df.sort_values('alpha', ascending=False, ignore_index=True)

    def get_GO_df(self):
        df = pd.DataFrame(
            [[
                g,
                self.cfgs.net.getNodeName(g),
                self.subgraph.nodes[g]['alpha'],
            ] for g in self.subgraph.nodes if self.cfgs.net.getNodeType(g) == 'GO'],
            columns=['id', 'GO', 'alpha']
        )
        return df.sort_values('alpha', ascending=False, ignore_index=True)


if __name__ == '__main__':
    # pmap_fp = OUT_PATH + '/pmaps/propagation_map_C1.00_T3.00_D6.00_G0.50_L0.10_E1e-06.pkl'
    # pmap_fp = OUT_PATH + '/pmaps/propagation_map_C1.00_T1.00_D3.00_G2.00_L0.10_E1e-06.pkl'
    # pmap_fp = OUT_PATH + '/pmaps/propagation_map_C1.00_T1.00_D1.00_G2.00_L0.67_E1e-06.pkl'
    pmap_fp = OUT_PATH + '/pmaps/propagation_map_C1.00_T1.00_D0.50_G1.50_L0.67_E1e-06_2024-02-26.pkl'
    configs = SubGraphConfigs(pmap_fp)
    configs.set_nodes(
        herbs=['HERB122'],
        dise='C0007222',
    )
    analyzer = SubGraphAnalyzer(configs=configs)
    analyzer.draw_subgraph()

    # generate_H2D_subgraph(configs, ['HERB122'], 'C0007222')
    # generate_H2D_subgraph(configs, ['HERB018'], 'C0007222')

    # configs.set_n_comps(20)
    # generate_H2D_subgraph(configs, ['HERB122', 'HERB018'], 'C0007222')
