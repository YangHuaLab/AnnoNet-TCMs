"""
带有重新开始的随机游走算法
"""
import numpy as np

def PMRWRH(multi_adj, LAMBDA, EPSILON, P0, S, J, MAX_ITER=10000):
    multi_adj = multi_adj.todense()

    P = P0
    old_p = P
    PM = np.array(P.dot(multi_adj)).flatten()
    P = LAMBDA * S + (1 - LAMBDA) * (PM + (P[J].sum()) * S)
    loss = sum(abs(P - old_p))
    ITER = 1
    while loss > EPSILON and ITER <= MAX_ITER:
        old_p = P
        PM = np.array(P.dot(multi_adj)).flatten()
        P = (LAMBDA * S + (1 - LAMBDA) * (PM + (P[J].sum()) * S))
        ITER += 1
        loss = sum(abs(P - old_p))
        """print('iter:{}, loss:{}'.format(ITER, loss), end='\r')"""
    return P
