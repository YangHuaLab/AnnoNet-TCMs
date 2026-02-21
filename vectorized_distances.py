import numpy as np
from typing import Literal, Optional

# ==================== CDIST 支持的距离函数 ====================

def cosine_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的余弦距离"""
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    X_norm = np.where(X_norm == 0, 1, X_norm)
    Y_norm = np.where(Y_norm == 0, 1, Y_norm)
    similarity = np.dot(X / X_norm, (Y / Y_norm).T)
    return 1 - similarity

def correlation_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的相关距离"""
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    X_norm = np.linalg.norm(X_centered, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y_centered, axis=1, keepdims=True)
    X_norm = np.where(X_norm == 0, 1, X_norm)
    Y_norm = np.where(Y_norm == 0, 1, Y_norm)
    similarity = np.dot(X_centered / X_norm, (Y_centered / Y_norm).T)
    return 1 - similarity

def euclidean_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """向量化的欧几里得距离"""
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1)
    dot_product = np.dot(X, Y.T)
    distance_sq = X_sq + Y_sq - 2 * dot_product
    return np.sqrt(np.maximum(distance_sq, 0))

def manhattan_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """向量化的曼哈顿距离"""
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    return np.sum(np.abs(X_exp - Y_exp), axis=2)

def chebyshev_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """向量化的切比雪夫距离"""
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    return np.max(np.abs(X_exp - Y_exp), axis=2)

def canberra_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的堪培拉距离"""
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    numerator = np.abs(X_exp - Y_exp)
    denominator = np.abs(X_exp) + np.abs(Y_exp) + epsilon
    return np.sum(numerator / denominator, axis=2)

def minkowski_distance_vectorized(X: np.ndarray, Y: np.ndarray, p: float = 2) -> np.ndarray:
    """向量化的闵可夫斯基距离"""
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    power_sum = np.sum(np.abs(X_exp - Y_exp) ** p, axis=2)
    return power_sum ** (1 / p)

def sqeuclidean_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """向量化的平方欧几里得距离"""
    X_sq = np.sum(X**2, axis=1, keepdims=True)
    Y_sq = np.sum(Y**2, axis=1)
    dot_product = np.dot(X, Y.T)
    return X_sq + Y_sq - 2 * dot_product

def braycurtis_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的Bray-Curtis距离"""
    X_exp = X[:, np.newaxis, :]
    Y_exp = Y[np.newaxis, :, :]
    numerator = np.sum(np.abs(X_exp - Y_exp), axis=2)
    denominator = np.sum(np.abs(X_exp) + np.abs(Y_exp), axis=2) + epsilon
    return numerator / denominator

# ==================== 其他重要距离函数 ====================

def kl_divergence_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的KL散度"""
    # 归一化为概率分布
    X_norm = X / (np.sum(X, axis=1, keepdims=True) + epsilon)
    Y_norm = Y / (np.sum(Y, axis=1, keepdims=True) + epsilon)
    
    # 扩展维度进行广播计算
    X_exp = X_norm[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = Y_norm[np.newaxis, :, :]  # (1, n, d)
    
    # 计算KL散度: sum(p * log(p/q))
    log_ratio = np.log(X_exp / (Y_exp + epsilon) + epsilon)
    kl = np.sum(X_exp * log_ratio, axis=2)
    
    return np.where(kl < 0, 0, kl)  # 确保非负

def jensen_shannon_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的Jensen-Shannon散度"""
    def _vectorized_kl_divergence(P: np.ndarray, Q: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        """
        向量化的KL散度计算辅助函数
        """
        # 避免log(0)和除0
        P_safe = np.clip(P, epsilon, None)
        Q_safe = np.clip(Q, epsilon, None)
        
        # 计算KL(P||Q) = sum(P * log(P/Q))
        log_ratio = np.log(P_safe / Q_safe)
        kl = np.sum(P_safe * log_ratio, axis=2)
        
        # 确保非负
        return np.maximum(kl, 0)
    
    # 确保非负并归一化为概率分布
    X_pos = np.clip(X, 0, None) + epsilon
    Y_pos = np.clip(Y, 0, None) + epsilon
    
    X_prob = X_pos / np.sum(X_pos, axis=1, keepdims=True)
    Y_prob = Y_pos / np.sum(Y_pos, axis=1, keepdims=True)
    
    # 扩展维度进行广播计算
    # X_prob: (m, d) -> (m, 1, d)
    # Y_prob: (n, d) -> (1, n, d)
    P = X_prob[:, np.newaxis, :]  # (m, 1, d)
    Q = Y_prob[np.newaxis, :, :]  # (1, n, d)
    
    # 计算中间分布 M = (P + Q) / 2
    M = 0.5 * (P + Q)
    
    # 计算KL(P || M)
    kl_pm = _vectorized_kl_divergence(P, M, epsilon)
    
    # 计算KL(Q || M)  
    kl_qm = _vectorized_kl_divergence(Q, M, epsilon)
    
    # JS散度 = 0.5 * (KL(P||M) + KL(Q||M))
    js_divergence = 0.5 * (kl_pm + kl_qm)
    
    # JS距离 = sqrt(JS散度)
    js_distance = np.sqrt(js_divergence)
    
    return js_distance

def pearson_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的皮尔逊距离"""
    # 中心化
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    Y_centered = Y - np.mean(Y, axis=1, keepdims=True)
    
    # 计算标准差
    X_std = np.std(X, axis=1, keepdims=True) + epsilon
    Y_std = np.std(Y, axis=1, keepdims=True) + epsilon
    
    # 归一化
    X_norm = X_centered / X_std
    Y_norm = Y_centered / Y_std
    
    # 计算相关系数矩阵
    correlation = np.dot(X_norm, Y_norm.T) / X.shape[1]
    
    return 1 - correlation

def chi_square_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的卡方距离"""
    X_exp = X[:, np.newaxis, :] + epsilon
    Y_exp = Y[np.newaxis, :, :] + epsilon
    
    numerator = (X_exp - Y_exp) ** 2
    denominator = X_exp + Y_exp
    
    return np.sum(numerator / denominator, axis=2)

def bhattacharyya_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的Bhattacharyya距离"""
    # 归一化为概率分布
    X_norm = X / (np.sum(X, axis=1, keepdims=True) + epsilon)
    Y_norm = Y / (np.sum(Y, axis=1, keepdims=True) + epsilon)
    
    X_exp = X_norm[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = Y_norm[np.newaxis, :, :]  # (1, n, d)
    
    # 计算Bhattacharyya系数
    bc = np.sum(np.sqrt(X_exp * Y_exp), axis=2)
    
    # 转换为距离
    return -np.log(bc + epsilon)

def hellinger_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的Hellinger距离"""
    # 归一化为概率分布
    X_norm = X / (np.sum(X, axis=1, keepdims=True) + epsilon)
    Y_norm = Y / (np.sum(Y, axis=1, keepdims=True) + epsilon)
    
    X_exp = np.sqrt(X_norm)[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = np.sqrt(Y_norm)[np.newaxis, :, :]  # (1, n, d)
    
    # 计算Hellinger距离
    diff = X_exp - Y_exp
    return np.sqrt(0.5 * np.sum(diff ** 2, axis=2))

def sorensen_dice_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的Sørensen-Dice距离"""
    X_exp = X[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = Y[np.newaxis, :, :]  # (1, n, d)
    
    intersection = 2 * np.sum(np.minimum(X_exp, Y_exp), axis=2)
    union = np.sum(X, axis=1, keepdims=True) + np.sum(Y, axis=1) + epsilon
    
    similarity = intersection / union
    return 1 - similarity

def wasserstein_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    向量化的Wasserstein距离（1阶）
    注意：这是近似实现，真正的Wasserstein距离计算更复杂
    """
    # 对每个向量进行排序
    X_sorted = np.sort(X, axis=1)
    Y_sorted = np.sort(Y, axis=1)
    
    X_exp = X_sorted[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = Y_sorted[np.newaxis, :, :]  # (1, n, d)
    
    # 计算排序后向量的平均绝对差
    return np.mean(np.abs(X_exp - Y_exp), axis=2)

def total_variation_distance_vectorized(X: np.ndarray, Y: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """向量化的总变差距离"""
    # 归一化为概率分布
    X_norm = X / (np.sum(X, axis=1, keepdims=True) + epsilon)
    Y_norm = Y / (np.sum(Y, axis=1, keepdims=True) + epsilon)
    
    X_exp = X_norm[:, np.newaxis, :]  # (m, 1, d)
    Y_exp = Y_norm[np.newaxis, :, :]  # (1, n, d)
    
    # 计算总变差距离
    return 0.5 * np.sum(np.abs(X_exp - Y_exp), axis=2)

def energy_distance_vectorized(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    高效的能量距离计算
    """
    m, d1 = X.shape
    n, d2 = Y.shape
    
    # 计算X内部的距离矩阵
    XX = np.sum(X**2, axis=1, keepdims=True)
    XY = np.dot(X, Y.T)
    YY = np.sum(Y**2, axis=1, keepdims=True).T
    
    # 计算欧几里得距离矩阵
    dist_xy = np.sqrt(XX - 2*XY + YY)
    
    # 计算X内部的平均距离
    if m > 1:
        XX_full = np.sqrt(XX + XX.T - 2 * np.dot(X, X.T))
        mean_xx = np.mean(XX_full)
    else:
        mean_xx = 0.0
    
    # 计算Y内部的平均距离
    if n > 1:
        YY_full = np.sqrt(YY.T + YY - 2 * np.dot(Y, Y.T))
        mean_yy = np.mean(YY_full)
    else:
        mean_yy = 0.0
    
    # 计算energy距离: 2*E[|X-Y|] - E[|X-X'|] - E[|Y-Y'|]
    return 2 * dist_xy - mean_xx - mean_yy


# ==================== 统一的向量化距离计算接口 ====================

def vectorized_distance_matrix(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = 'euclidean',
    **kwargs
) -> np.ndarray:
    """
    完全向量化的距离矩阵计算
    
    支持的度量:
    - cdist度量: 'cosine', 'correlation', 'euclidean', 'manhattan', 
                'chebyshev', 'canberra', 'minkowski', 'sqeuclidean', 'braycurtis'
    - 其他度量: 'kl', 'jensenshannon', 'pearson', 'chi-square', 'bhattacharyya',
               'hellinger', 'sorensen_dice', 'wasserstein', 'total_variation', 'energy'
    """
    if Y is None:
        Y = X
    
    metric = metric.lower()
    epsilon = kwargs.get('epsilon', 1e-10)
    
    # cdist度量
    cdist_metrics = {
        'cosine': cosine_distance_vectorized,
        'correlation': correlation_distance_vectorized,
        'euclidean': euclidean_distance_vectorized,
        'manhattan': manhattan_distance_vectorized,
        'l1': manhattan_distance_vectorized,
        'l2': euclidean_distance_vectorized,
        'chebyshev': chebyshev_distance_vectorized,
        'canberra': canberra_distance_vectorized,
        'minkowski': lambda x, y: minkowski_distance_vectorized(x, y, kwargs.get('p', 2)),
        'sqeuclidean': sqeuclidean_distance_vectorized,
        'braycurtis': braycurtis_distance_vectorized
    }
    
    # 其他度量
    other_metrics = {
        'kl': kl_divergence_vectorized,
        'jensenshannon': jensen_shannon_vectorized,
        'js': jensen_shannon_vectorized,
        'pearson': pearson_distance_vectorized,
        'chi-square': chi_square_distance_vectorized,
        'bhattacharyya': bhattacharyya_distance_vectorized,
        'hellinger': hellinger_distance_vectorized,
        'sorensen_dice': sorensen_dice_distance_vectorized,
        'sorensen': sorensen_dice_distance_vectorized,
        'dice': sorensen_dice_distance_vectorized,
        'wasserstein': wasserstein_distance_vectorized,
        'total_variation': total_variation_distance_vectorized,
        'energy': energy_distance_vectorized
    }
    
    all_metrics = {**cdist_metrics, **other_metrics}
    
    if metric in all_metrics:
        return all_metrics[metric](X, Y)
    else:
        raise ValueError(f"不支持的度量: {metric}。支持的度量: {list(all_metrics.keys())}")

# ==================== 一维向量的计算接口 ====================

from scipy.spatial.distance import (
    jensenshannon,
    cosine,
    correlation,
    chebyshev,
    canberra
)

def distance_1d(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = 'l1',
    epsilon = 1e-10,
    **kwargs
) -> np.ndarray:
    assert X.shape == Y.shape
    assert len(X.shape) == 1
    
    # mask = np.logical_and(X!=0, Y!=0)
    # X = X[mask] + 1e-15
    # Y = Y[mask] + 1e-15
    
    X = X + 1e-15
    Y = Y + 1e-15
    
    X = X / np.sum(X)
    Y = Y / np.sum(Y)
    
    if metric == 'cosine':
        return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    elif metric == 'correlation':
        return correlation(X, Y)
    elif metric == 'l1':
        return np.sum(np.abs(X - Y))
    elif metric == 'l2':
        return np.sqrt(np.sum(np.abs(X - Y) ** 2))
    elif metric == 'chebyshev':
        return chebyshev(X, Y)
    elif metric == 'canberra':
        return np.sum(np.abs(X - Y) / (np.abs(X) + np.abs(Y)) + epsilon)
    elif metric == 'js':
        dist = jensenshannon(X, Y)
        if np.isnan(dist):
            return 0
        return dist
    elif metric == 'chi-square':
        return np.sum((X - Y) ** 2 / (X + Y + 1e-10))  # 避免除以零
    elif metric == 'bhattacharyya':
        return -np.log(np.sum(np.sqrt(X * Y)))
    elif metric == 'hellinger':
        return np.sqrt(0.5 * np.sum((np.sqrt(X) - np.sqrt(Y)) ** 2))
    elif metric == 'wasserstein':  # 近似计算
        return np.mean(np.abs(np.sort(X) - np.sort(Y)))
    elif metric == 'total_variation':
        X_norm = X / (np.sum(X) + epsilon)
        Y_norm = Y / (np.sum(Y) + epsilon)
        return 0.5 * np.sum(np.abs(X_norm - Y_norm))
    else:
        raise ValueError(f"不支持的1d距离函数{metric}。")


# ==================== 性能测试和验证 ====================

def test_vectorized_performance():
    """性能测试"""
    import time
    np.random.seed(42)
    
    # 测试数据
    m, n, d = 1, 1, 30000
    X = np.random.rand(m, d)
    Y = np.random.rand(n, d)
    
    print(f"测试数据: X={X.shape}, Y={Y.shape}")
    print("=" * 60)
    
    # 测试各种距离度量
    test_metrics = [
        'cosine', 'correlation', 'euclidean', 'manhattan',
        'chebyshev', 'canberra', 'minkowski', 'sqeuclidean',
        'braycurtis',
        
        'l1', 'l2',
        
        'kl', 'jensenshannon', 'js', 'pearson', 'chi-square',
        'bhattacharyya', 'hellinger', 'sorensen_dice', 'sorensen',
        'dice', 'wasserstein', 'total_variation', 'energy'
    ]
    
    for metric in test_metrics:
        start_time = time.time()
        dist_matrix = vectorized_distance_matrix(X, Y, metric=metric)
        elapsed = time.time() - start_time
        
        print(f"{metric:15s}: {elapsed:.4f}s, 形状 {dist_matrix.shape}")
        
        # 小数据验证
        if m == n == 1:
            from scipy.spatial.distance import cdist
            if metric in ['cosine', 'correlation','canberra','js',
                          'chebyshev']:
                result = cdist(X, Y, metric=metric)
            elif metric == 'kl':
                from scipy.stats import entropy
                result = entropy(X[0], Y[0])
            elif metric == 'energy':
                from scipy.stats import energy_distance
                result = energy_distance(X[0], Y[0])
            elif metric == 'wasserstein':
                from scipy.stats import wasserstein_distance
                result = wasserstein_distance(X[0], Y[0])
            else:
                continue
            if np.allclose(dist_matrix, result, atol=1e-6):
                print("  ✓ 验证通过")
            else:
                max_diff = np.max(np.abs(dist_matrix - result))
                print(f"  ⚠ 最大差异: {max_diff:.6f}")


if __name__ == "__main__":
    # 使用示例
    # X = np.random.rand(5, 10)
    # Y = np.random.rand(3, 10)
    
    # print("向量化距离计算示例:")
    # for metric in ['euclidean', 'cosine', 'pearson', 'bhattacharyya']:
    #     dist = vectorized_distance_matrix(X, Y, metric=metric)
    #     print(f"\n{metric} 距离:")
    #     print(dist)
    
    # 性能测试
    # test_vectorized_performance()
    
    X = np.random.rand(1, 10)
    Y = np.random.rand(1, 10)
    
    print(vectorized_distance_matrix(X, Y, 'l1')[0,0])
    