import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
from loguru import logger
from tqdm import tqdm

np.random.seed(42)


def dist_validate(arr, indices, num_samples=1000, random_seed=None):
    np.random.seed(random_seed)  # 设置随机种子（可选）

    all_pairs = np.random.choice(num_samples, size=(num_samples, 2), replace=True)
    all_pairs = np.unique(all_pairs, axis=0)  # 去重（避免 i == j）
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]  # 确保 i != j

    logger.info("calculating baseline...")
    sampled_distances = distance.cdist(arr[all_pairs[:, 0]], arr[all_pairs[:, 1]], "euclidean").diagonal()
    avg_distance_arr = np.mean(sampled_distances)

    logger.info("calculating selected points...")
    selected_rows = np.squeeze(arr[indices])
    # print(selected_rows.shape)
    dists_indices = distance.pdist(selected_rows, "euclidean")
    avg_distance_indices = np.mean(dists_indices)

    return avg_distance_arr, avg_distance_indices


class Initializer:
    def __init__(self, numerical_data: np.array = None, name_data: np.array = None):
        self.numerical_data = numerical_data
        self.name_data = name_data
        assert type(numerical_data) != None or type(name_data) != None, "Please provide ether numerical data or name data"

    def sampling(self, method="LHS", batch_size=5, random_seed=42):
        self.batch_size = batch_size
        np.random.seed(random_seed)
        logger.info(f"sampling with method: {method}")
        match method:
            case "LHS":
                selected_indices = self.lhs_sampling()
            case "sobol":
                selected_indices = self.sobel_sequence_sampling()
            case "kmeans":
                selected_indices = self.kmeans_sampling()
            case "cvt":
                selected_indices = self.cvt_sampling()
            case "hypersphere":
                selected_indices = self.hypersphere_sampling()
            case "minmax":
                selected_indices = self.min_max_sampling()
            case "random":
                return np.random.randint(0, len(self.name_data), batch_size)
            case _:
                raise ValueError("Invalid method")

        if len(selected_indices) > 1:
            ave_dist, select_ave_dist = dist_validate(self.numerical_data, selected_indices)
        logger.info(f"average distance is {ave_dist:.2f}, and selected average distance is {select_ave_dist:.2f}.")
        selected_conditions = self.name_data[selected_indices].squeeze()
        return selected_conditions

    def lhs_sampling(self):
        from pyDOE import lhs

        lhs_samples = lhs(self.numerical_data.shape[1], samples=self.batch_size, criterion="maximin")
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.numerical_data)
        _, indices = nbrs.kneighbors(lhs_samples)
        return indices

    def cvt_sampling(self):
        from sklearn.decomposition import PCA
        from scipy.spatial import Voronoi, KDTree
        import numpy as np

        # 降维到50维（可调整）
        pca = PCA(n_components=10)
        data_lowdim = pca.fit_transform(self.numerical_data)
        # 随机初始化种子点（降维后）
        if data_lowdim.shape[0] <= self.batch_size:
            return np.arange(data_lowdim.shape[0])
        seeds = data_lowdim[np.random.choice(data_lowdim.shape[0], self.batch_size, replace=False)]
        # 迭代优化（简化版）
        for _ in tqdm(range(50)):
            kdtree = KDTree(data_lowdim)
            _, regions = kdtree.query(seeds, k=100)  # 每个种子找最近100个点作为区域
            new_seeds = np.array([data_lowdim[r].mean(axis=0) for r in regions])
            seeds = new_seeds
        # 返回原始高维空间的最近邻索引
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.numerical_data)
        _, indices = nbrs.kneighbors(pca.inverse_transform(seeds))  # 将种子映射回高维
        return indices.flatten()

    def kmeans_sampling(self):
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=self.batch_size, random_state=42).fit(self.numerical_data)
        nbrs = NearestNeighbors(n_neighbors=1).fit(self.numerical_data)
        _, indices = nbrs.kneighbors(kmeans.cluster_centers_)
        return indices.flatten()

    def sobel_sequence_sampling(self):
        from botorch.utils.sampling import draw_sobol_samples
        import torch

        data = torch.as_tensor(self.numerical_data, dtype=torch.float32)
        # 在 301-D 单位超立方体中生成 Sobol 点
        sobol_points = draw_sobol_samples(bounds=torch.tensor([[0.0] * 301, [1.0] * 301]), n=self.batch_size, q=1).squeeze(1)
        # 最近邻搜索：找到 data_normalized 中最接近 sobol_points 的点
        nbrs = NearestNeighbors(n_neighbors=1).fit(data.numpy())
        _, indices = nbrs.kneighbors(sobol_points.numpy())
        indices = torch.from_numpy(indices).squeeze(1)  # (batch_size,)

        return indices

    def min_max_sampling(self):
        selected_indices = []

        # First select min and max
        min_idx = np.argmin(self.numerical_data)
        max_idx = np.argmax(self.numerical_data)

        selected_indices.extend([min_idx, max_idx])

        # If we only need 2 samples, return them
        if self.batch_size == 2:
            return sorted(selected_indices)

        # For remaining samples, iteratively select points that maximize the minimum distance
        remaining_indices = set(range(len(self.numerical_data))) - set(selected_indices)

        remaining_to_select = self.batch_size - len(selected_indices)
        with tqdm(total=remaining_to_select, desc="Selecting batch") as pbar:
            while len(selected_indices) < self.batch_size:
                max_min_dist = -1
                best_idx = -1
                for candidate in remaining_indices:
                    # Calculate minimum distance to already selected points
                    min_dist = min(np.linalg.norm(self.numerical_data[candidate] - self.numerical_data[s]) for s in selected_indices)

                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = candidate

                if best_idx != -1:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    pbar.update(1)

        return np.array([int(s) for s in selected_indices])
