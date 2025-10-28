"""Initialization methods for reaction optimization.

Modern initialization strategies with rich progress indicators.
"""

from typing import Literal, Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from tqdm import tqdm
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

console = Console()
np.random.seed(42)


def dist_validate(arr: np.ndarray, indices: np.ndarray, num_samples: int = 1000, random_seed: Optional[int] = None) -> tuple[float, float]:
    """Validate distance distribution of selected points.

    Args:
        arr: Full array of points
        indices: Indices of selected points
        num_samples: Number of samples for baseline calculation
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (baseline_avg_distance, selected_avg_distance)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate unique pairs for baseline calculation
    all_pairs = np.random.choice(num_samples, size=(num_samples, 2), replace=True)
    all_pairs = np.unique(all_pairs, axis=0)
    all_pairs = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        baseline_task = progress.add_task("Calculating baseline distances...", total=None)
        sampled_distances = distance.cdist(arr[all_pairs[:, 0]], arr[all_pairs[:, 1]], "euclidean").diagonal()
        avg_distance_arr = np.mean(sampled_distances)
        progress.update(baseline_task, description="Baseline calculated")

        selected_task = progress.add_task("Calculating selected distances...", total=None)
        selected_rows = np.squeeze(arr[indices])
        dists_indices = distance.pdist(selected_rows, "euclidean")
        avg_distance_indices = np.mean(dists_indices)
        progress.update(selected_task, description="Selected distances calculated")

    return avg_distance_arr, avg_distance_indices


class Initializer:
    """Modern initializer for reaction conditions.

    Provides various sampling strategies with rich progress indicators.

    Args:
        numerical_data: Numerical descriptor array
        name_data: Name array corresponding to conditions

    Raises:
        ValueError: If neither numerical nor name data provided
    """

    def __init__(self, numerical_data: Optional[np.ndarray] = None, name_data: Optional[np.ndarray] = None) -> None:
        if numerical_data is None and name_data is None:
            raise ValueError("Please provide either numerical data or name data")

        self.numerical_data = numerical_data
        self.name_data = name_data

    def sampling(
        self, method: Literal["LHS", "sobol", "kmeans", "cvt", "hypersphere", "random"] = "LHS", batch_size: int = 5, random_seed: int = 42
    ) -> np.ndarray:
        """Sample initial conditions using specified method.

        Args:
            method: Sampling strategy to use
            batch_size: Number of samples to generate
            random_seed: Random seed for reproducibility

        Returns:
            Array of selected condition combinations

        Raises:
            ValueError: If unknown sampling method specified
        """
        self.batch_size = batch_size
        np.random.seed(random_seed)

        console.print(f"Sampling {batch_size} conditions using {method} method", style="cyan")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Sampling with {method}...", total=None)

            match method.lower():
                case "lhs":
                    selected_indices = self.lhs_sampling()
                case "sobol":
                    selected_indices = self.sobel_sequence_sampling()
                case "kmeans":
                    selected_indices = self.kmeans_sampling()
                case "cvt":
                    selected_indices = self.cvt_sampling()
                case "hypersphere":
                    selected_indices = self.hypersphere_sampling()
                case "random":
                    selected_indices = self.random_sampling()
                case _:
                    raise ValueError(f"Unknown sampling method: {method}")

            progress.update(task, description=f"Sampling complete - {method}")

        selected_conditions = self.name_data[selected_indices].squeeze()

        return selected_conditions

    def random_sampling(self) -> np.ndarray:
        """Simple random sampling."""
        return np.random.randint(0, len(self.name_data), self.batch_size)

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
        for _ in range(50):
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
        sobol_points = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * data.shape[1], [1.0] * data.shape[1]]), n=self.batch_size, q=1
        ).squeeze(1)
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
