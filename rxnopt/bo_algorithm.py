import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf_discrete


class ProxyModel(ABC):
    """模型基类，定义统一接口"""

    @abstractmethod
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回预测均值与标准差"""
        pass


class GPModel(ProxyModel):
    """高斯过程实现示例"""

    def __init__(self):
        from gpytorch.models import ExactGP

        self.model = None

    def fit(self, X, y):
        # 实际实现需包含核函数选择、超参数优化等
        from botorch.models import SingleTaskGP

        self.model = SingleTaskGP(X, y).eval()

    def predict(self, X):
        posterior = self.model.posterior(X)
        return posterior.mean, posterior.variance.sqrt()


class BNNModel(ProxyModel):
    """贝叶斯神经网络实现示例（MC Dropout）"""

    def __init__(self, n_samples: int = 50):
        import torch.nn as nn

        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Dropout(0.1), nn.ReLU(), nn.Linear(64, 64), nn.Dropout(0.1), nn.ReLU(), nn.Linear(64, 1)
        )
        self.n_samples = n_samples

    def fit(self, X, y, epochs=100):
        optimizer = torch.optim.Adam(self.net.parameters())
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.MSELoss()(self.net(X), y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.net.train()  # 保持Dropout激活
        with torch.no_grad():
            preds = torch.cat([self.net(X) for _ in range(self.n_samples)], dim=1)
        return preds.mean(dim=1), preds.std(dim=1)


class AcquisitionFunctionBase(ABC):
    """采集函数基类"""

    @abstractmethod
    def evaluate(self, model: ProxyModel, X: torch.Tensor) -> torch.Tensor:
        """计算候选点的采集函数值"""
        pass


class EHVI(AcquisitionFunctionBase):
    """多目标EHVI采集函数"""

    def __init__(self, ref_point: List[float]):
        self.ref_point = torch.tensor(ref_point)

    def evaluate(self, model, X):
        from botorch.acquisition import qExpectedHypervolumeImprovement

        mean, std = model.predict(X)
        # 简化为单点EHVI计算（实际需处理多目标）
        return mean + 2 * std  # 示例简化逻辑


class UCB(AcquisitionFunctionBase):
    """上置信界采集函数"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def evaluate(self, model, X):
        mean, std = model.predict(X)
        return mean + self.beta * std


class BayesianOptimizer:
    """核心优化框架"""

    def __init__(
        self,
        model: ProxyModel,
        acq_func: AcquisitionFunctionBase,
        candidates: torch.Tensor,
    ):
        self.model = model
        self.acq_func = acq_func
        self.candidates = candidates  # 离散候选点集

    def update_observations(self, X: torch.Tensor, y: torch.Tensor):
        """更新观测数据"""
        self.model.fit(X, y)

    def suggest_next_point(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """推荐下一个实验点"""
        acq_values = self.acq_func.evaluate(self.model, self.candidates)
        best_idx = torch.argmax(acq_values)
        return self.candidates[best_idx], acq_values[best_idx]


# 使用示例
if __name__ == "__main__":
    # 1. 初始化组件
    model = GPModel()  # 可替换为BNNModel()
    acq_func = UCB(beta=2.0)  # 可替换为EHVI(ref_point=[0, 0])
    candidates = torch.rand(100, 2)  # 100个2维候选点

    # 2. 创建优化器
    optimizer = BayesianOptimizer(model, acq_func, candidates)

    # 3. 模拟观测数据
    X_observed = torch.rand(5, 2)
    y_observed = torch.sin(X_observed).sum(dim=1, keepdim=True)

    # 4. 迭代优化
    for i in range(10):
        optimizer.update_observations(X_observed, y_observed)
        next_point, acq_value = optimizer.suggest_next_point()
        print(f"Iter {i}: Next point = {next_point}, ACQ = {acq_value:.3f}")

        # 模拟新观测（实际应从实验获取）
        new_y = torch.sin(next_point).sum().view(1, 1)
        X_observed = torch.cat([X_observed, next_point.unsqueeze(0)])
        y_observed = torch.cat([y_observed, new_y])
