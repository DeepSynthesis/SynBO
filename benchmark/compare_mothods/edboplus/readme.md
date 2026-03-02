# EDBO+ Benchmark类工作原理总结

## 一、Benchmark类运转机制

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Benchmark工作流程                           │
└─────────────────────────────────────────────────────────────────────┘

1. 初始化阶段
   ├─ 加载完整数据集（ground truth）
   ├─ 计算真实Pareto前沿和hypervolume
   ├─ 创建EDBO+优化器实例
   └─ 生成不含目标值的初始反应范围文件

2. 优化循环（迭代执行）
   ┌─────────────────────────────────────────┐
   │  每个迭代执行以下步骤：               │
   │                                   │
   │ ① 初始采样（首次迭代）         │
   │    - 使用random/lhs/cvtsampling       │
   │    - 选择batch_size个初始样本    │
   │                                   │
   │ ② 训练代理模型                    │
   │    - 使用Gaussian Process           │
   │    - 为每个目标训练独立模型         │
   │                                   │
   │ ③ 优化采集函数                    │
   │    - EHVI: Expected Hypervolume  │
   │    - MOUCB: Multi-Objective UCB     │
   │    - MOGreedy: 贪婪策略           │
   │    - 选择batch_size个新实验       │
   │                                   │
   │ ④ 模拟实验（从ground truth获取） │
   │    - 为选中的样本分配真实目标值   │
   │    - 更新训练数据集               │
   │                                   │
   │ ⑤ 评估性能                        │
   │    - 计算当前Pareto前沿          │
   │    - 计算hypervolume完成度       │
   │    - 记录最佳目标值              │
   │    - 计算预测误差                │
   │                                   │
   │ ⑥ 保存结果                        │
   │    - 记录该步骤的所有指标          │
   │                                   │
   └─────────────────────────────────────────┘

3. 重复步骤2，直到达到指定迭代次数（steps）
```

## 二、输入参数详解

### 构造函数参数（初始化时）

| 参数名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| **df_ground** | `pd.DataFrame` | **完整数据集**，包含所有特征列和目标列 | 1728行×8列的DataFrame |
| **index_column** | `str` | 用于追踪实验的索引列名 | `'new_index'` |
| **objective_names** | `list[str]` | 要优化的目标列名列表 | `['yield', 'cost']` |
| **objective_modes** | `list[str]` | 每个目标的优化模式（'max'或'min'） | `['max', 'min']` |
| **objective_thresholds** | `list[Optional[float]]` | 每个目标的最差值阈值（用于hypervolume计算） | `[None, None]` |
| **features_regression** | `list[str]` | 用于回归模型的特征列名列表 | `['base', 'ligand', 'solvent', ...]` |
| **filename** | `str` | 临时反应范围文件名 | `'benchmark.csv'` |
| **filename_results** | `str` | 结果保存文件名 | `'results_benchmark.csv'` |
| **acquisition_function** | `str` | 采集函数类型 | `'EHVI'`, `'MOUCB'`, `'MOGreedy'` |

### 运行方法参数（bench.run()）

| 参数名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| **steps** | `int` | 迭代次数（总共要运行的批次数） | `12` |
| **batch** | `int` | 每批次建议的实验数 | `5` |
| **seed** | `int` | 随机种子，用于结果复现 | `1` |
| **init_method** | `str` | 初始采样方法 | `'cvtsampling'`, `'lhs'`, `'seed'` |
| **plot_ground** | `bool` | 是否绘制ground truth | `False` |
| **plot_predictions** | `bool` | 是否绘制预测结果 | `False` |
| **plot_train** | `bool` | 是否绘制训练过程 | `False` |

## 三、输出详解

### 1. 控制台输出（实时）

```
初始化阶段：
├─ High trade-off ground truth: [[...]]  # 高trade-off点
├─ Ground truth hypervolume: 0.991471       # 真实hypervolume值
└─ Number of Pareto optimal points: 9         # Pareto最优解数量

每次迭代：
├─ Best yield found: 76.02                     # 当前找到的最佳yield
├─ Best cost found: 0.028582675               # 当前找到的最佳cost
├─ Total number of experiments: 10                # 累计实验数
├─ Hypervolume train (%): 73.85                 # 相对真实的hypervolume百分比
├─ Maximin distance to Pareto: 23.98           # 到Pareto前沿的最大距离
└─ Maximin distance to Tradeoff: 10.44          # 到trade-off点的距离

预测误差：
├─ MAE_yield: 21.88                             # 平均绝对误差
├─ RMSE_yield: 25.23                            # 均方根误差
├─ R2_yield: -0.05                              # R²分数
└─ (对每个目标都有类似指标)
```

### 2. CSV文件输出

#### (1) 反应范围文件 (`{filename}.csv`)
```csv
new_index,base,ligand,solvent,concentration,temperature,yield,cost,priority
966,KOPiv,P(fur)3,DMAc,0.1,105,73.59,0.0377,1.0
```
- 包含所有反应条件
- `priority=1.0`: 建议进行的实验
- `priority=-1.0`: 已完成的实验
- `priority=0.0`: 未被选中的实验

#### (2) 结果文件 (`results_{filename_results}.csv`)
```csv
step,n_experiments,hypervolume_ground,hypervolume_sampled,
hypervolume completed (%),yield_best,cost_best,
dmaximin_pareto,dmaximin_tradeoff,...
0,5,0.991471,0.708429,71.49,73.59,0.02858,34.80,12.87,...
1,10,0.991471,0.732014,73.85,76.02,0.02858,23.98,10.44,...
```
**主要列说明：**
- `step`: 当前迭代步数（0, 1, 2, ...）
- `n_experiments`: 累计实验数量
- `hypervolume_ground`: 真实的hypervolume值（固定）
- `hypervolume_sampled`: 当前采样集的hypervolume值
- `hypervolume completed (%)`: 完成百分比 = `sampled/ground * 100`
- `{obj}_best`: 每个目标的当前最佳值
- `dmaximin_pareto`: 到真实Pareto前沿的Hausdorff距离
- `dmaximin_tradeoff`: 到真实trade-off点的距离
- `MAE_{obj}`, `RMSE_{obj}`, `R2_{obj}`: 预测误差指标

#### (3) 预测文件 (`pred_{filename}.csv`)
```csv
new_index,base,ligand,...,yield,priority,
yield_predicted_mean,yield_predicted_std_dev,yield_expected_improvement,
cost_predicted_mean,cost_predicted_std_dev,cost_expected_improvement
0,KOPiv,P(fur)3,...,73.59,1.0,
75.2,5.3,0.8,0.025,0.01,0.15
```
- `yield_predicted_mean`: GP模型的预测均值
- `yield_predicted_std_dev`: 预测标准差（不确定性）
- `yield_expected_improvement`: 期望改进值

## 四、关键数据流

```
┌──────────────────────────────────────────────────────────────┐
│                  数据流向图                           │
└──────────────────────────────────────────────────────────────┘

Ground Truth (完整数据集)
    │
    ├── 计算Pareto前沿和Hypervolume
    │
    └── 创建初始反应范围（不含目标值）
              │
              ↓
         优化循环
              │
    ┌─────────┴─────────┬─────────┐
    │                 │         │
  训练数据       候选集     新批次
    │                 │         │
    ↓                 ↓         ↓
  GP模型         优化采集   选中样本
    │                           │
    └───────────────┬───────────┘
                    │
                    ↓
               模拟实验（从ground truth获取真实值）
                    │
                    ↓
               更新训练数据
                    │
                    ↓
               评估并记录结果
                    │
                    └─────→ 下一迭代
```

## 五、使用示例（基于run.py）

```python
from edbo.plus.benchmark.multiobjective_benchmark import Benchmark
import pandas as pd

# 1. 加载数据
df_exp = pd.read_csv('../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv')

# 2. 定义特征和目标
features = ['base', 'ligand', 'solvent', 'concentration', 'temperature']
objectives = ['yield', 'cost']
modes = ['max', 'min']

# 3. 初始化Benchmark
bench = Benchmark(
    df_ground=df_exp,
    index_column='new_index',
    objective_names=objectives,
    objective_modes=modes,
    objective_thresholds=[None, None],
    features_regression=features,
    filename='benchmark.csv',
    filename_results='results.csv',
    acquisition_function='EHVI'
)

# 4. 运行优化
bench.run(
    steps=12,              # 12个迭代
    batch=5,               # 每步5个实验
    seed=1,                # 随机种子
    init_method='cvtsampling',  # CVT采样
    plot_ground=False,
    plot_predictions=False,
    plot_train=False
)

# 5. 读取结果
results = pd.read_csv('results/results.csv')
print(f"最终hypervolume完成度: {results.iloc[-1]['hypervolume completed (%)']:.2f}%")
```

## 六、核心算法

### Hypervolume计算
- **作用**: 衡量Pareto前沿的综合质量
- **参考点**: 每个目标的最差值（或用户指定阈值）
- **单调性**: 假设最大化，所有目标归一化到[0,1]

### EHVI (Expected Hypervolume Improvement)
- **原理**: 预测每个候选样本的hypervolume期望改进
- **优势**: 自动平衡探索和利用
- **适用场景**: 多目标优化，无实验噪声

### CVT采样 (Centroidal Voronoi Tessellation)
- **原理**: 将空间划分为Voronoi单元，均匀采样
- **优势**: 比随机采样更均匀覆盖空间
- **适用场景**: 初始样本选择