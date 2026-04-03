确保描述符空间中没有相同的分子

调试reagent boost机制

一个batch中，要既有探索，又有利用，避免过度贪婪

建立真实反应的benchmark

解决超大化学空间的BO问题？

可能要 聚类 - 优化 - 聚类？

- 相比起探索pareto frontiers，单纯的优化问题或许更希望快速探索到一个特定的结果（for example: yield = 60%, ee = 90%）

1. new evaluate targets

    - [ ] average optimal targets (like 10 random intial sampling)

    - [ ] AUC

    - [ ] confidence (like, in one batch, should have some `exploit` results and `exploration` results. `exploit` results to give chemists confidence)

    - [ ] optimization convergence (like, if easy to determine when to stop. like the EHVI will jump a lot or not)

2. evaluate with different algorithm

- [ ] 随机梯度下降

- [ ] 进化/遗传算法

- [ ] 粒子群算法

- [ ] 人工优化？


3. descriptor influence

    - [ ]

4. introduce LLM for space reduction

5. 

target bias effects on optimization results

