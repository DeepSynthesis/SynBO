


# Dataset Information

## Buchwald-Hartwig reaction HTE dataset

![alt text](assets/B-H_HTE.png)

- paper: Torres, J. A. G.; Lau, S. H.; Anchuri, P.; Stevens, J. M.; Tabora, J. E.; Li, J.; Borovika, A.; Adams, R. P.; Doyle, A. G. A Multi-Objective Active Learning Platform and Web App for Reaction Optimization. J. Am. Chem. Soc. 2022, 144 (43), 19999–20007. https://doi.org/10.1021/jacs.2c08592.

- dataset information

| feature           | value                                                  |
|-------------------|--------------------------------------------------------|
| dataset size      | 1728                                                   |
| reagent type      | base, ligand, solvent, concentration, temperature      |
| exp data coverage | 100.0%                                                 |
| target type       | yield, cost                                             |

- mark: have their calculated QM descriptor

## Suzuki reaction HTE dataset

![alt text](assets/suzuki_HTE.png)

- paper: Perera, D.; Tucker, J. W.; Brahmbhatt, S.; Helal, C. J.; Chong, A.; Farrell, W.; Richardson, P.; Sach, N. W. A Platform for Automated Nanomole-Scale Reaction Screening and Micromole-Scale Synthesis in Flow. Science 2018, 359 (6374), 429–434. https://doi.org/10.1126/science.aap9112.

- dataset information

| feature           | value                                      |
|-------------------|--------------------------------------------|
| dataset size      | 5760                                       |
| reagent type      | solvent, ligand, reactant1, reactant2, base |
| exp data coverage | 53.6%                                      |
| target type       | conversion                                 |

## α-asymmetric alkylation of aldehyde with photocatalyst

![alt text](assets/asym_alkylation.png)

- paper: Nie, W.; Wan, Q.; Sun, J.; Chen, M.; Gao, M.; Chen, S. Ultra-High-Throughput Mapping of the Chemical Space of Asymmetric Catalysis Enables Accelerated Reaction Discovery. Nat. Commun. 2023, 14 (1), 1–11. https://doi.org/10.1038/s41467-023-42446-5.

- dataset information

| feature           | value                                      |
|-------------------|--------------------------------------------|
| dataset size      | 1430                                       |
| reagent type      | reaction1, reaction2, catalyst1, Catalyst2 |
| exp data coverage | 100.0%                                     |
| target type       | ee, yield                                  |

## Asymmetric hydrogenation of alkene

![alt text](assets/asym_hydrogenation.png)

- paper: Kalikadien, A. V.; Valsecchi, C.; Putten, R. van; Maes, T.; Muuronen, M.; Dyubankova, N.; Lefort, L.; Pidko, E. A. Probing Machine Learning Models Based on High Throughput Experimentation Data for the Discovery of Asymmetric Hydrogenation Catalysts. Chem. Sci. 2024, 15 (34), 13618–13630. https://doi.org/10.1039/D4SC03647F.

- dataset information

| feature               | value                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| dataset size          | 3168                                                                  |
| reagent type          | reagent, solvent, ligand, metal_amount, ligand_amount, temperature, pressure, time |
| exp data coverage     | 10.3%                                                                 |
| target type           | ee, conversion                                                         |

## C-H arylation reaction HTE dataset

![alt text](assets/C-H_arylation.png)

- paper: 

- dataset information

| feature               | value                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| dataset size          | 1536                                                                  |
| reagent type          | ligand,electrophile,nucleophile                                       |
| exp data coverage     | 100.0%                                                                 |
| target type           | yield                                                         |

## Deoxyfluorination reaction HTE dataset

![alt text](assets/deoxyf.png)

- paper: Nielsen, M. K.; Ahneman, D. T.; Riera, O.; Doyle, A. G. Deoxyfluorination with Sulfonyl Fluorides: Navigating Reaction Space with Machine Learning. J. Am. Chem. Soc. 2018, 140 (15), 5004–5008. https://doi.org/10.1021/jacs.8b01523.

- dataset information

| feature               | value                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| dataset size          | 740                                                                   |
| reagent type          | base,fluoride,substrate                                               |
| exp data coverage     | 100.0%                                                                |
| target type           | yield                                                                 |

## Amide coupling reaction HTE dataset

![alt text](assets/amine_coupling.png)

- paper: 

- dataset information

| feature               | value                                                                 |
|-----------------------|-----------------------------------------------------------------------|
| dataset size          | 960                                                                   |
| reagent type          | solvent,base,activator,nucleophile                            |
| exp data coverage     | 100.0%                                                                |
| target type           | yield                                                                 |

-----

搜集一些已经比较成熟的数据集，方便后续实验使用。

来源：
1. https://github.com/doyle-lab-ucla/bandit-optimization/tree/main/datasets （主要是一些单目标的数据集，也搜集下来）
2. https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc03647f
3. https://pmc.ncbi.nlm.nih.gov/articles/PMC8568316
4. https://pmc.ncbi.nlm.nih.gov/articles/PMC11352728
5. https://www.nature.com/articles/s41467-023-42446-5
6. https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/6807493c50018ac7c596ecf5/original/generality-driven-optimization-of-enantio-and-regioselective-catalysis-by-high-throughput-experimentation-and-machine-learning.pdf

## HTE dataset

### Suzuki dataset

### Buchwald-Hartwig dataset

### 

## Experimental dataset/example