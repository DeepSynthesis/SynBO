from datetime import datetime
from itertools import product
from pathlib import Path
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

from .utils import track_called
from .initialize import Initializer
from .write_excel import ExcelWriter


def cartesian_product_3d(arr, data_type):
    cartesian_indices = np.array(list(product(*[range(len(middle)) for middle in arr])))
    # 计算结果矩阵的行数和列数
    num_rows = len(cartesian_indices)
    num_cols = sum(len(sub_arr[0]) for sub_arr in arr)
    # 初始化结果矩阵
    result = np.zeros((num_rows, len(arr)), dtype=data_type) if data_type == object else np.zeros((num_rows, num_cols), dtype=data_type)

    # 填充结果矩阵
    if data_type == object:
        for row_idx, indices in tqdm(enumerate(cartesian_indices), total=num_rows):
            for i, j in enumerate(indices):
                result[row_idx, i] = arr[i][j]
    else:
        for row_idx, indices in tqdm(enumerate(cartesian_indices), total=num_rows):
            col_idx = 0
            for i, j in enumerate(indices):
                inner_arr = arr[i][j]
                result[row_idx, col_idx : col_idx + len(inner_arr)] = inner_arr
                col_idx += len(inner_arr)

    return result


def normalize_data(total_desc_arr, desc_normalize):
    """
    Normalize the input array column-wise according to the specified method.

    Parameters:
    total_desc_arr (numpy.ndarray): Array to be normalized (2D array)
    desc_normalize (str): Normalization method, options:
        - 'minmax': Min-max scaling to [0,1] range
        - 'zscore': Standardization (zero mean, unit variance)
        - 'l2': L2 normalization (unit norm)
        - 'none': No normalization

    Returns:
    numpy.ndarray: Normalized array

    Raises:
    ValueError: If unknown normalization method is specified
    """
    try:
        match desc_normalize:
            case "minmax":
                return MinMaxScaler().fit_transform(total_desc_arr)
            case "zscore":
                return StandardScaler().fit_transform(total_desc_arr)
            case "l2":
                return Normalizer(norm="l2").fit_transform(total_desc_arr)
            case "none":
                return total_desc_arr.copy()
            case _:
                raise ValueError(f"Unknown normalization method: {desc_normalize}")
    except ValueError as e:
        # Handle cases where normalization might fail (e.g., constant columns)
        print(f"Normalization warning: {str(e)}")
        return np.zeros_like(total_desc_arr)


def array_process(desc_dict, condition_dict, condition_types, desc_normalize):
    desc_arrs = [[desc_dict[k].loc[name].values for name in condition_dict[k]] for k in condition_types]
    name_arrs = [names for names in condition_dict.values()]
    for tp, desc_arr in zip(condition_types, desc_arrs):
        logger.info(f"number of {tp}: {len(desc_arr)}")
    total_desc_arr = cartesian_product_3d(desc_arrs, data_type=float)
    total_name_arr = cartesian_product_3d(name_arrs, data_type=object)
    total_desc_arr = normalize_data(total_desc_arr, desc_normalize)
    return total_name_arr, total_desc_arr


class rxnOpter:
    def __init__(self, opt_metrics, opt_type="auto"):
        self.condition_dict = {}
        self.desc_dict = {}
        assert type(opt_metrics) == str or type(opt_metrics) == list, "opt_metrics must be str or list"
        self.opt_metrics = opt_metrics if type(opt_metrics) == list else [opt_metrics]
        self.opt_type = opt_type
        self.prev_rxn_info = None
        self.batch_id = 0
        assert opt_type in ["init", "opt", "auto"], "opt_type must be 'init', 'opt' or 'auto'"

    def load_rxn_space(self, condition_dict):
        condition_dict = {k: sorted(v) for k, v in sorted(condition_dict.items(), key=lambda x: x[0])}
        self.condition_types = condition_dict.keys()
        self.condition_dict = condition_dict

    def load_desc(self, desc_dict=None):
        if desc_dict == None:
            logger.warning("No desc path provided, using OneHot as alternative.")
            logger.warning("Need to finished.")
            # TODO: load OneHot
        else:
            assert desc_dict.keys() == self.condition_types, "Condition types do not match"
            self.desc_dict = desc_dict

    @track_called
    def load_prev_rxn(self, prev_rxn_info, metrics: list):
        self.opt_type == "opt" if self.opt_type == "auto" else self.opt_type
        self.batch_id = prev_rxn_info["batch_id"].max() + 1
        pass

    def run(self, batch_size=5, desc_normalize="minmax", expand_rxn_space=False):
        self.opt_type = "opt" if self.opt_type == "auto" and getattr(self, "_load_prev_rxn_called", False) else "init"
        if expand_rxn_space:
            pass

        if self.opt_type == "init":
            self.intialize(batch_size=batch_size, desc_normalize=desc_normalize)
        elif self.opt_type == "opt":
            self.optimize(batch_size=batch_size, desc_normalize=desc_normalize)
        else:
            raise ValueError("opt_type must be 'init' or 'opt'")
        # self.reagent_idx = list(product(*self.condition_dict.values()))
        # print(len(self.reagent_idx))

    def initialize(self, batch_size=5, desc_normalize="minmax", sampling_method="sobol"):
        logger.info("Now selecting initialize points...")
        self.total_name_arr, self.total_desc_arr = array_process(self.desc_dict, self.condition_dict, self.condition_types, desc_normalize)
        initializer = Initializer(numerical_data=self.total_desc_arr, name_data=self.total_name_arr)
        self.selected_conditions = initializer.sampling(method=sampling_method, batch_size=batch_size)
        # judgement selected points types: exploit or explore
        self.recommend_type = ["explore"] * batch_size

    def optimize(self, batch_size=5, desc_normalize="minmax", optimized_method="xxx"):
        logger.info("Now selecting optimize points...")

    def save_recommendations(self, save_task, filetype="csv", figure_output=None, figure_path=None):
        save_path = Path(save_task) / Path(f"batch-{self.batch_id}_{datetime.now().strftime('%Y%m%d')}")
        if save_path.parent.exists() == False:
            logger.warning("Parent directory does not exist, creating...")
            save_path.parent.mkdir(parents=True)
        output_df = pd.DataFrame(
            {
                "batch": [self.batch_id] * len(self.selected_conditions),
                "index": range(1, len(self.selected_conditions) + 1),
                "type": self.recommend_type,
                **pd.DataFrame(self.selected_conditions, columns=self.condition_types).to_dict("list"),
                **{metric: "" for metric in self.opt_metrics},
            }
        )

        if filetype == "csv":
            output_df.to_csv(save_path.with_suffix(".csv"), index=False)
        elif filetype == "excel":
            writer = ExcelWriter(condition_types=self.condition_types, opt_metrics=self.opt_metrics)
            writer.write_to_excel(
                output_df=output_df,
                batch_id=self.batch_id,
                figure_output=figure_output,
                figure_path=figure_path,
                save_path=save_path,
            )
        else:
            raise ValueError("Unknown filetype")
