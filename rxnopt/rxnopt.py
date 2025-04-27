from datetime import datetime
from pathlib import Path
from loguru import logger
import pandas as pd

from rxnopt.bo_opt import Optimizer


from .utils.utils import check_desc_completeness, done_array_process, generate_onehot_desc, track_called, array_process
from .initialize import Initializer
from .utils.write_excel import ExcelWriter


class ReactionOptimizer:
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
            self.desc_dict = generate_onehot_desc(self.condition_dict)
        else:
            assert desc_dict.keys() == self.condition_types, "Condition types do not match"
            self.desc_dict = desc_dict

    @track_called
    def load_prev_rxn(self, prev_rxn_info, drop_rxn=False):
        self.opt_type == "opt" if self.opt_type == "auto" else self.opt_type
        self.batch_id = prev_rxn_info["batch"].max() + 1

        assert all(t in prev_rxn_info.columns for t in self.condition_types), "Condition types do not match"
        for t in self.condition_types:
            missing_species = set(prev_rxn_info[t]) - set(self.condition_dict[t])
            if missing_species:
                if drop_rxn:
                    logger.warning(f"{missing_species} not in {t} condition space, dropping these reactions")
                    prev_rxn_info = prev_rxn_info[~prev_rxn_info[t].isin(missing_species)]
                else:
                    logger.error(f"{missing_species} not in {t} condition space")
                    raise ValueError(f"{missing_species} not in {t} condition space")

        self.prev_rxn_info = prev_rxn_info

    def run(self, batch_size=5, desc_normalize="minmax", expand_rxn_space=False):
        self.opt_type = "opt" if self.opt_type == "auto" and getattr(self, "_load_prev_rxn_called", False) else "init"
        if expand_rxn_space:
            pass

        if self.opt_type == "init":
            self.initialize(batch_size=batch_size, desc_normalize=desc_normalize)
        elif self.opt_type == "opt":
            self.optimize(batch_size=batch_size, desc_normalize=desc_normalize)
        else:
            raise ValueError("opt_type must be 'init' or 'opt'")
        # self.reagent_idx = list(product(*self.condition_dict.values()))
        # print(len(self.reagent_idx))

    def initialize(self, batch_size=5, desc_normalize="minmax", sampling_method="sobol"):
        check_desc_completeness(self.desc_dict, self.condition_dict)
        logger.info("Now selecting initialize points...")
        self.total_name_arr, self.total_desc_arr = array_process(self.desc_dict, self.condition_dict, self.condition_types, desc_normalize)
        initializer = Initializer(numerical_data=self.total_desc_arr, name_data=self.total_name_arr)
        self.selected_conditions = initializer.sampling(method=sampling_method, batch_size=batch_size)
        # judgement selected points types: exploit or explore
        self.recommend_type = ["explore"] * batch_size

    def optimize(self, batch_size=5, desc_normalize="minmax", optimized_method="default", opt_weights=None):
        check_desc_completeness(self.desc_dict, self.condition_dict)
        logger.info("Now selecting optimize points...")
        self.total_name_arr, self.total_desc_arr = array_process(self.desc_dict, self.condition_dict, self.condition_types, desc_normalize)
        self.done_arr_index = done_array_process(self.prev_rxn_info, self.total_name_arr, self.condition_types)
        done_arr_desc = self.total_desc_arr[self.done_arr_index]
        done_arr_metrics = {k: self.prev_rxn_info[k].values for k in self.opt_metrics}
        optimizer = Optimizer(method=optimized_method)
        self.selected_conditions = optimizer.optimize(
            training_X=done_arr_desc,
            training_y=done_arr_metrics,
            candidate_X=self.total_desc_arr,
            batch_size=batch_size,
            opt_weights=opt_weights,
        )  # self.recommend_type

    def save_recommendations(self, save_task, filetype="csv", figure_output=[], figure_path=None):
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
