import unittest
from pathlib import Path
import pandas as pd

from rxnopt import ReactionOptimizer
from rxnopt.utils.load_data import load_desc_dict


class TestReactionOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
        cls.save_dir = "test_results"

        # 预加载数据
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir="dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def tearDown(self):
        if Path(self.save_dir).exists:
            for f in Path(self.save_dir).glob("*"):
                if f.is_file():
                    f.unlink()

    def _run_optimization_workflow(self, opt_method, normalize, refine, filetype):
        # core code
        rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto")
        rxn_opt.load_rxn_space(condition_dict=self.condition_dict)
        rxn_opt.load_desc(desc_dict=self.desc_dict)
        rxn_opt.load_prev_rxn(pd.read_csv("testfile/start_file.csv", index_col=False))

        rxn_opt.optimize(batch_size=2, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method)
        rxn_opt.save_results(save_dir=self.save_dir, filetype=filetype)

    def test_combinations(self):
        """通过子测试覆盖不同的参数组合"""
        test_params = [
            ("default_BO", "minmax", "auto_select", "csv"),
            ("default_BO", "standard", None, "xlsx"),
            ("random_search", "none", "auto_select", "json"),  # 假设支持random_search
        ]

        for method, norm, refine, ftype in test_params:
            with self.subTest(method=method, norm=norm, refine=refine, ftype=ftype):
                self._run_optimization_workflow(method, norm, refine, ftype)


if __name__ == "__main__":
    unittest.main()
