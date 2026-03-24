import unittest
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from synbo import ReactionOptimizer
from synbo.utils.load_data import load_desc_dict


class TestReactionOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
        cls.save_dir = "test_results"

        # 预加载数据
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir=Path(__file__).parent / "dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def tearDown(self):
        if Path(self.save_dir).exists:
            for f in Path(self.save_dir).glob("*"):
                if f.is_file():
                    f.unlink()

    def _run_optimization_workflow(self, opt_method, normalize, refine, filetype, **opt_kwargs):
        # core code
        rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True)
        rxn_opt.load_rxn_space(condition_dict=self.condition_dict)
        rxn_opt.load_desc(desc_dict=self.desc_dict)
        rxn_opt.load_prev_rxn(pd.read_csv(Path(__file__).parent / "testfile/start_file.csv", index_col=False))

        rxn_opt.optimize(
            batch_size=2, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method, temperature=0.1, **opt_kwargs
        )
        rxn_opt.save_results(save_dir=self.save_dir, filetype=filetype)

    def test_combinations(self):
        test_params = [
            ("default_BO", "minmax", "auto_select", "csv", {"surrogate_model": "RF"}),
            # ("evolution", "minmax", "auto_select", "csv", {"surrogate_model": "RF"}),
            # ("particle_swarm", "minmax", "auto_select", "csv", {"surrogate_model": "RF"}),
            # ("evolution", "minmax", "auto_select", "csv", {"method": "Thompson", "surrogate_model": "RF"}),
            # ("evolution", "minmax", "auto_select", "csv", {"method": "Thompson", "surrogate_model": "ensemble"}),
            # ("evolution", "minmax", "auto_select", "csv", {"method": "Standard", "surrogate_model": "GP"}),
            # ("evolution", "minmax", "auto_select", "csv", {"method": "Thompson", "surrogate_model": "linear"}),
            # ("evolution", "minmax", "auto_select", "csv", {"method": "Standard"}),
            # ("default_BO", "minmax", "auto_select", "csv", {"acq_func": "NEI"}),
            # ("default_BO", "minmax", "auto_select", "csv", {"acq_func": "EHVI"}),
        ]

        passed, failed = 0, 0

        pbar = tqdm(test_params, desc="Optimization Tests", unit="task")

        for method, norm, refine, ftype, opt_kwargs in pbar:
            pbar.set_description(f"Testing: {method}")
            try:
                with self.subTest(method=method, norm=norm, refine=refine, ftype=ftype):
                    self._run_optimization_workflow(method, norm, refine, ftype, **opt_kwargs)
                    passed += 1
            except Exception as e:
                failed += 1
                raise e
            finally:
                pbar.set_postfix({"Passed": passed, "Failed": failed})


if __name__ == "__main__":
    unittest.main()
