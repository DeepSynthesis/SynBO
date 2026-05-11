import unittest
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch

from synbo import ReactionOptimizer
from synbo.algorithm.acq_function import BaseAcquisitionFunction
from synbo.algorithm.bo_core import DefaultBO
from synbo.utils.load_data import load_desc_dict


class FakeProgress:
    def __init__(self):
        self.advanced = 0

    def update(self, task, advance=0, **kwargs):
        self.advanced += advance


class FakeAcquisitionFunction:
    def __init__(self):
        self.pending = []

    def __call__(self, X):
        return X.squeeze(-2).sum(dim=-1)

    def set_X_pending(self, candidates):
        self.pending.append(candidates)


class TestReactionOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
        cls.save_dir = "test_results"

        # Pre-load data
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
        sbo = ReactionOptimizer(
            opt_metrics=["yield", "cost"],
            opt_metric_settings=self.opt_direct_info,
            opt_type="auto",
            quiet=False,
            save_dir=self.save_dir,
        )
        sbo.load_rxn_space(condition_dict=self.condition_dict)
        sbo.load_desc(desc_dict=self.desc_dict)
        sbo.load_prev_rxn(pd.read_csv(Path(__file__).parent / "testfile/start_file.csv", index_col=False))

        sbo.optimize(batch_size=2, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method, **opt_kwargs)
        sbo.save_results(filetype=filetype)

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

    def test_acquisition_progress_updates_per_evaluation_chunk(self):
        optimizer = BaseAcquisitionFunction(model=None, sampler=None, device=torch.device("cpu"))
        progress = FakeProgress()
        choices = torch.arange(10, dtype=torch.double).reshape(5, 2)

        optimizer.optimize_discrete(
            acq_func=FakeAcquisitionFunction(),
            q=5,
            choices=choices,
            unique=True,
            max_batch_size=2,
            progress=progress,
            task="acq",
        )

        self.assertEqual(progress.advanced, 9)

    def test_acquisition_progress_total_matches_evaluation_chunks(self):
        self.assertEqual(DefaultBO._acquisition_progress_total(num_choices=5, batch_size=5, max_batch_size=2), 9)

    def test_acquisition_progress_batch_size_smooths_small_candidate_spaces(self):
        self.assertEqual(
            DefaultBO._acquisition_progress_batch_size(
                num_choices=5,
                max_batch_size=2048,
                chunks_per_candidate=20,
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
