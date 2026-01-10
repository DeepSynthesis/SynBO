import unittest
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from rxnopt import ReactionOptimizer
from rxnopt.utils.load_data import load_desc_dict


class TestReactionOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

        # 设定保存目录为当前脚本下的 testfile 目录
        cls.save_dir = Path(__file__).parent / "testfile"

        # 确保目录存在
        if not cls.save_dir.exists():
            cls.save_dir.mkdir(parents=True)

        # 预加载数据
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir=Path(__file__).parent / "dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def _run_optimization_workflow(self, opt_method, normalize, refine, filetype, **opt_kwargs):
        rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=self.opt_direct_info, opt_type="auto", quiet=True)
        rxn_opt.load_rxn_space(condition_dict=self.condition_dict)
        rxn_opt.load_desc(desc_dict=self.desc_dict)

        start_file_path = self.save_dir / "start_file.csv"
        rxn_opt.load_prev_rxn(pd.read_csv(start_file_path, index_col=False))
        rxn_opt.optimize(batch_size=3, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method, **opt_kwargs)
        rxn_opt.save_results(save_dir=str(self.save_dir), filetype=filetype)

    def test_generate_and_rename_excel(self):
        # 参数设置
        method = "random_select"
        norm = "minmax"
        refine = "none"
        ftype = "xlsx"
        target_name = "output_excel.xlsx"  # 最终想要的文件名
        opt_kwargs = {"surrogate_model": "RF", "acq_func": "EHVI"}

        files_before = set(self.save_dir.glob("*.xlsx"))

        print(f"\n[Status] Starting optimization to generate {ftype}...")

        try:
            self._run_optimization_workflow(method, norm, refine, ftype, **opt_kwargs)
        except Exception as e:
            self.fail(f"Optimization failed: {e}")

        files_after = set(self.save_dir.glob("*.xlsx"))
        new_files = files_after - files_before

        if not new_files:
            self.fail("No new .xlsx file was detected after optimization!")

        generated_file = list(new_files)[0]
        target_path = self.save_dir / target_name

        print(f"[Status] Detected new file: {generated_file.name}")

        try:
            generated_file.replace(target_path)
            print(f"✅ Success! File renamed to: {target_path}")
            print(f"   You can now open '{target_name}' to check the format.")
        except Exception as e:
            self.fail(f"Failed to rename file: {e}")


if __name__ == "__main__":
    unittest.main()
