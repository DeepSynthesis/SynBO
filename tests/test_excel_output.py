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
        # 优化目标设置
        cls.opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

        # [修改点 1]：将保存路径指向 testfile 目录，使其与输入文件同目录，方便查看
        cls.save_dir = str(Path(__file__).parent / "testfile")

        # 预加载数据
        cls.desc_dict, cls.condition_dict = load_desc_dict(
            reagent_types=cls.reagent_types,
            desc_dir=Path(__file__).parent / "dataset/descriptors",
            name_suffix=["_dft", "_dft", "_dft", None, None],
            index_col=cls.reagent_types,
            return_condition_dict=True,
        )

    def tearDown(self):
        # [修改点 2]：注释掉删除文件的逻辑，以便保留 Excel 文件检查格式
        pass
        # if Path(self.save_dir).exists():
        #     for f in Path(self.save_dir).glob("*"):
        #         if f.is_file():
        #             f.unlink()

    def _run_optimization_workflow(self, opt_method, normalize, refine, filetype):
        print(f"\nRunning optimization with: {opt_method}, Output: {filetype}")

        rxn_opt = ReactionOptimizer(
            opt_metrics=["yield", "cost"],
            opt_metric_settings=self.opt_direct_info,
            opt_type="auto",
            quiet=False,  # [可选] 打开输出以便观察进度
        )
        rxn_opt.load_rxn_space(condition_dict=self.condition_dict)
        rxn_opt.load_desc(desc_dict=self.desc_dict)

        # 读取初始数据
        start_file_path = Path(__file__).parent / "testfile/start_file.csv"
        rxn_opt.load_prev_rxn(pd.read_csv(start_file_path, index_col=False))

        # 执行优化
        rxn_opt.optimize(batch_size=3, desc_normalize=normalize, refine_desc=refine, optimize_method=opt_method)

        # 保存结果
        # 注意：save_results 通常会自动拼接文件名，这里指定目录
        saved_files = rxn_opt.save_results(save_dir=self.save_dir, filetype=filetype)
        print(f"File saved to: {self.save_dir}")

    def test_single_excel_output(self):
        # [修改点 3]：创建一个单一的、针对 Excel 输出的测试用例
        # 使用 default_BO (贝叶斯优化) 作为典型方法，指定输出为 xlsx

        method = "random_select"
        norm = "minmax"
        refine = "none"
        ftype = "xlsx"
        # opt_kwargs = {"surrogate_model": "RF", "acq_func": "EHVI"}

        try:
            self._run_optimization_workflow(method, norm, refine, ftype)
            print("✅ Test execution finished successfully. Please check the 'testfile' folder.")
        except Exception as e:
            self.fail(f"❌ Test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main()
