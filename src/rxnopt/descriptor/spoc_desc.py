from pathlib import Path
import numpy as np
from typing import Any, List, Literal
from collections.abc import Sequence
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.EState import Fingerprinter
from rdkit.ML.Descriptors import MoleculeDescriptors


class SPOCDescriptor:
    def __init__(self, smiles_list: Sequence[str], desc_type: str = "OneHot", desc_type_to_filename: bool = False) -> None:
        self.smiles_list = smiles_list
        if isinstance(self.smiles_list, pd.Series):
            self.smiles_list = self.smiles_list.tolist()
        elif not isinstance(self.smiles_list, list):
            raise TypeError("smiles_list must be a list or pandas Series.")
        self.desc_type = desc_type
        self.desc_type_to_filename = desc_type_to_filename
        self.desc_names = None
        self.console = Console()

    def _smiles_to_mol(self) -> List[Any]:
        mol_list = []
        try:
            mol_list = []
            for smi in self.smiles_list:
                mol = Chem.MolFromSmiles(smi)
                assert mol is not None
                mol_list.append(mol)

        except Exception:
            self.console.print(f"🚨 Cannot convert SMILES: {smi}.", style="bold red")
            raise Exception(f"Cannot convert SMILES: {smi}.")
        return mol_list

    def one_hot_descriptor(self) -> list[int | bool]:
        """One Hot Encoding for categorical variables.

        Categorical variables are converted into a list which contains True/False
        or 1/0 representing the existence or non-existence of a specific category.
        """
        self.desc_array = np.eye(len(self.smiles_list), dtype=int)

    def rdkit_descriptor(self):
        mol_list = self._smiles_to_mol()

        self.desc_names = [desc_name for desc_name, _ in Chem.Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.desc_names)
        self.desc_array = np.array([calc.CalcDescriptors(mol) for mol in mol_list])
        self.desc_array = self.desc_array.round(5)

    def rdkit_fp_descriptor(self, radius: int = 2, fp_length: int = 1024):
        mol_list = self._smiles_to_mol()
        match self.desc_type:
            case "RDKitFP":
                self.desc_array = np.array([Chem.RDKFingerprint(mol=mol, maxPath=radius, fpSize=fp_length) for mol in mol_list])

            case "RDKitLinear":
                self.desc_array = np.array(
                    [Chem.RDKFingerprint(mol=mol, maxPath=radius, branchedPaths=False, fpSize=fp_length) for mol in mol_list]
                )

            case "AtomPaires":
                generator = Chem.rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_length)
                self.desc_array = generator.GetFingerprints(mol_list)
                self.desc_array = np.array([[x for x in fp] for fp in self.desc_array])

            case "TopologicalTorsions":
                tt_generator = Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_length)
                self.desc_array = tt_generator.GetFingerprints(mol_list)
                self.desc_array = np.array([[x for x in fp] for fp in self.desc_array])

            case "MACCSKeys":
                self.desc_array = np.array([MACCSkeys.GenMACCSKeys(mol) for mol in mol_list])

            case "Morgan":
                mg_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_length)
                self.desc_array = mg_generator.GetFingerprints(mol_list)
                self.desc_array = np.array([[x for x in fp] for fp in self.desc_array])

            case "Avalon":
                self.desc_array = np.array([pyAvalonTools.GetAvalonFP(mol, nBits=fp_length) for mol in mol_list])

            case "LayeredFingerprint":
                self.desc_array = np.array([Chem.LayeredFingerprint(mol, maxPath=radius, fpSize=fp_length) for mol in mol_list])

            case "Estate":
                self.desc_array = np.array([(Fingerprinter.FingerprintMol(mol)[0]) for mol in mol_list])

            case "EstateIndices":
                self.desc_array = np.array([Fingerprinter.FingerprintMol(mol)[1] for mol in mol_list])

            case _:
                raise ValueError(f"Unsupported RDKit fingerprint type: {self.desc_type}")

        self.desc_array = self.desc_array.round(5)

    def ob_fp_descriptor(self, fp_length=1024):
        # Not use now...
        # import pybel
        # mol_list = [pybel.readstring("smi", smi) for smi in self.smiles_list]
        # fp = [mol.calcfp(self.fp_type) for mol in mol_list]
        # bits = [x for x in fp.bits if x < fp_length]

        # self.desc_array = np.array([x for x in bits])
        pass

    def save_results(self, save_path: Path | str):
        desc_df = pd.DataFrame(self.desc_array, index=self.smiles_list)
        if self.desc_names is not None:
            desc_df.columns = self.desc_names
        if desc_df.empty:
            self.console.print("⚠️ No data to save. DataFrame is empty.", style="bold yellow")
            raise Exception("No data to save. DataFrame is empty.")

        save_path = Path(save_path)
        if self.desc_type_to_filename:
            save_path = save_path.parent / Path(str(save_path.stem) + "_" + self.desc_type + save_path.suffix)
        assert save_path.parent.exists(), f"The directory {save_path.parent} does not exist."
        try:
            if save_path.suffix.lower() == ".csv":
                desc_df.to_csv(save_path, index=True)
                self.console.print(f"✅ Results saved to CSV: {save_path}, data shape is {desc_df.shape}", style="bold green")

            elif save_path.suffix.lower() in [".xlsx"]:
                desc_df.to_excel(save_path, index=True)
                self.console.print(f"✅ Results saved to Excel: {save_path}, data shape is {desc_df.shape}", style="bold green")

            else:
                self.console.print(f"🚨 Unsupported format: {save_path.suffix}. Supported formats: csv and xlsx", style="bold red")
                raise Exception(f"Unsupported format: {save_path.suffix}")

        except Exception as e:
            self.console.print(f"🚨 Failed to save results: {e}", style="bold red")
            raise e


def calc_spoc_desc(
    smiles_list: List[str],
    save_path: Path | str,
    fp_type: str = "RDKit",
    size: int = 1024,
    radius: int = 2,
    desc_type_to_filename: bool = True,
) -> None:
    """Generate molecular descriptors based on fingerprint type.

    Args:
        smiles: SMILES strings for molecules
        fp_type: Type of fingerprint/descriptor to generate
        size: Size of fingerprint vector
        radius: Radius for circular fingerprints

    """
    spoc_desc = SPOCDescriptor(smiles_list=smiles_list, desc_type=fp_type, desc_type_to_filename=desc_type_to_filename)
    match fp_type:
        case "OneHot":
            spoc_desc.one_hot_descriptor()
        case "RDKit":
            spoc_desc.rdkit_descriptor()
        case fp if fp in ["ECFP", "ECFP0", "ECFP2", "ECFP4", "ECFP6", "ECFP8", "ECFP10", "FP2", "FP3", "FP4", "MACCS"]:
            spoc_desc.ob_fp_descriptor(fp_length=size)
        case fp if fp in [
            "Avalon",
            "AtomPaires",
            "TopologicalTorsions",
            "MACCSKeys",
            "RDKitFP",
            "RDKitLinear",
            "LayeredFingerprint",
            "Morgan",
            "FeaturedMorgan",
            "Estate",
            "EstateIndices",
        ]:
            spoc_desc.rdkit_fp_descriptor(radius=radius, fp_length=size)

        case _:
            raise ValueError(f"Unsupported SPOC descriptor type: {fp_type}")

    spoc_desc.save_results(save_path)
