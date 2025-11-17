from pathlib import Path
from re import S
import numpy as np
from typing import Any, List, Literal
from collections.abc import Sequence
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn


class SPOCDescriptor:
    def __init__(self, smiles_list: Sequence[str], desc_type: str = "OneHot", desc_type_to_filename: bool = True) -> None:
        self.smiles_list = smiles_list
        if isinstance(self.smiles_list, pd.Series):
            self.smiles_list = self.smiles_list.tolist()
        elif not isinstance(self.smiles_list, list):
            raise TypeError("smiles_list must be a list or pandas Series.")
        self.desc_type = desc_type
        self.desc_type_to_filename = desc_type_to_filename
        self.console = Console()

    def one_hot_encoding(self) -> list[int | bool]:
        """One Hot Encoding for categorical variables.

        Categorical variables are converted into a list which contains True/False
        or 1/0 representing the existence or non-existence of a specific category.
        """
        self.desc_array = np.eye(len(self.smiles_list), dtype=int)

    def rdkit_descriptor(self):
        """
        Parameters
        ----------
        smi: str
            SMILES of molecules

        Returns:
        -------
        type: list
            If SMILES is valid, a list will be returned.
            If SMILES is not valid, a list containing zero or Flase will be returned.

        Source:
        -------
        RDKit: https://www.rdkit.org/

        """
        from rdkit import Chem
        from rdkit.ML.Descriptors import MoleculeDescriptors

        try:
            mol_list = []
            for smi in self.smiles_list:
                mol = Chem.MolFromSmiles(smi)
                assert mol is not None
                mol_list.append(mol)

        except Exception:
            self.console.print(f"🚨 Cannot convert {smi}.", style="bold red")

        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Chem.Descriptors._descList])
        desc_array = calc.CalcDescriptors(mol)

    def rdkit_fingerprint(smi, fp_type="rdkit", radius=2, max_path=2, fp_length=1024, output="bit"):
        """Molecular fingerprint generation by rdkit package.

        Parameters:
        ------------
        smi: str
            SMILES string.
        fp_type: str
            • Avalon -- Avalon Fingerprint
            • AtomPaires -- Atom-Pairs Fingerprint
            • TopologicalTorsions -- Topological-Torsions Fingerprint
            • MACCSKeys Fingerprint 167
            • RDKit -- RDKit Fingerprint
            • RDKitLinear -- RDKit linear Fingerprint
            • LayeredFingerprint -- RDKit layered Fingerprint
            • Morgan -- Morgan-Circular Fingerprint
        radius: int
        max_path: int
        fp_length: int
        output: str
            "bit" -- the index of fp exist
            "vect" -- represeant by 0,1
            "bool" -- represeant by 1,-1

        Returns:
        -------
        type: list
            If SMILES is valid, a list will be returned.
            If SMILES is not valid, a list containing zero or Flase will be returned.

        Source:
        -------
        RDKit: https://www.rdkit.org/
        """

        mol = Chem.MolFromSmiles(smi)

        if mol:
            if fp_type == "RDKit":
                fp = Chem.RDKFingerprint(mol=mol, maxPath=max_path, fpSize=fp_length)

            elif fp_type == "RDKitLinear":
                fp = Chem.RDKFingerprint(mol=mol, maxPath=max_path, branchedPaths=False, fpSize=fp_length)

            elif fp_type == "AtomPaires":
                generator = Chem.rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_length)
                fp = generator.GetFingerprintAsNumPy(mol)

            elif fp_type == "TopologicalTorsions":
                tt_generator = Chem.rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_length)
                fp = tt_generator.GetFingerprintAsNumPy(mol)

            elif fp_type == "MACCSKeys":
                fp = MACCSkeys.GenMACCSKeys(mol)

            elif fp_type == "Morgan":
                mg_generator = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_length)
                fp = mg_generator.GetFingerprintAsNumPy(mol)

            elif fp_type == "Avalon":
                fp = pyAvalonTools.GetAvalonFP(mol, nBits=fp_length)

            elif fp_type == "LayeredFingerprint":
                fp = Chem.LayeredFingerprint(mol, maxPath=max_path, fpSize=fp_length)

            elif fp_type == "Estate":
                fp = list(Fingerprinter.FingerprintMol(mol)[0])

            elif fp_type == "EstateIndices":
                fp = list(Fingerprinter.FingerprintMol(mol)[1])

            else:
                print("Invalid fingerprint type!")

            # fp = fp2string(fp, output, fp_type)
            fp = fp2string(fp, output, fp_type)

        else:
            if fp_type == "MACCSKeys":
                fp_length = 167
            if fp_type == "Estate":
                fp_length = 79
            if fp_type == "EstateIndices":
                fp_length = 79
            fp = ExplicitBitVect(fp_length)
            fp = fp2string(fp, output="vect")

        return fp

    def obabel_fingerprint(smi, fp_type="FP2", nbit=1024, output="vect"):
        """Molecular fingerprint generation by OpebBabel.

        Parameters:
        -----------
        smi: str
            SMILES string
        fp_type: str
            • ECFP0/2/4/6/8 -- Extended-Connectivity Fingerprints (ECFPs)
            • FP2 -- linear fragments of length 1 to 7 (with some exceptions) using a hash code generating bits 0≤bit#<1021
            • FP3 -- SMARTS patterns based on 55 SMARTS patterns specified in the file patterns.txt
            • FP4 -- SMARTS patterns based on 307 SMARTS patterns specified in the file SMARTS_InteLigand.txt
            • MACCS -- SMARTS patterns specified in the file MACCS.txt
        nbit: int
            The length of obabel_fp
        output: str
            "bit" -- the index of fp exist
            "vect" -- represeant by 0,1
            "bool" -- represeant by 1,-1

        Returns:
        -------
        type: list
            If SMILES is valid, a list will be returned.
            If SMILES is not valid, a list containing zero or Flase will be returned.

        resource:
        ---------
        https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python_PybelAPI.html#pybel.fps
        http://openbabel.org/docs/dev/Features/Fingerprints.html
        http://openbabel.org/docs/dev/FileFormats/Fingerprint_format.html#fingerprint-format
        """

        try:
            import pybel

            mol = pybel.readstring("smi", smi)
            fp = mol.calcfp(fp_type)
            bits = list(fp.bits)
            bits = [x for x in bits if x < nbit]

            if output == "bit":
                fp = bits

            elif output == "vect":
                fp = np.zeros(nbit)
                fp[bits] = 1
                fp = fp.astype(int)

            elif output == "bool":
                fp = np.full(nbit, -1)
                fp[bits] = 1
                fp = fp.astype(int)

        except:
            fp = ExplicitBitVect(nbit)
            fp = fp2string(fp, output="vect")

        finally:
            return list(fp)

    def fp2string(fp, output, fp_type="Others"):

        if fp_type in ["Estate", "EstateIndices"]:
            fp = fp
        elif isinstance(fp, np.ndarray):  # Handle numpy arrays from certain fingerprint types
            if output == "bit":
                fp = list(np.where(fp > 0)[0])  # Get indices where values are non-zero
            elif output == "vect":
                fp = list(fp.astype(int))  # Convert to integer list
            elif output == "bool":
                fp = [1 if val > 0 else -1 for val in fp]
        elif output == "bit":
            fp = list(fp.GetOnBits())

        elif output == "vect":
            fp = list(fp.ToBitString())
            fp = [1 if val in ["1", 1] else 0 for val in fp]

        elif output == "bool":
            fp = list(fp.ToBitString())
            fp = [1 if val == "1" else -1 for val in fp]

        return fp

    def save_results(self, save_path: Path | str):
        desc_df = pd.DataFrame(self.desc_array, index=self.smiles_list)
        if desc_df.empty:
            self.console.print("⚠️ No data to save. DataFrame is empty.", style="bold yellow")
            raise Exception("No data to save. DataFrame is empty.")

        save_path = Path(save_path)
        assert save_path.parent.exists(), f"The directory {save_path.parent} does not exist."
        try:
            if save_path.suffix().lower() == "csv":
                desc_df.to_csv(save_path, index=True)
                self.console.print(f"✅ Results saved to CSV: {save_path}", style="bold green")

            elif save_path.suffix().lower() in ["xlsx"]:
                desc_df.to_excel(save_path, index=True)
                self.console.print(f"✅ Results saved to Excel: {save_path}", style="bold green")

            else:
                self.console.print(f"🚨 Unsupported format: {save_path.suffix()}. Supported formats: csv and xlsx", style="bold red")
                raise Exception(f"Unsupported format: {save_path.suffix()}")

        except Exception as e:
            self.console.print(f"🚨 Failed to save results: {e}", style="bold red")
            raise e


def calc_spoc_desc(
    smiles_list: List[str],
    save_path: Path | str,
    fp_type: str = "RDKit",
    size: int = 1024,
    radius: int = 2,
) -> None:
    """Generate molecular descriptors based on fingerprint type.

    Args:
        smiles: SMILES strings for molecules
        fp_type: Type of fingerprint/descriptor to generate
        size: Size of fingerprint vector
        radius: Radius for circular fingerprints

    """
    spoc_desc = SPOCDescriptor(smiles_list)
    match fp_type:
        case "OneHot":
            return spoc_desc.one_hot_calc(smiles_list)
        case fp if fp in ["ECFP", "ECFP0", "ECFP2", "ECFP4", "ECFP6", "ECFP8", "ECFP10", "FP2", "FP3", "FP4", "MACCS"]:
            return spoc_desc.ob_fp_calc(smiles_list, fp_type=fp, nbit=size)
        case fp if fp in [
            "Avalon",
            "AtomPaires",
            "TopologicalTorsions",
            "MACCSKeys",
            "RDKit",
            "RDKitLinear",
            "LayeredFingerprint",
            "Morgan",
            "FeaturedMorgan",
            "Estate",
            "EstateIndices",
        ]:
            return spoc_desc.rdkit_fp_calc(smiles_list, fp_type=fp, radius=radius, fp_length=size)

        case _:
            raise ValueError(f"Unsupported SPOC descriptor type: {fp_type}")

    spoc_desc.save_results(save_path)
