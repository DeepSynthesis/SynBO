Based on the following reaction optimization data, please identify which reagents/conditions should be eliminated from the reaction space.

Previous Reaction Data Summary:
- Total reactions: 5
- Condition types: base, ligand, solvent, concentration, temperature
- Optimization metrics: index, type, pred yield, pred cost, yield, cost

Performance statistics:
  - index: min=1.00, max=5.00, mean=3.00

Top 5 performing reactions:
  - index: 5.00, base=CsOAc, ligand=BrettPhos, solvent=DMAc
  - index: 4.00, base=CsOAc, ligand=CgMe-PPh, solvent=DMAc
  - index: 3.00, base=CsOAc, ligand=PPh2Me, solvent=DMAc
  - index: 2.00, base=CsOAc, ligand=PPhtBu2, solvent=DMAc
  - index: 1.00, base=CsOAc, ligand=PPh3, solvent=DMAc



Reaction Space Summary:
- Goal: Eliminate approximately 2000% of reagents per condition type
- Condition types and their options:

base:
  - Total options: 4
  - Target to keep: ~3
  - All options: CsOAc, CsOPiv, KOAc, KOPiv

ligand:
  - Total options: 12
  - Target to keep: ~9
  - All options: BrettPhos, COC1=CC=C(OC)C(P(C2=CC(C(F)(F)F)=CC(C(F)(F)F)=C2)C3=CC(C(F)(F)F)=CC(C(F)(F)F)=C3)=C1C4=C(C(C)C)C=C(C(C)C)C=C4C(C)C, CgMe-PPh, GorlosPhos HBF4, P(fur)3, PCy3 HBF4, PPh2Me, PPh3, PPhMe2, PPhtBu2 ... and 2 more

solvent:
  - Total options: 4
  - Target to keep: ~3
  - All options: BuCN, BuOAc, DMAc, p-Xylene

concentration:
  - Total options: 3
  - Target to keep: ~2
  - All options: 0.057, 0.1, 0.153

temperature:
  - Total options: 3
  - Target to keep: ~2
  - All options: 105, 120, 90



Your task:
1. For each condition type, analyze the data to identify reagents that are performing poorly or are unlikely to lead to optimal reactions
2. Select approximately 2000% of reagents to eliminate from each condition type
3. Focus on eliminating the least promising reagents based on:
   - Poor performance in the existing data
   - Chemical incompatibility
   - Unlikely combinations
4. Return your answer in the following JSON format:

{
    "base": ["base_value_1", "base_value_2"],
    "ligand": ["ligand_value_1", "ligand_value_2"],
    "solvent": ["solvent_value_1", "solvent_value_2"]
}

The keys should be the condition types, and the values should be lists of reagents to KEEP (not eliminate).