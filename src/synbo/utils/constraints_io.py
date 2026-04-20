"""Utility functions for loading and saving prohibited reagents constraints."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from synbo.utils.logger import console


def load_prohibited_reagents(save_dir: Path) -> Optional[Dict[str, List[Any]]]:
    """Load prohibited reagents from prohibited_reagent.json.

    Args:
        save_dir: Directory path where prohibited_reagent.json is located

    Returns:
        Dictionary of prohibited reagents {condition_type: [prohibited_values]}
        Returns None if file doesn't exist or is empty
    """
    prohibited_file = Path(save_dir) / "prohibited_reagent.json"

    if not prohibited_file.exists():
        return None

    try:
        with open(prohibited_file, "r") as f:
            prohibited_reagents = json.load(f)

        if prohibited_reagents:
            console.print(f"✓ Loaded prohibited reagents from: {prohibited_file}", style="green")
            for condition_type, prohibited_values in prohibited_reagents.items():
                if prohibited_values:
                    console.print(f"  - {condition_type}: {len(prohibited_values)} prohibited", style="cyan")

        return prohibited_reagents if prohibited_reagents else None

    except Exception as e:
        console.print(f"⚠️ Error loading prohibited reagents: {e}", style="yellow")
        return None


def save_prohibited_reagents(
    save_dir: Path, new_prohibited: Dict[str, List[Any]], existing_prohibited: Optional[Dict[str, List[Any]]] = None
) -> None:
    """Save prohibited reagents to prohibited_reagent.json.

    This function merges new prohibited reagents with existing ones and saves to file.

    Args:
        save_dir: Directory path to save prohibited_reagent.json
        new_prohibited: New prohibited reagents from LLM recommendation
        existing_prohibited: Existing prohibited reagents (optional)
    """
    # Create directory if it doesn't exist
    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # Merge with existing prohibited reagents
    merged_prohibited = {}

    # Start with existing prohibited reagents
    if existing_prohibited:
        for condition_type, prohibited_values in existing_prohibited.items():
            merged_prohibited[condition_type] = list(set(prohibited_values))

    # Add new prohibited reagents
    for condition_type, prohibited_values in new_prohibited.items():
        if condition_type in merged_prohibited:
            # Merge with existing, avoiding duplicates
            merged_prohibited[condition_type] = list(set(merged_prohibited[condition_type] + prohibited_values))
        else:
            merged_prohibited[condition_type] = list(set(prohibited_values))

    # Save to file
    prohibited_file = save_dir / "prohibited_reagent.json"

    try:
        with open(prohibited_file, "w") as f:
            json.dump(merged_prohibited, f, indent=2, default=str)

        console.print(f"✓ Saved prohibited reagents to: {prohibited_file}", style="green")
        total_prohibited = sum(len(v) for v in merged_prohibited.values())
        console.print(f"  Total prohibited reagents: {total_prohibited}", style="cyan")

    except Exception as e:
        console.print(f"🚨 Error saving prohibited reagents: {e}", style="red")
        raise


def merge_constraints(
    constraints1: Optional[Dict[str, List[Any]]], constraints2: Optional[Dict[str, List[Any]]]
) -> Optional[Dict[str, List[Any]]]:
    """Merge two constraint dictionaries.

    Args:
        constraints1: First constraint dictionary
        constraints2: Second constraint dictionary

    Returns:
        Merged constraint dictionary
        Returns None if both inputs are None or empty
    """
    if not constraints1 and not constraints2:
        return None

    merged = {}

    if constraints1:
        for condition_type, prohibited_values in constraints1.items():
            merged[condition_type] = list(set(prohibited_values))

    if constraints2:
        for condition_type, prohibited_values in constraints2.items():
            if condition_type in merged:
                merged[condition_type] = list(set(merged[condition_type] + prohibited_values))
            else:
                merged[condition_type] = list(set(prohibited_values))

    return merged if merged else None
