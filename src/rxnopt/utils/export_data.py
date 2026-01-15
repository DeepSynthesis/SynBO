import numpy as np
from pathlib import Path

import pandas as pd
from rxnopt.utils.logger import console
from datetime import datetime
from typing import Dict, List, Literal, Optional, Union


def save_df(
    save_path: Union[str, Path],
    filetype: Literal["csv", "excel", "json"] = "csv",
    selected_conditions: np.ndarray = None,
    condition_dict: Dict[str, pd.DataFrame] = None,
    recommend_type: List[str] = None,
    batch_id: int = 0,
    pred_mean: Optional[np.ndarray] = None,
    pred_std: Optional[np.ndarray] = None,
    opt_metrics: List[str] = None,
    figure_output: List[str] = None,
    figure_path: Optional[Union[str, Path]] = None,
    suffix: Optional[str] = None,
    transpose: Optional[bool] = False,
) -> None:
    """Save reaction optimization results to file.

    Args:
        save_path: Path to save results (directory or full file path)
        filetype: Output file format ('csv', 'excel', or 'json')
        selected_conditions: Array of selected condition combinations
        condition_types: List of condition type names
        recommend_type: List of recommendation types ('explore' or 'exploit')
        batch_id: Batch identifier
        pred_mean: Prediction mean values (n_samples x n_metrics)
        pred_std: Prediction standard deviation values (n_samples x n_metrics)
        opt_metrics: List of optimization metric names
        figure_output: List of figure types to generate
        figure_path: Path for figures
        suffix: Optional filename suffix
        transpose: Whether to transpose Excel output
        console_obj: Console object for logging (optional)
    """
    if figure_output is None:
        figure_output = []
    save_path = Path(save_path)

    file_name = f"batch-{batch_id}_{datetime.now().strftime('%Y%m%d')}"
    if suffix:
        file_name = f"{file_name}_{suffix}"

    if save_path.is_dir() or not save_path.suffix:
        full_save_path = save_path / file_name
    else:
        full_save_path = save_path

    if not full_save_path.parent.exists():

        console.print(f"Creating directory: {full_save_path.parent}", style="yellow")
        full_save_path.parent.mkdir(parents=True, exist_ok=True)

    pred_data = {}
    if pred_mean is not None and pred_std is not None:
        for i, metric in enumerate(opt_metrics):
            pred_data[f"pred {metric}"] = [f"{mean:.2f}±{sigma:.2f}" for mean, sigma in zip(pred_mean[:, i], pred_std[:, i])]
    else:
        for metric in opt_metrics:
            pred_data[f"pred {metric}"] = ["-"] * len(selected_conditions)

    output_df = pd.DataFrame(
        {
            "batch": [batch_id] * len(selected_conditions),
            "index": range(1, len(selected_conditions) + 1),
            "type": recommend_type,
            **pd.DataFrame(selected_conditions, columns=condition_dict.keys()).to_dict("list"),
            **pred_data,
            **{metric: "[exp_data]" for metric in opt_metrics},
        }
    )

    if filetype == "csv":
        output_df.to_csv(full_save_path.with_suffix(".csv"), index=False)
    elif filetype == "excel" or filetype == "xlsx":
        from rxnopt.utils.write_excel import ExcelWriter

        writer = ExcelWriter(condition_dict=condition_dict, opt_metrics=opt_metrics)
        writer.write_to_excel(
            output_df=output_df,
            batch_id=batch_id,
            figure_output=figure_output,
            figure_path=figure_path,
            save_path=full_save_path,
            transpose=transpose,
        )
    elif filetype == "json":
        output_df.to_json(full_save_path.with_suffix(".json"), index=False, orient="records")
    else:
        raise ValueError(f"Unknown filetype: {filetype}")

    file_suffix = filetype if filetype != "excel" else "xlsx"
    console.print(f"✓ Saved recommendations to: [cyan]{full_save_path.with_suffix('.' + file_suffix)}[/cyan]", style="green")


def resave_output_results(
    input_file: str,
    output_file: str,
    condition_columns: List[str],
    metrics_columns: List[str],
    condition_dict: Dict[str, pd.DataFrame] = None,
    figure_output: List[str] = None,
    figure_path: Optional[str] = None,
    transpose: Optional[bool] = False,
) -> None:
    """Resave output results from one format to another.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        condition_columns: List of column names that represent conditions
        metrics_columns: List of column names that represent metrics
        figure_output: List of figure types to generate (optional)
        figure_path: Path for figures (optional)
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    assert input_file.exists(), f"Input file {input_file} does not exist."

    condition_dict = condition_dict if condition_dict is not None else [{k: None} for k in condition_columns]

    # Determine input file format and load data
    input_suffix = input_file.suffix.lower()
    if input_suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(input_file)
    elif input_suffix == ".json":
        df = pd.read_json(input_file)
    else:
        raise ValueError(f"Unsupported input file format: {input_suffix}")

    # Determine output file format
    output_suffix = output_file.suffix.lower()
    if output_suffix == ".csv":
        filetype = "csv"
    elif output_suffix in [".xlsx", ".xls"]:
        filetype = "excel"
    elif output_suffix == ".json":
        filetype = "json"
    else:
        raise ValueError(f"Unsupported output file format: {output_suffix}")

    # opt_metrics = metrics_columns

    # Extract batch ID and recommendation type
    batch_id = df["batch"].iloc[0] if "batch" in df.columns else 0
    recommend_type = df["type"].tolist() if "type" in df.columns else ["explore"] * len(df)

    # Extract selected conditions
    selected_conditions = df[condition_columns].values

    # Extract prediction data if available
    pred_mean = None
    pred_std = None

    # Look for pred columns matching the metrics
    pred_columns = [col for col in df.columns if col.startswith("pred ")]

    if pred_columns:
        # Parse pred columns (format: "mean±std")
        pred_data = {}
        for col in pred_columns:
            metric_name = col.replace("pred ", "")
            if metric_name in opt_metrics:
                pred_values = []
                for val in df[col]:
                    if val != "-" and pd.notna(val):
                        try:
                            # Parse "mean±std" format
                            parts = str(val).split("±")
                            mean = float(parts[0])
                            std = float(parts[1]) if len(parts) > 1 else 0.0
                            pred_values.append((mean, std))
                        except:
                            pred_values.append((0.0, 0.0))
                    else:
                        pred_values.append((0.0, 0.0))
                pred_data[metric_name] = pred_values

        # Convert to arrays
        if pred_data:
            n_metrics = len(opt_metrics)
            n_samples = len(df)
            pred_mean = np.zeros((n_samples, n_metrics))
            pred_std = np.zeros((n_samples, n_metrics))

            for i, metric_name in enumerate(opt_metrics):
                if metric_name in pred_data:
                    for j, (mean, std) in enumerate(pred_data[metric_name]):
                        pred_mean[j, i] = mean
                        pred_std[j, i] = std

    # Use the extracted save_reaction_results function
    save_df(
        save_path=output_file,
        filetype=filetype,
        selected_conditions=selected_conditions,
        condition_dict=condition_dict,
        recommend_type=recommend_type,
        batch_id=batch_id,
        pred_mean=pred_mean,
        pred_std=pred_std,
        opt_metrics=opt_metrics,
        figure_output=figure_output,
        figure_path=figure_path,
        transpose=transpose,
    )
