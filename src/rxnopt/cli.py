"""Modern CLI interface for rxnopt using typer and rich."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from .rxnopt import ReactionOptimizer

app = typer.Typer(
    name="rxnopt",
    help="🧪 Modern Reaction Optimization Framework",
    rich_markup_mode="rich",
)
console = Console()


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print("[bold blue]rxnopt[/bold blue] version [green]0.1.0[/green]")
        console.print("A modern reaction optimization framework using Bayesian Optimization")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option("--version", "-v", callback=version_callback, is_eager=True),
    ] = None,
):
    """🧪 Modern Reaction Optimization Framework."""
    pass


@app.command()
def init(
    config_file: Annotated[
        Path,
        typer.Argument(help="Configuration file path (JSON format)"),
    ],
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Number of initial samples"),
    ] = 5,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Sampling method"),
    ] = "sobol",
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory"),
    ] = None,
) -> None:
    """Initialize reaction optimization with initial sampling."""

    if not config_file.exists():
        console.print(f"Error: Configuration file {config_file} not found", style="red")
        raise typer.Exit(1)

    # Load configuration
    with open(config_file) as f:
        config = json.load(f)

    console.print(
        Panel(
            f"[bold blue]Initializing Reaction Optimization[/bold blue]\n"
            f"Config: {config_file}\n"
            f"Batch size: {batch_size}\n"
            f"Method: {method}",
            title="🚀 Initialization",
        )
    )

    # Create optimizer
    optimizer = ReactionOptimizer(opt_metrics=config["opt_metrics"], opt_type="init")

    # Load reaction space and descriptors
    optimizer.load_rxn_space(config["condition_dict"])
    optimizer.load_desc(config.get("desc_dict"))

    # Run initialization
    optimizer.run(batch_size=batch_size)

    # Save results
    output_path = output_dir or Path.cwd()
    optimizer.save_results(save_dir=output_path, filetype="csv")

    console.print("[green]✓ Initialization completed successfully![/green]")


@app.command()
def optimize(
    config_file: Annotated[
        Path,
        typer.Argument(help="Configuration file path (JSON format)"),
    ],
    prev_data: Annotated[
        Path,
        typer.Argument(help="Previous reaction data (CSV format)"),
    ],
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Number of optimization suggestions"),
    ] = 5,
    method: Annotated[
        str,
        typer.Option("--method", "-m", help="Optimization method"),
    ] = "default",
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory"),
    ] = None,
    gpu_id: Annotated[
        int,
        typer.Option("--gpu", help="GPU device ID"),
    ] = 0,
) -> None:
    """Optimize reaction conditions using Bayesian Optimization."""

    if not config_file.exists():
        console.print(f"Error: Configuration file {config_file} not found", style="red")
        raise typer.Exit(1)

    if not prev_data.exists():
        console.print(f"Error: Previous data file {prev_data} not found", style="red")
        raise typer.Exit(1)

    # Load configuration and data
    with open(config_file) as f:
        config = json.load(f)

    prev_rxn_info = pd.read_csv(prev_data)

    console.print(
        Panel(
            f"[bold blue]Optimizing Reaction Conditions[/bold blue]\n"
            f"Config: {config_file}\n"
            f"Previous data: {prev_data} ({len(prev_rxn_info)} reactions)\n"
            f"Batch size: {batch_size}\n"
            f"Method: {method}\n"
            f"GPU: {gpu_id}",
            title="🎯 Optimization",
        )
    )

    # Create optimizer
    optimizer = ReactionOptimizer(opt_metrics=config["opt_metrics"], opt_type="opt")

    # Load reaction space, descriptors, and previous data
    optimizer.load_rxn_space(config["condition_dict"])
    optimizer.load_desc(config.get("desc_dict"))
    optimizer.load_prev_rxn(prev_rxn_info)

    # Run optimization
    optimizer.run(batch_size=batch_size)

    # Save results
    output_path = output_dir or Path.cwd()
    optimizer.save_results(save_dir=output_path, filetype="csv")

    console.print("[green]✓ Optimization completed successfully![/green]")


@app.command()
def validate(
    config_file: Annotated[
        Path,
        typer.Argument(help="Configuration file path (JSON format)"),
    ],
) -> None:
    """Validate configuration file."""

    if not config_file.exists():
        console.print(f"[red]Error: Configuration file {config_file} not found[/red]")
        raise typer.Exit(1)

    try:
        with open(config_file) as f:
            config = json.load(f)

        console.print("[green]✓ Configuration file is valid JSON[/green]")

        # Validate required fields
        required_fields = ["opt_metrics", "condition_dict"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            console.print(f"[red]Missing required fields: {missing_fields}[/red]")
            raise typer.Exit(1)

        # Display configuration summary
        table = Table(title="Configuration Summary")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Optimization Metrics", str(config["opt_metrics"]))
        table.add_row("Condition Types", str(list(config["condition_dict"].keys())))

        if "desc_dict" in config:
            table.add_row("Descriptor Types", str(list(config["desc_dict"].keys())))
        else:
            table.add_row("Descriptors", "[yellow]OneHot (auto-generated)[/yellow]")

        console.print(table)
        console.print("[green]✓ Configuration is valid![/green]")

    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON format - {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_config(
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output configuration file path"),
    ] = Path("rxnopt_config.json"),
) -> None:
    """Create a sample configuration file."""

    sample_config = {
        "opt_metrics": ["yield", "selectivity"],
        "condition_dict": {
            "catalyst": ["Pd(PPh3)4", "Pd(OAc)2", "PdCl2"],
            "solvent": ["THF", "DMF", "toluene", "water"],
            "temperature": [25, 50, 75, 100, 125],
            "base": ["K2CO3", "NaOH", "Et3N", "none"],
        },
        "desc_dict": {
            "catalyst": "Optional: path to catalyst descriptor file or dict",
            "solvent": "Optional: path to solvent descriptor file or dict",
            "temperature": "Will use numerical values directly",
            "base": "Optional: path to base descriptor file or dict",
        },
    }

    with open(output_file, "w") as f:
        json.dump(sample_config, f, indent=2)

    console.print(
        Panel(
            f"[green]✓ Sample configuration created at {output_file}[/green]\n\n"
            f"[yellow]Next steps:[/yellow]\n"
            f"1. Edit the configuration file to match your reaction system\n"
            f"2. Prepare descriptor files if needed (optional)\n"
            f"3. Run: [cyan]rxnopt init {output_file}[/cyan]",
            title="🔧 Configuration Created",
        )
    )


if __name__ == "__main__":
    app()
