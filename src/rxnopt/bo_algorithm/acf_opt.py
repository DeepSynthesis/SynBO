import gpytorch
import torch
from torch import Tensor
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from rich.console import Console


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 128,
    unique: bool = True,
    maximum_metrics: bool = True,
    progress: object = None,
    task: object = None,
) -> tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).
        X_avoid: An `n x d` tensor of candidates that we aren't allowed to pick.
            These will be removed from the set of choices.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
            Infeasible points will be removed from the set of choices.

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    acf_console = Console()
    len_choices = len(choices)
    if len_choices < q and unique:
        acf_console.print(
            f"Requested {q=} candidates from fully discrete search space, but only {len_choices} possible choices remain. ",
            style="yellow",
        )
        q = len_choices
    choices_batched = choices.unsqueeze(-2)

    if q > 1:
        candidate_list, acq_value_list = [], []
        for q_i in range(q):
            progress.log(f"Chooseing candidate {q_i+1} of {q}", style="yellow")
            progress.update(task, advance=1)
            with torch.no_grad():
                with gpytorch.settings.cholesky_jitter(1e-3):
                    acq_values = _split_batch_eval_acqf(
                        acq_function=acq_function,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                        maximum_metrics=maximum_metrics,
                    )
            best_idx = torch.argmax(acq_values)
            candidate_list.append(choices_batched[best_idx])
            acq_value_list.append(acq_values[best_idx])
            # set pending points
            candidates = torch.cat(candidate_list, dim=-2)
            torch.cuda.empty_cache()  # 清空缓存（可选）
            acq_function.set_X_pending(candidates)

        # Reset acq_func to previous X_pending state
        # TODO: Deal with unique... need to remove choice from choice set if enforcing uniqueness
        acq_function.set_X_pending(acq_function.X_pending)
        return candidates, torch.stack(acq_value_list)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size, maximum_metrics=maximum_metrics
        )
    best_idx = torch.argmax(acq_values)
    return choices_batched[best_idx], acq_values[best_idx]


def _split_batch_eval_acqf(acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int, maximum_metrics: bool) -> Tensor:

    acq_values_list = []
    for X_batches in X.split(max_batch_size):
        with torch.no_grad():
            acq_values = acq_function(X_batches)
            acq_values_list.append(acq_values)
        acq_values = torch.cat(acq_values_list)

    if maximum_metrics:
        return acq_values
    else:
        return -acq_values
