from loguru import logger
import torch
from torch import Tensor
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from tqdm import tqdm


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 64,
    unique: bool = True,
    maximum_metrics: bool = True,
    # X_avoid: Tensor | None = None,
    # inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None,
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
    len_choices = len(choices)
    if len_choices < q and unique:
        logger.warning(
            (f"Requested {q=} candidates from fully discrete search " f"space, but only {len_choices} possible choices remain. "),
        )
        q = len_choices
    choices_batched = choices.unsqueeze(-2)
    if q > 1:
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending

        for q_i in range(q):
            logger.info(f"Choosing candidate {q_i} of {q}...")
            with torch.no_grad():
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
            if hasattr(acq_function, "X_pending"):
                del acq_function.X_pending  # 释放旧张量q
                torch.cuda.empty_cache()  # 清空缓存（可选）
            acq_function.set_X_pending(torch.cat([base_X_pending, candidates], dim=-2) if base_X_pending is not None else candidates)
            # need to remove choice from choice set if enforcing uniqueness
            if unique:
                # choices_batched = torch.cat([choices_batched[:best_idx], choices_batched[best_idx + 1 :]])
                mask = torch.ones(len(choices_batched), dtype=bool)
                mask[best_idx] = False
                choices_batched = choices_batched[mask]  # 更高效的内存操作

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    with torch.no_grad():
        acq_values = _split_batch_eval_acqf(
            acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size, maximum_metrics=maximum_metrics
        )
    best_idx = torch.argmax(acq_values)
    return choices_batched[best_idx], acq_values[best_idx]


def _split_batch_eval_acqf(acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int, maximum_metrics: bool) -> Tensor:

    acq_values_list = []
    for X_batches in tqdm(X.split(max_batch_size)):
        with torch.no_grad():  # 确保无梯度计算
            acq_values_list.append(acq_function(X_batches))

        acq_values = torch.cat(acq_values_list)

    if maximum_metrics:
        return acq_values
    else:
        return -acq_values

    # if torch.cuda.device_count() > 1:
    #     # 将X均匀分配到各GPU
    #     X_split = X.chunk(torch.cuda.device_count(), dim=0)
    #     acq_values = []
    #     for i, x in enumerate(X_split):
    #         with torch.cuda.device(i):
    #             X_batches = x.split(max_batch_size)  # 提前拆分
    #             acq_values = torch.cat([acq_function(X_batch) for X_batch in tqdm(X_batches)])
    # else:
    #     acq_values = acq_function(X)

    # from concurrent.futures import ThreadPoolExecutor

    # def eval_acq_on_gpu(gpu_id, x):
    #     # 确保输入数据在正确的GPU上
    #     x = x.to(f"cuda:{gpu_id}")
    #     with torch.cuda.device(gpu_id):
    #         return acq_function(x).to("cpu")
    #         # return torch.cat([acq_function(batch) for batch in x.split(max_batch_size)])

    # with ThreadPoolExecutor(max_workers=torch.cuda.device_count()) as executor:
    #     for X_batch in tqdm(X.split(4096)):
    #         results = list(
    #             executor.map(eval_acq_on_gpu, range(1, torch.cuda.device_count()), X_batch.chunk(torch.cuda.device_count(), dim=0))
    #         )
    # acq_values = torch.cat(results)
