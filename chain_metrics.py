"""Shared scoring helpers comparing repeater-chain surrogate samples with SeQUeNCe simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import ks_2samp, wasserstein_distance

from train_surrogate_chain import (FIDELITY_PATH, GLOBAL_FEATURE_NAMES, LINK_DISTANCES_NAME, TIME_PATH,
                                   USER_OPTIONAL_PATH)

DEFAULT_SAMPLE_COUNT = 2000
DEFAULT_JOINT_METRIC_SAMPLE_COUNT = 2000
DEFAULT_SEED = 7
FIDELITY_LOGIT_EPSILON = 1e-6
# Metrics worth aggregating across a whole testing set. Lower is better for every one of them.
SWEEP_METRICS = ["normalized_time_wasserstein", "time_ks", "fidelity_wasserstein", "fidelity_ks",
                 "joint_energy", "absolute_correlation_difference"]


def read_condition(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Read one chain configuration and its paired SeQUeNCe samples.

    Args:
        path (Path): Chain simulation HDF5 file.

    Returns:
        tuple[dict[str, Any], np.ndarray]: Chain configuration and paired (time, fidelity) rows.
    """
    with h5py.File(path, "r") as h5_file:
        attributes = h5_file[USER_OPTIONAL_PATH].attrs
        config = {LINK_DISTANCES_NAME: np.asarray(attributes[LINK_DISTANCES_NAME], dtype=np.float32).tolist()}
        for name in GLOBAL_FEATURE_NAMES:
            config[name] = float(attributes[name])
        time_samples = np.asarray(h5_file[TIME_PATH][:], dtype=np.float32)
        fidelity_samples = np.asarray(h5_file[FIDELITY_PATH][:], dtype=np.float32)

    if time_samples.ndim != 1 or fidelity_samples.ndim != 1 or time_samples.shape != fidelity_samples.shape:
        raise ValueError(f"{path}: SeQUeNCe time and fidelity must be paired one-dimensional arrays")
    return config, np.column_stack((time_samples, fidelity_samples))


def joint_energy_distance(sequence: np.ndarray, surrogate: np.ndarray,
                          sample_limit: int = DEFAULT_JOINT_METRIC_SAMPLE_COUNT,
                          seed: int = DEFAULT_SEED) -> float:
    """Calculate standardized multivariate energy distance with bounded memory use.

    Args:
        sequence (np.ndarray): Paired SeQUeNCe samples.
        surrogate (np.ndarray): Paired surrogate samples.
        sample_limit (int): Maximum samples used from each source.
        seed (int): Random seed used to subsample.

    Returns:
        float: Energy distance after standardizing both dimensions on the SeQUeNCe scale.
    """
    rng = np.random.default_rng(seed)
    if len(sequence) > sample_limit:
        sequence = sequence[rng.choice(len(sequence), sample_limit, replace=False)]
    if len(surrogate) > sample_limit:
        surrogate = surrogate[rng.choice(len(surrogate), sample_limit, replace=False)]

    mean = np.mean(sequence, axis=0)
    std = np.std(sequence, axis=0)
    std[std < 1e-12] = 1.0
    sequence = (sequence - mean) / std
    surrogate = (surrogate - mean) / std
    energy_squared = 2 * np.mean(cdist(sequence, surrogate)) - np.mean(pdist(sequence)) - np.mean(pdist(surrogate))
    return float(np.sqrt(max(energy_squared, 0.0)))


def time_fidelity_correlation(samples: np.ndarray) -> float:
    """Return Pearson correlation between paired time and fidelity samples.

    Args:
        samples (np.ndarray): Paired (time, fidelity) rows.

    Returns:
        float: Correlation, or NaN when either dimension is constant.
    """
    if np.std(samples[:, 0]) == 0 or np.std(samples[:, 1]) == 0:
        return float("nan")
    return float(np.corrcoef(samples[:, 0], samples[:, 1])[0, 1])


def logit_fidelity_std_ratio(sequence: np.ndarray, surrogate: np.ndarray) -> float:
    """Return the surrogate-to-SeQUeNCe spread ratio of fidelity in logit space.

    Fidelity is trained in logit space, so this reports over-dispersion on the scale the model
    actually predicts. Values above one mean the surrogate is wider than the simulation.

    Args:
        sequence (np.ndarray): Paired SeQUeNCe samples.
        surrogate (np.ndarray): Paired surrogate samples.

    Returns:
        float: Ratio of surrogate to SeQUeNCe logit-fidelity standard deviation.
    """
    def logit(fidelity: np.ndarray) -> np.ndarray:
        clipped = np.clip(fidelity, FIDELITY_LOGIT_EPSILON, 1 - FIDELITY_LOGIT_EPSILON)
        return np.log(clipped / (1 - clipped))

    sequence_std = float(np.std(logit(sequence[:, 1])))
    if sequence_std < 1e-12:
        return float("nan")
    return float(np.std(logit(surrogate[:, 1]))) / sequence_std


def score_condition(sequence: np.ndarray, surrogate: np.ndarray,
                    joint_metric_sample_count: int = DEFAULT_JOINT_METRIC_SAMPLE_COUNT,
                    seed: int = DEFAULT_SEED) -> dict[str, float]:
    """Return every distribution-comparison metric for one paired condition.

    Args:
        sequence (np.ndarray): Paired SeQUeNCe samples.
        surrogate (np.ndarray): Paired surrogate samples.
        joint_metric_sample_count (int): Maximum samples used for the joint energy distance.
        seed (int): Random seed used by the joint energy distance.

    Returns:
        dict[str, float]: Named metrics for this condition.
    """
    time_wasserstein = float(wasserstein_distance(sequence[:, 0], surrogate[:, 0]))
    fidelity_wasserstein = float(wasserstein_distance(sequence[:, 1], surrogate[:, 1]))
    time_ks = ks_2samp(sequence[:, 0], surrogate[:, 0])
    fidelity_ks = ks_2samp(sequence[:, 1], surrogate[:, 1])
    sequence_correlation = time_fidelity_correlation(sequence)
    surrogate_correlation = time_fidelity_correlation(surrogate)
    return {
        "time_wasserstein_ms": time_wasserstein,
        "normalized_time_wasserstein": time_wasserstein / float(np.mean(sequence[:, 0])),
        "time_ks": float(time_ks.statistic),
        "time_ks_pvalue": float(time_ks.pvalue),
        "fidelity_wasserstein": fidelity_wasserstein,
        "fidelity_ks": float(fidelity_ks.statistic),
        "fidelity_ks_pvalue": float(fidelity_ks.pvalue),
        "joint_energy": joint_energy_distance(sequence, surrogate, joint_metric_sample_count, seed),
        "logit_fidelity_std_ratio": logit_fidelity_std_ratio(sequence, surrogate),
        "sequence_correlation": sequence_correlation,
        "surrogate_correlation": surrogate_correlation,
        "absolute_correlation_difference": abs(sequence_correlation - surrogate_correlation),
    }


def condition_record(path: Path, sampler: Callable[..., np.ndarray], test_index: int,
                     sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Score one condition file and return chain structure, global features, and metrics.

    Args:
        path (Path): Chain simulation HDF5 file.
        sampler (Callable[..., np.ndarray]): Sampler from build_condition_sampler.
        test_index (int): Position of this condition within the scored set.
        sample_count (int): Surrogate samples drawn for this condition.
        seed (int): Random seed used for sampling and the joint energy distance.

    Returns:
        dict[str, Any]: One flat row describing this condition.
    """
    config, sequence = read_condition(path)
    surrogate = sampler(config, sample_count, seed)
    link_distances = np.asarray(config[LINK_DISTANCES_NAME], dtype=np.float32)
    return {
        "condition": path.stem,
        "test_index": test_index,
        "nodes": int(link_distances.size + 1),
        "total_distance_km": float(link_distances.sum()) / 1000.0,
        "mean_link_distance_km": float(link_distances.mean()) / 1000.0,
        "max_link_distance_km": float(link_distances.max()) / 1000.0,
        **{name: config[name] for name in GLOBAL_FEATURE_NAMES},
        "sequence_sample_count": int(len(sequence)),
        "sequence_time_mean_ms": float(np.mean(sequence[:, 0])),
        "sequence_fidelity_mean": float(np.mean(sequence[:, 1])),
        "sequence_fidelity_std": float(np.std(sequence[:, 1])),
        **score_condition(sequence, surrogate, seed=seed),
    }


def score_conditions(paths: list[Path], sampler: Callable[..., np.ndarray],
                     sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Score every condition file with one already-built sampler.

    Args:
        paths (list[Path]): Chain simulation HDF5 files.
        sampler (Callable[..., np.ndarray]): Sampler from build_condition_sampler.
        sample_count (int): Surrogate samples drawn per condition.
        seed (int): Random seed used for sampling and the joint energy distance.

    Returns:
        list[dict[str, Any]]: One row per condition.
    """
    return [condition_record(path, sampler, index, sample_count, seed) for index, path in enumerate(paths)]
