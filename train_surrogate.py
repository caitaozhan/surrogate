"""Train a conditional Mixture Density Network for two-node link service time."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.distributions as torch_distributions
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset


USER_OPTIONAL_PATH = "qnet/metadata/user_optional"
TIME_PATH = "qnet/simulation_log/time"
FEATURE_NAMES = ("link_distance", "memory_efficiency", "detector_efficiency", "attenuation", "classical_delay_node")
TARGET_NAME = "time"
OUTPUT_DIM = 1
TIME_EPSILON = np.float32(1e-12)
STD_EPSILON = np.float32(1e-12)
MdnArtifact = dict[str, Any]
FeatureRows = list[dict[str, float]]
SampleRows = list[np.ndarray]
ArrayPair = tuple[np.ndarray, np.ndarray]


def iter_hdf5_paths(inputs: list[Path]) -> list[Path]:
    """Return all HDF5 files from files or recursively from directories.

    Args:
        inputs (list[Path]): Input files or directories to search.

    Returns:
        list[Path]: Sorted unique HDF5 paths.
    """
    paths = []
    for input_path in inputs:
        if input_path.is_file():
            paths.append(input_path)
        elif input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.h5")))
        else:
            raise FileNotFoundError(f"input path does not exist: {input_path}")

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("no .h5 files found in the requested input path(s)")
    return paths


def read_features(h5_file: h5py.File) -> dict[str, float]:
    """Read the five link features from user_optional attributes.

    Args:
        h5_file (h5py.File): Open link simulation HDF5 file.

    Returns:
        dict[str, float]: Feature names mapped to numeric values.
    """
    if USER_OPTIONAL_PATH not in h5_file:
        raise KeyError(f"missing required group: {USER_OPTIONAL_PATH}")

    attributes = h5_file[USER_OPTIONAL_PATH].attrs
    missing = [name for name in FEATURE_NAMES if name not in attributes]
    if missing:
        raise KeyError(f"missing required feature(s): {', '.join(missing)}")

    features = {}
    for name in FEATURE_NAMES:
        value = float(attributes[name])
        if not math.isfinite(value):
            raise ValueError(f"feature {name} is not finite")
        features[name] = value
    return features


def read_time_samples(h5_file: h5py.File) -> np.ndarray:
    """Read time-to-serve samples as a one-dimensional float32 array.

    Args:
        h5_file (h5py.File): Open link simulation HDF5 file.

    Returns:
        np.ndarray: Positive time-to-serve samples in milliseconds.
    """
    if TIME_PATH not in h5_file:
        raise KeyError(f"missing required dataset: {TIME_PATH}")

    time = np.asarray(h5_file[TIME_PATH][:], dtype=np.float32)
    if time.ndim != 1:
        raise ValueError(f"{TIME_PATH} must be a 1-D time sample series, got shape {time.shape}")
    if time.size == 0:
        raise ValueError(f"{TIME_PATH} is empty")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{TIME_PATH} contains non-finite values")
    if np.any(time <= 0):
        raise ValueError(f"{TIME_PATH} must be positive before log transform")
    return time


def load_training_rows(paths: list[Path]) -> tuple[FeatureRows, SampleRows]:
    """Load one feature dictionary and one time series per HDF5 file.

    Args:
        paths (list[Path]): HDF5 files to load.

    Returns:
        tuple[FeatureRows, SampleRows]: Features and time samples grouped by condition.
    """
    feature_rows = []
    sample_rows = []
    for path in paths:
        with h5py.File(path, "r") as h5_file:
            feature_rows.append(read_features(h5_file))
            sample_rows.append(read_time_samples(h5_file))
    return feature_rows, sample_rows


def train_test_indices(condition_count: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split complete conditions between training and validation.

    Args:
        condition_count (int): Number of link conditions.
        test_fraction (float): Fraction reserved for validation.
        seed (int): Random seed used to shuffle conditions.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training and validation condition indices.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if condition_count < 2:
        raise ValueError("need at least two link conditions for a train/test split")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(condition_count)
    test_count = max(1, int(round(condition_count * test_fraction)))
    test_count = min(test_count, condition_count - 1)
    return permutation[test_count:], permutation[:test_count]


def build_sample_arrays(feature_rows: FeatureRows, sample_rows: SampleRows, condition_indices: np.ndarray,
                        max_samples_per_file: int | None, seed: int) -> ArrayPair:
    """Flatten selected conditions into float32 feature and time rows.

    Args:
        feature_rows (FeatureRows): Feature values grouped by condition.
        sample_rows (SampleRows): Time samples grouped by condition.
        condition_indices (np.ndarray): Conditions to include.
        max_samples_per_file (int | None): Maximum samples selected from each condition.
        seed (int): Random seed used for sample selection.

    Returns:
        ArrayPair: Flattened feature and target arrays.
    """
    rng = np.random.default_rng(seed)
    selected_rows = []
    for index in condition_indices:
        samples = sample_rows[index]
        if max_samples_per_file is not None and samples.size > max_samples_per_file:
            sample_indices = rng.choice(samples.size, size=max_samples_per_file, replace=False)
            samples = samples[np.sort(sample_indices)]
        selected_rows.append(samples)

    sample_count = sum(samples.size for samples in selected_rows)
    features = np.empty((sample_count, len(FEATURE_NAMES)), dtype=np.float32)
    samples = np.empty((sample_count, OUTPUT_DIM), dtype=np.float32)
    offset = 0
    for condition_index, time in zip(condition_indices, selected_rows):
        end = offset + time.size
        features[offset:end] = np.asarray([feature_rows[condition_index][name] for name in FEATURE_NAMES], dtype=np.float32)
        samples[offset:end, 0] = time
        offset = end
    return features, samples


def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate float32 normalization statistics.

    Args:
        values (np.ndarray): Values to summarize.

    Returns:
        tuple[np.ndarray, np.ndarray]: Column means and standard deviations.
    """
    mean = np.mean(values, axis=0, dtype=np.float32)
    std = np.std(values, axis=0, dtype=np.float32)
    std[std < STD_EPSILON] = 1.0
    return mean, std


def standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Standardize an array in place.

    Args:
        values (np.ndarray): Values to standardize.
        mean (np.ndarray): Column means.
        std (np.ndarray): Column standard deviations.
    """
    values -= mean
    values /= std


def transform_samples(samples: np.ndarray):
    """Log-transform positive time samples in place.

    Args:
        samples (np.ndarray): Positive time-to-serve samples.
    """
    np.maximum(samples, TIME_EPSILON, out=samples)
    np.log(samples, out=samples)


def inverse_transform_samples(transformed_samples: np.ndarray) -> np.ndarray:
    """Transform generated log-time samples back to milliseconds.

    Args:
        transformed_samples (np.ndarray): Generated samples in log-time space.

    Returns:
        np.ndarray: Time-to-serve samples in milliseconds.
    """
    return np.exp(transformed_samples)


def resolve_device(device_name: str) -> torch.device:
    """Validate and return the requested PyTorch device.

    Args:
        device_name (str): Device name such as cpu, cuda, or cuda:0.

    Returns:
        torch.device: Validated PyTorch device.
    """
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a CUDA GPU")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} was requested, but only {torch.cuda.device_count()} GPU(s) are available")
    return device


def build_mdn(input_dim: int, hidden_size: int, hidden_layers: int, components: int):
    """Build a one-dimensional Gaussian Mixture Density Network.

    Args:
        input_dim (int): Number of input features.
        hidden_size (int): Neurons in each hidden layer.
        hidden_layers (int): Number of hidden layers.
        components (int): Number of Gaussian mixture components.

    Returns:
        nn.Module: Mixture Density Network model.
    """
    class MixtureDensityNetwork(nn.Module):  # type: ignore[misc]
        """Map link features to Gaussian mixture parameters."""

        def __init__(self):
            """Initialize the shared layers and mixture output heads."""
            super().__init__()
            layers = []
            current_dim = input_dim
            for _ in range(hidden_layers):
                layers.append(nn.Linear(current_dim, hidden_size))
                layers.append(nn.ReLU())
                current_dim = hidden_size
            self.backbone = nn.Sequential(*layers)
            self.logit_head = nn.Linear(current_dim, components)
            self.mean_head = nn.Linear(current_dim, components * OUTPUT_DIM)
            self.scale_head = nn.Linear(current_dim, components * OUTPUT_DIM)

        def forward(self, x):
            """Calculate mixture logits, means, and standard deviations.

            Args:
                x (torch.Tensor): Standardized link features.

            Returns:
                tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mixture logits, means, and scales.
            """
            hidden = self.backbone(x)
            logits = self.logit_head(hidden)
            means = self.mean_head(hidden).reshape(-1, components, OUTPUT_DIM)
            raw_scales = self.scale_head(hidden).reshape(-1, components, OUTPUT_DIM)
            scales = torch_functional.softplus(raw_scales) + 1e-4
            return logits, means, scales

    return MixtureDensityNetwork()


def mdn_distribution(model, x):
    """Create the conditional Gaussian mixture predicted by the model.

    Args:
        model (nn.Module): Trained or trainable MDN.
        x (torch.Tensor): Standardized link features.

    Returns:
        torch_distributions.MixtureSameFamily: Predicted conditional distribution.
    """
    logits, means, scales = model(x)
    mixture = torch_distributions.Categorical(logits=logits)
    components = torch_distributions.Independent(torch_distributions.Normal(means, scales), 1)
    return torch_distributions.MixtureSameFamily(mixture, components)


def negative_log_likelihood(model, x, y):
    """Calculate the mean negative log likelihood of observed samples.

    Args:
        model (nn.Module): Mixture Density Network.
        x (torch.Tensor): Standardized link features.
        y (torch.Tensor): Standardized log-time samples.

    Returns:
        torch.Tensor: Scalar mean negative log likelihood.
    """
    return -mdn_distribution(model, x).log_prob(y).mean()


def evaluate_nll(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> float:
    """Evaluate negative log likelihood without updating the model.

    Args:
        model (nn.Module): Mixture Density Network to evaluate.
        x (np.ndarray): Standardized features.
        y (np.ndarray): Standardized log-time samples.
        batch_size (int): Samples evaluated per batch.
        device (torch.device): PyTorch execution device.

    Returns:
        float: Sample-weighted mean negative log likelihood.
    """
    model.eval()
    losses = []
    weights = []
    use_cuda = device.type == "cuda"
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=use_cuda)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=use_cuda)
            batch_y = batch_y.to(device, non_blocking=use_cuda)
            loss = negative_log_likelihood(model, batch_x, batch_y)
            losses.append(float(loss.item()))
            weights.append(int(batch_x.shape[0]))
    return float(np.average(losses, weights=weights))


def train_mdn(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
              args: argparse.Namespace) -> tuple[MdnArtifact, dict[str, Any]]:
    """Train the MDN and retain the state with the best validation NLL.

    Args:
        train_x (np.ndarray): Training features.
        train_y (np.ndarray): Training time samples.
        test_x (np.ndarray): Validation features.
        test_y (np.ndarray): Validation time samples.
        args (argparse.Namespace): Training hyperparameters.

    Returns:
        tuple[MdnArtifact, dict[str, Any]]: Model artifact and NLL history.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transform_samples(train_y)
    transform_samples(test_y)
    feature_mean, feature_std = mean_and_std(train_x)
    target_mean, target_std = mean_and_std(train_y)
    standardize(train_x, feature_mean, feature_std)
    standardize(test_x, feature_mean, feature_std)
    standardize(train_y, target_mean, target_std)
    standardize(test_y, target_mean, target_std)

    model = build_mdn(input_dim=train_x.shape[1], hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                      components=args.components).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    use_cuda = args.device.type == "cuda"
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=use_cuda)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_nll = math.inf
    train_nll_history = []
    validation_nll_history = []
    training_start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_weights = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device, non_blocking=use_cuda)
            batch_y = batch_y.to(args.device, non_blocking=use_cuda)
            optimizer.zero_grad()
            loss = negative_log_likelihood(model, batch_x, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_weights.append(int(batch_x.shape[0]))

        train_nll = float(np.average(epoch_losses, weights=epoch_weights))
        validation_nll = evaluate_nll(model, test_x, test_y, args.batch_size, args.device)
        train_nll_history.append(train_nll)
        validation_nll_history.append(validation_nll)
        if validation_nll < best_validation_nll:
            best_epoch = epoch
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            elapsed_time = time.perf_counter() - training_start_time
            estimated_runtime_left = elapsed_time / epoch * (args.epochs - epoch)
            print(f"Epoch {epoch:04d}: train_nll={train_nll:.6g}, validation_nll={validation_nll:.6g}")
            print(f"Estimated runtime left: {timedelta(seconds=round(estimated_runtime_left))}")

    model.load_state_dict(best_state)
    final_train_nll = evaluate_nll(model, train_x, train_y, args.batch_size, args.device)
    final_validation_nll = evaluate_nll(model, test_x, test_y, args.batch_size, args.device)
    artifact = {
        "model_type": "conditional_mixture_density_network",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "components": args.components,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "output_dim": OUTPUT_DIM,
        "target_name": TARGET_NAME,
        "output_transform": "log",
        "dtype": "float32",
    }
    metrics = {
        "train_nll": train_nll_history,
        "validation_nll": validation_nll_history,
        "best_epoch": best_epoch,
        "best_train_nll": final_train_nll,
        "best_validation_nll": final_validation_nll,
    }
    return artifact, metrics


def save_model(model_path: Path, artifact: MdnArtifact, paths: list[Path], train_conditions: int, validation_conditions: int,
               train_samples: int, validation_samples: int, metrics: dict[str, Any], args: argparse.Namespace):
    """Save the trained model and training metadata.

    Args:
        model_path (Path): Destination for the PyTorch model artifact.
        artifact (MdnArtifact): Trained model state and normalization metadata.
        paths (list[Path]): HDF5 files used for training and validation.
        train_conditions (int): Number of training conditions.
        validation_conditions (int): Number of validation conditions.
        train_samples (int): Number of training samples.
        validation_samples (int): Number of validation samples.
        metrics (dict[str, Any]): Per-epoch and best-model NLL metrics.
        args (argparse.Namespace): Training hyperparameters.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)
    metadata = {
        "model": "conditional Mixture Density Network",
        "model_path": str(model_path),
        "feature_names": list(FEATURE_NAMES),
        "target": TIME_PATH,
        "output_transform": "log",
        "dtype": "float32",
        "components": args.components,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "input_file_count": len(paths),
        "train_condition_count": train_conditions,
        "validation_condition_count": validation_conditions,
        "train_sample_count": train_samples,
        "validation_sample_count": validation_samples,
        "metrics": metrics,
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model(model_path: Path, device: torch.device) -> MdnArtifact:
    """Load a saved MDN artifact onto the requested device.

    Args:
        model_path (Path): Saved PyTorch artifact path.
        device (torch.device): Device used to map stored tensors.

    Returns:
        MdnArtifact: Loaded model artifact.
    """
    try:
        artifact = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        artifact = torch.load(model_path, map_location=device)
    return artifact


def parse_config(config_json: str | None, config_file: Path | None) -> dict[str, float]:
    """Read one prediction configuration from JSON text or a file.

    Args:
        config_json (str | None): Inline JSON configuration.
        config_file (Path | None): JSON configuration file.

    Returns:
        dict[str, float]: Parsed numeric feature values.
    """
    if config_json and config_file:
        raise ValueError("use either --config-json or --config-file, not both")
    if config_file:
        raw = config_file.read_text(encoding="utf-8")
    elif config_json:
        raw = config_json
    else:
        raise ValueError("prediction requires --config-json or --config-file")

    parsed = json.loads(raw)
    return {name: float(value) for name, value in parsed.items()}


def sample_from_model(artifact: MdnArtifact, config: dict[str, float], sample_count: int, seed: int,
                      device: torch.device) -> np.ndarray:
    """Generate time-to-serve samples for one link configuration.

    Args:
        artifact (MdnArtifact): Trained model and normalization metadata.
        config (dict[str, float]): Five link feature values.
        sample_count (int): Number of time samples to generate.
        seed (int): PyTorch random seed.
        device (torch.device): PyTorch execution device.

    Returns:
        np.ndarray: Generated time-to-serve samples in milliseconds.
    """
    missing = [name for name in FEATURE_NAMES if name not in config]
    if missing:
        raise ValueError(f"prediction config is missing required feature(s): {', '.join(missing)}")

    torch.manual_seed(seed)
    feature_vector = np.asarray([[config[name] for name in FEATURE_NAMES]], dtype=np.float32)
    feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(artifact["feature_std"], dtype=np.float32)
    target_mean = np.asarray(artifact["target_mean"], dtype=np.float32)
    target_std = np.asarray(artifact["target_std"], dtype=np.float32)
    standardized_features = (feature_vector - feature_mean) / feature_std

    model = build_mdn(input_dim=len(FEATURE_NAMES), hidden_size=int(artifact["hidden_size"]),
                      hidden_layers=int(artifact["hidden_layers"]), components=int(artifact["components"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(standardized_features).to(device)
        standardized_samples = mdn_distribution(model, x).sample((sample_count,))[:, 0, :].cpu().numpy()

    transformed_samples = standardized_samples * target_std + target_mean
    return inverse_transform_samples(transformed_samples)[:, 0]


def command_train(args: argparse.Namespace):
    """Load HDF5 data, train the MDN, and save the model artifact.

    Args:
        args (argparse.Namespace): Parsed training arguments.
    """
    args.device = resolve_device(args.device)
    if args.device.type == "cuda":
        print(f"Using device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    else:
        print(f"Using device: {args.device}")

    paths = iter_hdf5_paths(args.input)
    feature_rows, sample_rows = load_training_rows(paths)
    train_indices, validation_indices = train_test_indices(len(paths), args.test_fraction, args.seed)
    train_x, train_y = build_sample_arrays(feature_rows, sample_rows, train_indices, args.max_samples_per_file, args.seed)
    validation_x, validation_y = build_sample_arrays(feature_rows, sample_rows, validation_indices,
                                                      args.max_samples_per_file, args.seed + 1)
    train_condition_count = train_indices.size
    validation_condition_count = validation_indices.size
    del feature_rows, sample_rows, train_indices, validation_indices

    artifact, metrics = train_mdn(train_x, train_y, validation_x, validation_y, args)
    model_path = args.output.with_name(f"{args.output.stem}_validation_nll={metrics['best_validation_nll']:.10f}{args.output.suffix}")
    save_model(model_path, artifact, paths, train_condition_count, validation_condition_count, train_x.shape[0], validation_x.shape[0],
               metrics, args)
    print(f"Used {len(paths)} HDF5 file(s)")
    print(f"Training conditions: {train_condition_count}, samples: {train_x.shape[0]}")
    print(f"Validation conditions: {validation_condition_count}, samples: {validation_x.shape[0]}")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Target: {TIME_PATH}")
    print(f"Wrote model: {model_path}")
    print(f"Wrote metadata: {model_path.with_suffix('.json')}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best-model NLL: train={metrics['best_train_nll']:.6g}, validation={metrics['best_validation_nll']:.6g}")


def command_predict(args: argparse.Namespace):
    """Load an MDN and print generated time-to-serve samples.

    Args:
        args (argparse.Namespace): Parsed prediction arguments.
    """
    args.device = resolve_device(args.device)
    artifact = load_model(args.model, args.device)
    config = parse_config(args.config_json, args.config_file)
    samples = sample_from_model(artifact, config, args.sample_count, args.seed, args.device)
    print(json.dumps({"target": TARGET_NAME, "samples": samples.tolist()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the training and prediction command-line interface.

    Returns:
        argparse.ArgumentParser: Configured top-level parser.
    """
    parser = argparse.ArgumentParser(description="Train or query a conditional MDN for two-node link service time.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a conditional MDN surrogate.")
    train_parser.add_argument("--input", type=Path, nargs="+", default=[Path("link/7.20.2026/h5")],
                              help="HDF5 file(s) or directory tree(s). Defaults to link/7.20.2026/h5.")
    train_parser.add_argument("--output", type=Path, default=Path("hdf5/surrogate_mdn.pt"))
    train_parser.add_argument("--components", type=int, default=8, help="Number of Gaussian mixture components.")
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--hidden-layers", type=int, default=2)
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=4096)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--max-samples-per-file", type=int, default=1000, help="Maximum samples used from each HDF5 file.")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--log-interval", type=int, default=1)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Generate time-to-serve samples for one configuration.")
    predict_parser.add_argument("--model", type=Path, default=Path("hdf5/surrogate_mdn.pt"))
    predict_parser.add_argument("--config-json", help="JSON object containing the five link features.")
    predict_parser.add_argument("--config-file", type=Path, help="Path to a JSON configuration object.")
    predict_parser.add_argument("--sample-count", type=int, default=10000)
    predict_parser.add_argument("--seed", type=int, default=7)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.set_defaults(func=command_predict)
    return parser


def main():
    """Parse command-line arguments and run the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
