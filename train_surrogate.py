"""Train a conditional Mixture Density Network surrogate from qnet HDF5 files.

The surrogate learns a conditional density over individual ``(time, fidelity)``
samples:

    p(time, fidelity | qnet/metadata/user_optional configuration)

At prediction time, the model samples from that learned distribution to produce
as many synthetic ``(time, fidelity)`` rows as requested for one configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.distributions as torch_distributions
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, TensorDataset


USER_OPTIONAL_PATH = "qnet/metadata/user_optional"
TIME_PATH = "qnet/simulation_log/time"
STATE_PATH = "qnet/simulation_log/state"
TARGET_NAME = "samples"
TARGET_COLUMNS = ("time", "fidelity")
OUTPUT_DIM = len(TARGET_COLUMNS)
TIME_EPSILON = 1e-12
FIDELITY_EPSILON = 1e-9
STD_EPSILON = 1e-12
FILENAME_PATTERN = re.compile(r"^linear_(?P<nodes>\d+)(?:_swap_prob=(?P<swap_prob>\d+(?:\.\d+)?))?\.h5$")
MdnArtifact = dict[str, Any]
FeatureRows = list[dict[str, float]]
SampleRows = list[np.ndarray]
ArrayPair = tuple[np.ndarray, np.ndarray]


def iter_hdf5_paths(inputs: list[Path]) -> list[Path]:
    """Return all HDF5 files from files or recursively from directories."""
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            paths.append(input_path)
        elif input_path.is_dir():
            paths.extend(sorted(input_path.rglob("*.h5")))
        else:
            raise FileNotFoundError(f"input path does not exist: {input_path}")

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError("no .h5 files found in the requested input path(s)")
    return unique_paths


def scalar_attr(value: Any) -> Any:
    """Convert common HDF5 scalar attribute values into Python scalars."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return scalar_attr(value.item())
        if value.size == 1:
            return scalar_attr(value.reshape(()).item())
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def numeric_user_optional_attrs(h5_file: h5py.File) -> dict[str, float]:
    """Read numeric user_optional attributes for use as surrogate inputs."""
    if USER_OPTIONAL_PATH not in h5_file:
        raise KeyError(f"missing required group: {USER_OPTIONAL_PATH}")

    features: dict[str, float] = {}
    for name, raw_value in h5_file[USER_OPTIONAL_PATH].attrs.items():
        value = scalar_attr(raw_value)
        if isinstance(value, bool):
            features[name] = float(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            features[name] = float(value)
    return features


def supplement_features_from_filename(features: dict[str, float], path: Path) -> dict[str, float]:
    """Backfill topology features for HDF5 files generated before metadata included them."""
    match = FILENAME_PATTERN.match(path.name)
    if not match:
        return features

    supplemented = dict(features)
    supplemented.setdefault("node_count", float(match.group("nodes")))
    swap_prob = match.group("swap_prob")
    if swap_prob is not None:
        supplemented.setdefault("swapping_success_prob", float(swap_prob))
    return supplemented


def read_output_samples(h5_file: h5py.File) -> np.ndarray:
    """Read simulation outputs as ordered ``(time, fidelity)`` samples."""
    if TIME_PATH not in h5_file:
        raise KeyError(f"missing required dataset: {TIME_PATH}")
    if STATE_PATH not in h5_file:
        raise KeyError(f"missing required dataset: {STATE_PATH}")

    time = np.asarray(h5_file[TIME_PATH][:], dtype=np.float64)
    state = np.asarray(h5_file[STATE_PATH][:], dtype=np.float64)
    if time.ndim != 1:
        raise ValueError(f"{TIME_PATH} must be a 1-D time sample series, got shape {time.shape}")
    if state.ndim != 1:
        raise ValueError(f"{STATE_PATH} must be a 1-D state sample series, got shape {state.shape}")
    if time.size == 0:
        raise ValueError(f"{TIME_PATH} is empty")
    if state.size == 0:
        raise ValueError(f"{STATE_PATH} is empty")
    if time.size != state.size:
        raise ValueError(f"{TIME_PATH} and {STATE_PATH} have different sample counts: {time.size} != {state.size}")
    if not np.all(np.isfinite(time)):
        raise ValueError(f"{TIME_PATH} contains non-finite values")
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{STATE_PATH} contains non-finite values")
    if np.any(time <= 0):
        raise ValueError(f"{TIME_PATH} must be positive before log transform")
    if np.any((state <= 0) | (state >= 1)):
        raise ValueError(f"{STATE_PATH} must be strictly between 0 and 1 before logit transform")

    return np.column_stack((time, state))


def load_training_rows(paths: list[Path]) -> tuple[list[dict[str, float]], list[np.ndarray], list[Path]]:
    """Load one feature dictionary and many ``(time, fidelity)`` samples per HDF5 file."""
    feature_rows: list[dict[str, float]] = []
    sample_rows: list[np.ndarray] = []
    used_paths: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    for path in paths:
        with h5py.File(path, "r") as h5_file:
            features = supplement_features_from_filename(numeric_user_optional_attrs(h5_file), path)
            samples = read_output_samples(h5_file)

        if features:
            feature_rows.append(features)
            sample_rows.append(samples)
            used_paths.append(path)
        else:
            skipped.append((path, "no numeric user_optional attributes"))

    if skipped:
        print(f"Skipped {len(skipped)} file(s). First skipped file: {skipped[0][0]} ({skipped[0][1]})")
    if not sample_rows:
        raise ValueError("no usable HDF5 files found")
    return feature_rows, sample_rows, used_paths


def choose_feature_names(feature_rows: list[dict[str, float]]) -> list[str]:
    """Use numeric attributes present in every file and drop constants."""
    common_names = set(feature_rows[0])
    for row in feature_rows[1:]:
        common_names &= set(row)

    if not common_names:
        raise ValueError("no numeric user_optional attributes are shared by all usable files")

    varying_names = []
    for name in sorted(common_names):
        values = np.array([row[name] for row in feature_rows], dtype=np.float64)
        if np.nanmax(values) > np.nanmin(values):
            varying_names.append(name)

    if not varying_names:
        raise ValueError("all shared numeric user_optional attributes are constant; "
                         "a surrogate for various configurations needs multiple configurations")
    return varying_names


def build_sample_arrays(feature_rows: FeatureRows, sample_rows: SampleRows, feature_names: list[str],
                        max_samples_per_file: Optional[int], seed: int) -> ArrayPair:
    """Repeat each configuration row once per observed output sample."""
    rng = np.random.default_rng(seed)
    feature_blocks: list[np.ndarray] = []
    sample_blocks: list[np.ndarray] = []

    for features, samples in zip(feature_rows, sample_rows):
        selected_samples = samples
        if max_samples_per_file is not None and samples.shape[0] > max_samples_per_file:
            indices = rng.choice(samples.shape[0], size=max_samples_per_file, replace=False)
            selected_samples = samples[np.sort(indices)]

        feature_vector = np.array([features[name] for name in feature_names], dtype=np.float64)
        feature_blocks.append(np.repeat(feature_vector.reshape(1, -1), selected_samples.shape[0], axis=0))
        sample_blocks.append(selected_samples)

    return np.vstack(feature_blocks), np.vstack(sample_blocks)


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std < STD_EPSILON, 1.0, std)
    return (values - mean) / std, mean, std


def transform_samples(samples: np.ndarray) -> np.ndarray:
    """Map positive/bounded outputs into unconstrained MDN training space."""
    time = np.maximum(samples[:, 0], TIME_EPSILON)
    fidelity = np.clip(samples[:, 1], FIDELITY_EPSILON, 1.0 - FIDELITY_EPSILON)
    return np.column_stack((np.log(time), np.log(fidelity / (1.0 - fidelity))))


def inverse_transform_samples(transformed_samples: np.ndarray) -> np.ndarray:
    time = np.exp(transformed_samples[:, 0])
    fidelity_logits = np.clip(transformed_samples[:, 1], -60.0, 60.0)
    fidelity = 1.0 / (1.0 + np.exp(-fidelity_logits))
    return np.column_stack((time, fidelity))


def build_mdn(input_dim: int, hidden_size: int, hidden_layers: int, components: int):
    """Build a diagonal-covariance MDN with Gaussian components."""
    class MixtureDensityNetwork(nn.Module):  # type: ignore[misc]
        def __init__(self):
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
            hidden = self.backbone(x)
            logits = self.logit_head(hidden)
            means = self.mean_head(hidden).reshape(-1, components, OUTPUT_DIM)
            raw_scales = self.scale_head(hidden).reshape(-1, components, OUTPUT_DIM)
            scales = torch_functional.softplus(raw_scales) + 1e-4
            return logits, means, scales

    return MixtureDensityNetwork()


def mdn_distribution(model, x):
    logits, means, scales = model(x)
    mixture = torch_distributions.Categorical(logits=logits)
    components = torch_distributions.Independent(torch_distributions.Normal(means, scales), 1)
    return torch_distributions.MixtureSameFamily(mixture, components)


def negative_log_likelihood(model, x, y):
    return -mdn_distribution(model, x).log_prob(y).mean()


def train_test_indices(row_count: int, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if row_count < 2:
        raise ValueError("need at least 2 sample rows for a train/test split")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(row_count)
    test_count = max(1, int(round(row_count * test_fraction)))
    test_count = min(test_count, row_count - 1)
    return permutation[test_count:], permutation[:test_count]


def evaluate_nll(model, x: np.ndarray, y: np.ndarray, batch_size: int, device: str) -> float:
    model.eval()
    losses: list[float] = []
    weights: list[int] = []
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            loss = negative_log_likelihood(model, batch_x, batch_y)
            losses.append(float(loss.item()))
            weights.append(int(batch_x.shape[0]))
    return float(np.average(losses, weights=weights))


def train_mdn(features: np.ndarray, samples: np.ndarray,
              args: argparse.Namespace) -> tuple[MdnArtifact, dict[str, float]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    transformed_samples = transform_samples(samples)
    standardized_features, feature_mean, feature_std = standardize(features)
    standardized_targets, target_mean, target_std = standardize(transformed_samples)

    train_idx, test_idx = train_test_indices(standardized_features.shape[0], args.test_fraction, args.seed)
    train_x = standardized_features[train_idx]
    train_y = standardized_targets[train_idx]
    test_x = standardized_features[test_idx]
    test_y = standardized_targets[test_idx]

    model = build_mdn(input_dim=train_x.shape[1], hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                      components=args.components).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(torch.as_tensor(train_x, dtype=torch.float32),
                            torch.as_tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_nll = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        epoch_weights: list[int] = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(args.device)
            batch_y = batch_y.to(args.device)
            optimizer.zero_grad()
            loss = negative_log_likelihood(model, batch_x, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            epoch_weights.append(int(batch_x.shape[0]))

        train_nll = float(np.average(epoch_losses, weights=epoch_weights))
        validation_nll = evaluate_nll(model, test_x, test_y, args.batch_size, args.device)
        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            print(f"Epoch {epoch:04d}: train_nll={train_nll:.6g}, validation_nll={validation_nll:.6g}")

    model.load_state_dict(best_state)
    final_train_nll = evaluate_nll(model, train_x, train_y, args.batch_size, args.device)
    final_validation_nll = evaluate_nll(model, test_x, test_y, args.batch_size, args.device)

    artifact = {
        "model_type": "conditional_mixture_density_network",
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "target_mean": target_mean.tolist(),
        "target_std": target_std.tolist(),
        "components": args.components,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "output_dim": OUTPUT_DIM,
        "target_name": TARGET_NAME,
        "target_columns": TARGET_COLUMNS,
        "output_transform": {"time": "log", "fidelity": "logit"},
    }
    metrics = {"train_nll": final_train_nll, "validation_nll": final_validation_nll}
    return artifact, metrics


def save_model(model_path: Path, artifact: MdnArtifact, feature_names: list[str], train_paths: list[Path],
               sample_count: int, metrics: dict[str, float], args: argparse.Namespace):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(artifact)
    artifact["feature_names"] = feature_names
    torch.save(artifact, model_path)

    metadata = {
        "model": "conditional Mixture Density Network",
        "model_path": str(model_path),
        "feature_names": feature_names,
        "target": [TIME_PATH, STATE_PATH],
        "target_format": TARGET_NAME,
        "target_columns": list(TARGET_COLUMNS),
        "output_transform": artifact["output_transform"],
        "components": args.components,
        "hidden_size": args.hidden_size,
        "hidden_layers": args.hidden_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "training_file_count": len(train_paths),
        "training_sample_count": sample_count,
        "metrics": metrics,
    }
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model(model_path: Path, device: str) -> MdnArtifact:
    try:
        artifact = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        artifact = torch.load(model_path, map_location=device)
    return artifact


def parse_config(config_json: Optional[str], config_file: Optional[Path]) -> dict[str, float]:
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
                      device: str) -> np.ndarray:
    feature_names = list(artifact["feature_names"])
    missing = [name for name in feature_names if name not in config]
    if missing:
        raise ValueError(f"prediction config is missing required feature(s): {', '.join(missing)}")

    torch.manual_seed(seed)
    feature_vector = np.array([[config[name] for name in feature_names]], dtype=np.float64)
    feature_mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
    feature_std = np.asarray(artifact["feature_std"], dtype=np.float64)
    target_mean = np.asarray(artifact["target_mean"], dtype=np.float64)
    target_std = np.asarray(artifact["target_std"], dtype=np.float64)

    standardized_features = (feature_vector - feature_mean) / feature_std
    model = build_mdn(input_dim=len(feature_names), hidden_size=int(artifact["hidden_size"]),
                      hidden_layers=int(artifact["hidden_layers"]), components=int(artifact["components"])).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.as_tensor(standardized_features, dtype=torch.float32, device=device)
        standardized_samples = mdn_distribution(model, x).sample((sample_count,))[:, 0, :].cpu().numpy()

    transformed_samples = standardized_samples * target_std + target_mean
    return inverse_transform_samples(transformed_samples)


def command_train(args: argparse.Namespace):
    paths = iter_hdf5_paths(args.input)
    feature_rows, sample_rows, used_paths = load_training_rows(paths)

    if len(feature_rows) < 2:
        raise ValueError("only one usable configuration was found; "
                         "train on HDF5 files with different user_optional values")

    feature_names = choose_feature_names(feature_rows)
    features, samples = build_sample_arrays(feature_rows=feature_rows, sample_rows=sample_rows,
                                            feature_names=feature_names,
                                            max_samples_per_file=args.max_samples_per_file, seed=args.seed)

    artifact, metrics = train_mdn(features, samples, args)
    save_model(args.output, artifact, feature_names, used_paths, samples.shape[0], metrics, args)

    print(f"Used {len(used_paths)} HDF5 file(s)")
    print(f"Training samples: {samples.shape[0]}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Wrote model: {args.output}")
    print(f"Wrote metadata: {args.output.with_suffix('.json')}")
    print(f"Final NLL: train={metrics['train_nll']:.6g}, validation={metrics['validation_nll']:.6g}")


def command_predict(args: argparse.Namespace):
    artifact = load_model(args.model, args.device)
    config = parse_config(args.config_json, args.config_file)
    samples = sample_from_model(artifact, config, args.sample_count, args.seed, args.device)
    print(json.dumps({"columns": list(TARGET_COLUMNS), TARGET_NAME: samples.tolist()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or query a conditional MDN surrogate from qnet HDF5 files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a conditional MDN surrogate.")
    train_parser.add_argument("--input", type=Path, nargs="+", default=[Path("hdf5/sequence_linear")],
                              help="HDF5 file(s) or directory tree(s). Defaults to hdf5/sequence_linear.")
    train_parser.add_argument("--output", type=Path, default=Path("hdf5/surrogate_mdn.pt"))
    train_parser.add_argument("--components", type=int, default=8, help="Number of Gaussian mixture components.")
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--hidden-layers", type=int, default=2)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=1024)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--test-fraction", type=float, default=0.2)
    train_parser.add_argument("--max-samples-per-file", type=int, default=None,
                              help="Optional cap for subsampling samples from each HDF5 file during training.")
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--log-interval", type=int, default=10)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Generate synthetic output samples for one configuration.")
    predict_parser.add_argument("--model", type=Path, default=Path("hdf5/surrogate_mdn.pt"))
    predict_parser.add_argument("--config-json", help="JSON object with user_optional values.")
    predict_parser.add_argument("--config-file", type=Path, help="Path to a JSON configuration object.")
    predict_parser.add_argument("--sample-count", type=int, default=10000)
    predict_parser.add_argument("--seed", type=int, default=7)
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.set_defaults(func=command_predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
