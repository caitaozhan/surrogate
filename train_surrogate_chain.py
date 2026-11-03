"""Train a globally conditioned Transformer-MDN for repeater-chain time and fidelity."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
import torch.distributions as torch_distributions
import torch.nn as nn
import torch.nn.functional as torch_functional
from torch.utils.data import DataLoader, Dataset


USER_OPTIONAL_PATH = "qnet/metadata/user_optional"
TIME_PATH = "qnet/simulation_log/time"
FIDELITY_PATH = "qnet/simulation_log/state"
LINK_DISTANCES_NAME = "link_distances"
GLOBAL_FEATURE_NAMES = ("memory_efficiency", "detector_efficiency", "attenuation", "classical_delay_node",
                        "swapping_probability", "initial_fidelity", "coherence_time", "classical_delay_hop")
TARGET_NAMES = ("time", "fidelity")
OUTPUT_DIM = 2
CONDITIONING_METHOD = "global_conditioned_class_token"
TIME_EPSILON = np.float32(1e-12)
FIDELITY_EPSILON = np.float32(1e-6)
STD_EPSILON = np.float32(1e-12)
MdnArtifact = dict[str, Any]
Normalization = dict[str, np.ndarray]
RepeaterChainBatch = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class RepeaterChainData:
    """One repeater chain configuration and its paired simulation outputs.
    """
    path: Path                   # Source HDF5 file path for this chain configuration.
    link_distances: np.ndarray   # Per-link distances with shape (link_count,).
    global_features: np.ndarray  # Homogeneous features ordered by GLOBAL_FEATURE_NAMES.
    targets: np.ndarray          # Paired (time, fidelity) rows with shape (sample_count, 2).


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


def read_chain_data(path: Path, max_samples_per_file: int | None, rng: np.random.Generator) -> RepeaterChainData:
    """Read one chain configuration and its paired time-fidelity samples.

    Args:
        path (Path): Chain simulation HDF5 file.
        max_samples_per_file (int | None): Maximum paired samples selected from the file.
        rng (np.random.Generator): Random generator used to subsample targets.

    Returns:
        RepeaterChainData: Validated configuration and paired targets.
    """
    with h5py.File(path, "r") as h5_file:
        if USER_OPTIONAL_PATH not in h5_file:
            raise KeyError(f"{path}: missing required group: {USER_OPTIONAL_PATH}")
        attributes = h5_file[USER_OPTIONAL_PATH].attrs
        missing = [name for name in (LINK_DISTANCES_NAME, *GLOBAL_FEATURE_NAMES) if name not in attributes]
        if missing:
            raise KeyError(f"{path}: missing required feature(s): {', '.join(missing)}")

        link_distances = np.asarray(attributes[LINK_DISTANCES_NAME], dtype=np.float32)
        if link_distances.ndim != 1 or link_distances.size == 0:
            raise ValueError(f"{path}: {LINK_DISTANCES_NAME} must be a nonempty one-dimensional array")
        if not np.all(np.isfinite(link_distances)) or np.any(link_distances <= 0):
            raise ValueError(f"{path}: {LINK_DISTANCES_NAME} must contain positive finite values")
        if "number_of_nodes" in attributes and int(attributes["number_of_nodes"]) != link_distances.size + 1:
            raise ValueError(f"{path}: number_of_nodes does not match {LINK_DISTANCES_NAME}")

        global_features = np.asarray([attributes[name] for name in GLOBAL_FEATURE_NAMES], dtype=np.float32)
        if not np.all(np.isfinite(global_features)):
            raise ValueError(f"{path}: global features contain non-finite values")
        if TIME_PATH not in h5_file or FIDELITY_PATH not in h5_file:
            raise KeyError(f"{path}: missing {TIME_PATH} or {FIDELITY_PATH}")
        time_samples = np.asarray(h5_file[TIME_PATH][:], dtype=np.float32)
        fidelity_samples = np.asarray(h5_file[FIDELITY_PATH][:], dtype=np.float32)

    if time_samples.ndim != 1 or fidelity_samples.ndim != 1 or time_samples.shape != fidelity_samples.shape:
        raise ValueError(f"{path}: time and fidelity must be paired one-dimensional arrays")
    if time_samples.size == 0:
        raise ValueError(f"{path}: target arrays are empty")
    if not np.all(np.isfinite(time_samples)) or np.any(time_samples <= 0):
        raise ValueError(f"{path}: time samples must be positive and finite")
    if not np.all(np.isfinite(fidelity_samples)) or np.any((fidelity_samples < 0) | (fidelity_samples > 1)):
        raise ValueError(f"{path}: fidelity samples must be finite and between 0 and 1")

    if max_samples_per_file is not None and time_samples.size > max_samples_per_file:
        selected = np.sort(rng.choice(time_samples.size, size=max_samples_per_file, replace=False))
        time_samples = time_samples[selected]
        fidelity_samples = fidelity_samples[selected]
    targets = np.column_stack((time_samples, fidelity_samples)).astype(np.float32, copy=False)
    return RepeaterChainData(path, link_distances, global_features, targets)


def load_chain_data(paths: list[Path], max_samples_per_file: int | None, seed: int) -> list[RepeaterChainData]:
    """Load all requested chain data into memory.

    Args:
        paths (list[Path]): HDF5 condition files.
        max_samples_per_file (int | None): Maximum paired samples loaded from each file.
        seed (int): Random seed used for target subsampling.

    Returns:
        list[RepeaterChainData]: Loaded chain data.
    """
    rng = np.random.default_rng(seed)
    return [read_chain_data(path, max_samples_per_file, rng) for path in paths]


def split_conditions(chain_data: list[RepeaterChainData], validation_fraction: float,
                     seed: int) -> tuple[list[RepeaterChainData], list[RepeaterChainData]]:
    """Split complete conditions between training and validation.

    Args:
        chain_data (list[RepeaterChainData]): Loaded chain data.
        validation_fraction (float): Fraction reserved for validation.
        seed (int): Random seed used to shuffle conditions.

    Returns:
        tuple[list[RepeaterChainData], list[RepeaterChainData]]: Training and validation data.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1")
    if len(chain_data) < 2:
        raise ValueError("need at least two chain conditions for a training/validation split")

    permutation = np.random.default_rng(seed).permutation(len(chain_data))
    validation_count = max(1, int(round(len(chain_data) * validation_fraction)))
    validation_count = min(validation_count, len(chain_data) - 1)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    training = [chain_data[index] for index in training_indices]
    validation = [chain_data[index] for index in validation_indices]
    return training, validation


def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate float32 column means and standard deviations.

    Args:
        values (np.ndarray): Values to summarize.

    Returns:
        tuple[np.ndarray, np.ndarray]: Column means and standard deviations.
    """
    mean = np.mean(values, axis=0, dtype=np.float32)
    std = np.std(values, axis=0, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std[std < STD_EPSILON] = 1.0
    return np.asarray(mean, dtype=np.float32), std


def transform_targets(targets: np.ndarray) -> np.ndarray:
    """Transform positive time and bounded fidelity into unconstrained space.

    Args:
        targets (np.ndarray): Raw paired time and fidelity samples.

    Returns:
        np.ndarray: Log-time and logit-fidelity samples.
    """
    transformed = targets.astype(np.float32, copy=True)
    np.maximum(transformed[:, 0], TIME_EPSILON, out=transformed[:, 0])
    np.log(transformed[:, 0], out=transformed[:, 0])
    np.clip(transformed[:, 1], FIDELITY_EPSILON, 1 - FIDELITY_EPSILON, out=transformed[:, 1])
    transformed[:, 1] = np.log(transformed[:, 1] / (1 - transformed[:, 1]))
    return transformed


def inverse_transform_targets(transformed: np.ndarray) -> np.ndarray:
    """Convert generated log-time and logit-fidelity samples to physical values.

    Args:
        transformed (np.ndarray): Generated samples in transformed target space.

    Returns:
        np.ndarray: Paired time in milliseconds and fidelity in [0, 1].
    """
    targets = transformed.astype(np.float32, copy=True)
    targets[:, 0] = np.exp(targets[:, 0])
    targets[:, 1] = 1 / (1 + np.exp(-np.clip(targets[:, 1], -80, 80)))
    return targets


def calculate_normalization(training: list[RepeaterChainData]) -> Normalization:
    """Calculate input and transformed-target statistics from training conditions.

    Args:
        training (list[RepeaterChainData]): Training chain data.

    Returns:
        Normalization: Means and standard deviations for model inputs and targets.
    """
    link_distances = np.concatenate([data.link_distances for data in training]).reshape(-1, 1)
    global_features = np.stack([data.global_features for data in training])
    targets = np.concatenate([transform_targets(data.targets) for data in training])
    distance_mean, distance_std = mean_and_std(link_distances)
    global_mean, global_std = mean_and_std(global_features)
    target_mean, target_std = mean_and_std(targets)
    return {"distance_mean": distance_mean, "distance_std": distance_std,
            "global_mean": global_mean, "global_std": global_std,
            "target_mean": target_mean, "target_std": target_std}


def normalize_chain_data(chain_data: list[RepeaterChainData], normalization: Normalization) -> list[RepeaterChainData]:
    """Standardize chain inputs and transformed targets.

    Args:
        chain_data (list[RepeaterChainData]): Chain data to standardize.
        normalization (Normalization): Training normalization statistics.

    Returns:
        list[RepeaterChainData]: Standardized copies of the chain data.
    """
    normalized_data = []
    for data in chain_data:
        link_distances = (data.link_distances - normalization["distance_mean"][0]) / normalization["distance_std"][0]
        global_features = (data.global_features - normalization["global_mean"]) / normalization["global_std"]
        targets = (transform_targets(data.targets) - normalization["target_mean"]) / normalization["target_std"]
        normalized_data.append(RepeaterChainData(data.path, link_distances.astype(np.float32), 
                                                 global_features.astype(np.float32), targets.astype(np.float32)))
    return normalized_data


class RepeaterChainDataset(Dataset):
    """Expose repeater-chain data to a condition-level DataLoader."""

    def __init__(self, chain_data: list[RepeaterChainData]):
        self.chain_data = chain_data

    def __len__(self) -> int:
        return len(self.chain_data)

    def __getitem__(self, index: int) -> RepeaterChainData:
        return self.chain_data[index]


def collate_chain_data(chain_data: list[RepeaterChainData]) -> RepeaterChainBatch:
    """Pad variable link sequences and target sets for one condition batch.

    Args:
        chain_data (list[RepeaterChainData]): Chain data in the batch.

    Returns:
        RepeaterChainBatch: Distances, padding mask, global features, targets, and target mask.
    """
    batch_size = len(chain_data)
    maximum_links = max(data.link_distances.size for data in chain_data)
    maximum_samples = max(data.targets.shape[0] for data in chain_data)
    link_distances = torch.zeros((batch_size, maximum_links, 1), dtype=torch.float32)
    link_padding_mask = torch.ones((batch_size, maximum_links), dtype=torch.bool)
    global_features = torch.empty((batch_size, len(GLOBAL_FEATURE_NAMES)), dtype=torch.float32)
    targets = torch.zeros((batch_size, maximum_samples, OUTPUT_DIM), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, maximum_samples), dtype=torch.bool)
    for index, data in enumerate(chain_data):
        link_count = data.link_distances.size
        sample_count = data.targets.shape[0]
        link_distances[index, :link_count, 0] = torch.from_numpy(data.link_distances)
        link_padding_mask[index, :link_count] = False
        global_features[index] = torch.from_numpy(data.global_features)
        targets[index, :sample_count] = torch.from_numpy(data.targets)
        target_mask[index, :sample_count] = True
    return link_distances, link_padding_mask, global_features, targets, target_mask


class ChainTransformerMdn(nn.Module):
    """Encode globally conditioned link distances and predict a full-covariance Gaussian mixture."""

    def __init__(self, components: int, embedding_dim: int, transformer_layers: int, attention_heads: int,
                 feedforward_dim: int, dropout: float, hidden_size: int, hidden_layers: int, max_links: int):
        super().__init__()
        self.components = components                                                          # Number of Gaussian mixture components.
        self.embedding_dim = embedding_dim                                                    # Width of each Transformer token.
        self.max_links = max_links                                                            # Maximum link sequence accepted by the model.
        self.distance_projection = nn.Linear(1, embedding_dim)                                # Embed each scalar distance as one token.
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))                     # Shared learnable summary token.
        self.position_embedding = nn.Parameter(torch.zeros(1, max_links + 1, embedding_dim))  # Identify each token's chain position.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=attention_heads,
                                                   dim_feedforward=feedforward_dim, dropout=dropout, activation="gelu",
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers,
                                                 enable_nested_tensor=False)                  # Contextualize the summary and link tokens.
        self.chain_norm = nn.LayerNorm(embedding_dim)                                         # Normalize the encoded summary token.
        self.global_projection = nn.Sequential(nn.Linear(len(GLOBAL_FEATURE_NAMES), embedding_dim),
                                               nn.GELU())                                     # Embed global features to condition the summary token.

        layers = []
        current_dim = embedding_dim
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(current_dim, hidden_size), nn.GELU(), nn.Dropout(dropout)))
            current_dim = hidden_size
        self.backbone = nn.Sequential(*layers)                                                # Produce shared hidden features for the MDN heads.
        self.logit_head = nn.Linear(current_dim, components)                                  # Predict unnormalized mixture weights.
        self.mean_head = nn.Linear(current_dim, components * OUTPUT_DIM)                      # Predict one bivariate mean per component.
        self.cholesky_head = nn.Linear(current_dim, components * 3)                           # Predict three Cholesky entries per component.
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, link_distances: torch.Tensor, link_padding_mask: torch.Tensor,
                global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate mixture logits, means, and Cholesky factors.

        Args:
            link_distances (torch.Tensor): Standardized distances with shape (batch, links, 1).
            link_padding_mask (torch.Tensor): True for padded link positions.
            global_features (torch.Tensor): Standardized homogeneous chain features.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Mixture logits, means, and lower-triangular scales.
        """
        batch_size, link_count, _ = link_distances.shape                                                 # Read batch and sequence dimensions.
        if link_count > self.max_links:
            raise ValueError(f"received {link_count} links, but the model supports at most {self.max_links}")
        link_tokens = self.distance_projection(link_distances)                                           # Embed each scalar link distance.
        global_embedding = self.global_projection(global_features)                                       # Embed the global chain parameters.
        class_token = self.class_token.expand(batch_size, -1, -1) + global_embedding.unsqueeze(1)        # Create sample-specific summary tokens.
        tokens = torch.cat((class_token, link_tokens), dim=1)                                            # Prepend the summary token.
        tokens = tokens + self.position_embedding[:, :link_count + 1]                                    # Add learned absolute positions.
        class_padding = torch.zeros((batch_size, 1), dtype=torch.bool, device=link_padding_mask.device)  # Keep the summary token unmasked.
        padding_mask = torch.cat((class_padding, link_padding_mask), dim=1)                              # Prepend the summary-mask column.
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)                            # Contextualize all non-padded tokens.
        hidden = self.backbone(self.chain_norm(encoded[:, 0]))                                           # Process the encoded summary token.
        logits = self.logit_head(hidden)                                                                 # Predict mixture-component logits.
        means = self.mean_head(hidden).reshape(-1, self.components, OUTPUT_DIM)                          # Predict bivariate component means.
        raw_cholesky = self.cholesky_head(hidden).reshape(-1, self.components, 3)                        # Predict unconstrained Cholesky values.
        diagonal_1 = torch_functional.softplus(raw_cholesky[..., 0]) + 1e-4                              # Make the first diagonal positive.
        off_diagonal = raw_cholesky[..., 1]                                                              # Keep covariance coupling unrestricted.
        diagonal_2 = torch_functional.softplus(raw_cholesky[..., 2]) + 1e-4                              # Make the second diagonal positive.
        zeros = torch.zeros_like(diagonal_1)                                                             # Create fixed upper-right zeros.
        first_row = torch.stack((diagonal_1, zeros), dim=-1)                                             # Build the first Cholesky row.
        second_row = torch.stack((off_diagonal, diagonal_2), dim=-1)                                     # Build the second Cholesky row.
        scale_tril = torch.stack((first_row, second_row), dim=-2)                                        # Assemble lower-triangular factors.
        return logits, means, scale_tril                                                                 # Return the MDN parameters.


def mdn_distribution(model: ChainTransformerMdn, link_distances: torch.Tensor, link_padding_mask: torch.Tensor,
                     global_features: torch.Tensor) -> torch_distributions.MixtureSameFamily:
    """Create the conditional bivariate Gaussian mixture predicted by the model.

    Args:
        model (ChainTransformerMdn): Transformer-conditioned MDN.
        link_distances (torch.Tensor): Standardized padded link distances.
        link_padding_mask (torch.Tensor): True for padded link positions.
        global_features (torch.Tensor): Standardized homogeneous features.

    Returns:
        torch_distributions.MixtureSameFamily: Predicted joint time-fidelity distribution.
    """
    logits, means, scale_tril = model(link_distances, link_padding_mask, global_features)
    mixture = torch_distributions.Categorical(logits=logits)
    components = torch_distributions.MultivariateNormal(loc=means, scale_tril=scale_tril)
    return torch_distributions.MixtureSameFamily(mixture, components)


def negative_log_likelihood(model: ChainTransformerMdn, batch: RepeaterChainBatch) -> torch.Tensor:
    """Calculate sample-weighted negative log likelihood for a condition batch.

    Args:
        model (ChainTransformerMdn): Transformer-conditioned MDN.
        batch (RepeaterChainBatch): Padded repeater-chain batch.

    Returns:
        torch.Tensor: Scalar mean negative log likelihood.
    """
    link_distances, link_padding_mask, global_features, targets, target_mask = batch
    distribution = mdn_distribution(model, link_distances, link_padding_mask, global_features)
    log_probabilities = distribution.log_prob(targets.transpose(0, 1)).transpose(0, 1)
    return -log_probabilities[target_mask].mean()


def move_batch(batch: RepeaterChainBatch, device: torch.device) -> RepeaterChainBatch:
    """Move a condition batch to the requested device.

    Args:
        batch (RepeaterChainBatch): CPU repeater-chain batch.
        device (torch.device): Destination device.

    Returns:
        RepeaterChainBatch: Batch tensors on the destination device.
    """
    non_blocking = device.type == "cuda"
    return tuple(tensor.to(device, non_blocking=non_blocking) for tensor in batch)  # type: ignore[return-value]


def build_loader(chain_data: list[RepeaterChainData], batch_size: int, shuffle: bool, device: torch.device,
                 workers: int, seed: int) -> DataLoader:
    """Build a condition-level DataLoader.

    Args:
        chain_data (list[RepeaterChainData]): Chain data exposed by the loader.
        batch_size (int): Conditions per batch.
        shuffle (bool): Whether to shuffle conditions.
        device (torch.device): Training or evaluation device.
        workers (int): Number of DataLoader worker processes.
        seed (int): Seed used for condition shuffling.

    Returns:
        DataLoader: Configured condition-level loader.
    """
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(RepeaterChainDataset(chain_data), batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=collate_chain_data, pin_memory=device.type == "cuda", generator=generator)


def evaluate_nll(model: ChainTransformerMdn, chain_data: list[RepeaterChainData], batch_size: int, device: torch.device,
                 workers: int) -> float:
    """Evaluate sample-weighted negative log likelihood without updating the model.

    Args:
        model (ChainTransformerMdn): Model to evaluate.
        chain_data (list[RepeaterChainData]): Standardized chain data.
        batch_size (int): Conditions per batch.
        device (torch.device): Evaluation device.
        workers (int): Number of DataLoader worker processes.

    Returns:
        float: Sample-weighted mean negative log likelihood.
    """
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    loader = build_loader(chain_data, batch_size, False, device, workers, seed=0)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            count = int(batch[-1].sum().item())
            loss_sum += float(negative_log_likelihood(model, batch).item()) * count
            sample_count += count
    return loss_sum / sample_count


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


def build_model(args: argparse.Namespace, max_links: int) -> ChainTransformerMdn:
    """Build the Transformer-conditioned full-covariance MDN.

    Args:
        args (argparse.Namespace): Model architecture arguments.
        max_links (int): Maximum supported link count.

    Returns:
        ChainTransformerMdn: Untrained chain surrogate model.
    """
    return ChainTransformerMdn(components=args.components, embedding_dim=args.embedding_dim,
                               transformer_layers=args.transformer_layers, attention_heads=args.attention_heads,
                               feedforward_dim=args.feedforward_dim, dropout=args.dropout, hidden_size=args.hidden_size,
                               hidden_layers=args.hidden_layers, max_links=max_links)


def train_model(training: list[RepeaterChainData], validation: list[RepeaterChainData], normalization: Normalization,
                args: argparse.Namespace) -> tuple[MdnArtifact, dict[str, Any]]:
    """Train the chain surrogate and retain the best validation-NLL state.

    Args:
        training (list[RepeaterChainData]): Standardized training chain data.
        validation (list[RepeaterChainData]): Standardized validation chain data.
        normalization (Normalization): Training normalization statistics.
        args (argparse.Namespace): Training and architecture arguments.

    Returns:
        tuple[MdnArtifact, dict[str, Any]]: Model artifact and complete NLL history.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    max_links = max(data.link_distances.size for data in training + validation)
    model = build_model(args, max_links).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_reduction_factor,
                                                           patience=args.lr_patience, threshold=args.lr_threshold,
                                                           threshold_mode="abs", min_lr=args.min_learning_rate)
    loader = build_loader(training, args.batch_size, True, args.device, args.workers, args.seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_validation_nll = math.inf
    train_nll_history = []
    validation_nll_history = []
    learning_rate_history = []
    training_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = optimizer.param_groups[0]["lr"]
        loss_sum = 0.0
        sample_count = 0
        for batch in loader:
            batch = move_batch(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = negative_log_likelihood(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"encountered non-finite training loss at epoch {epoch}")
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            count = int(batch[-1].sum().item())
            loss_sum += float(loss.item()) * count
            sample_count += count

        train_nll = loss_sum / sample_count
        validation_nll = evaluate_nll(model, validation, args.batch_size, args.device, args.workers)
        train_nll_history.append(train_nll)
        validation_nll_history.append(validation_nll)
        learning_rate_history.append(learning_rate)
        if validation_nll < best_validation_nll:
            best_epoch = epoch
            best_validation_nll = validation_nll
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step(validation_nll)
        next_learning_rate = optimizer.param_groups[0]["lr"]

        if epoch == 1 or epoch == args.epochs or epoch % args.log_interval == 0:
            elapsed_time = time.perf_counter() - training_start_time
            estimated_runtime_left = elapsed_time / epoch * (args.epochs - epoch)
            print(f"Epoch {epoch:04d}: train_nll={train_nll:.6g}, validation_nll={validation_nll:.6g}, learning_rate={learning_rate:.3g}")
            print(f"Estimated runtime left: {timedelta(seconds=round(estimated_runtime_left))}")
        if next_learning_rate < learning_rate:
            print(f"Reduced learning rate from {learning_rate:.3g} to {next_learning_rate:.3g} at epoch {epoch}")

    model.load_state_dict(best_state)
    final_train_nll = evaluate_nll(model, training, args.batch_size, args.device, args.workers)
    final_validation_nll = evaluate_nll(model, validation, args.batch_size, args.device, args.workers)
    artifact = {"model_type": "transformer_full_covariance_mixture_density_network",
                "conditioning_method": CONDITIONING_METHOD,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "link_feature": LINK_DISTANCES_NAME,
                "global_feature_names": list(GLOBAL_FEATURE_NAMES),
                "target_names": list(TARGET_NAMES),
                "distance_mean": normalization["distance_mean"].tolist(),
                "distance_std": normalization["distance_std"].tolist(),
                "global_mean": normalization["global_mean"].tolist(),
                "global_std": normalization["global_std"].tolist(),
                "target_mean": normalization["target_mean"].tolist(),
                "target_std": normalization["target_std"].tolist(),
                "components": args.components,
                "embedding_dim": args.embedding_dim,
                "transformer_layers": args.transformer_layers,
                "attention_heads": args.attention_heads,
                "feedforward_dim": args.feedforward_dim,
                "dropout": args.dropout,
                "hidden_size": args.hidden_size,
                "hidden_layers": args.hidden_layers,
                "max_links": max_links,
                "target_transforms": ["log", "logit"],
                "dtype": "float32"}
    metrics = {"train_nll": train_nll_history,
               "validation_nll": validation_nll_history,
               "learning_rate": learning_rate_history,
               "best_epoch": best_epoch,
               "best_train_nll": final_train_nll,
               "best_validation_nll": final_validation_nll}
    return artifact, metrics


def save_model(model_path: Path, artifact: MdnArtifact, training: list[RepeaterChainData], validation: list[RepeaterChainData],
               metrics: dict[str, Any], args: argparse.Namespace):
    """Save the model artifact and human-readable training metadata.

    Args:
        model_path (Path): Destination for the PyTorch model artifact.
        artifact (MdnArtifact): Trained model state and normalization metadata.
        training (list[RepeaterChainData]): Training chain data.
        validation (list[RepeaterChainData]): Validation chain data.
        metrics (dict[str, Any]): Per-epoch and best-model metrics.
        args (argparse.Namespace): Training hyperparameters.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, model_path)
    metadata = {"model": "Globally conditioned Transformer full-covariance Mixture Density Network",
                "model_path": str(model_path),
                "conditioning_method": CONDITIONING_METHOD,
                "link_feature": LINK_DISTANCES_NAME,
                "global_feature_names": list(GLOBAL_FEATURE_NAMES),
                "targets": [TIME_PATH, FIDELITY_PATH],
                "target_transforms": ["log", "logit"],
                "dtype": "float32",
                "components": args.components,
                "embedding_dim": args.embedding_dim,
                "transformer_layers": args.transformer_layers,
                "attention_heads": args.attention_heads,
                "feedforward_dim": args.feedforward_dim,
                "dropout": args.dropout,
                "hidden_size": args.hidden_size,
                "hidden_layers": args.hidden_layers,
                "epochs": args.epochs,
                "batch_size_conditions": args.batch_size,
                "learning_rate": args.learning_rate,
                "lr_scheduler": "ReduceLROnPlateau",
                "lr_reduction_factor": args.lr_reduction_factor,
                "lr_patience": args.lr_patience,
                "lr_threshold": args.lr_threshold,
                "lr_threshold_mode": "abs",
                "min_learning_rate": args.min_learning_rate,
                "weight_decay": args.weight_decay,
                "gradient_clip": args.gradient_clip,
                "max_samples_per_file": args.max_samples_per_file,
                "training_condition_count": len(training),
                "validation_condition_count": len(validation),
                "training_sample_count": sum(data.targets.shape[0] for data in training),
                "validation_sample_count": sum(data.targets.shape[0] for data in validation),
                "training_directories": sorted({str(data.path.parent) for data in training}),
                "validation_directories": sorted({str(data.path.parent) for data in validation}),
                "metrics": metrics}
    model_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_model(model_path: Path, device: torch.device) -> MdnArtifact:
    """Load a saved chain surrogate artifact onto the requested device.

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


def parse_config(config_json: str | None, config_file: Path | None) -> dict[str, Any]:
    """Read one prediction configuration from JSON text or a file.

    Args:
        config_json (str | None): Inline JSON configuration.
        config_file (Path | None): JSON configuration file.

    Returns:
        dict[str, Any]: Parsed chain configuration.
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
    if not isinstance(parsed, dict):
        raise ValueError("prediction configuration must be a JSON object")
    return parsed


def build_model_from_artifact(artifact: MdnArtifact) -> ChainTransformerMdn:
    """Reconstruct a chain surrogate model from artifact architecture metadata.

    Args:
        artifact (MdnArtifact): Saved model artifact.

    Returns:
        ChainTransformerMdn: Reconstructed model.
    """
    conditioning_method = artifact.get("conditioning_method")
    if conditioning_method != CONDITIONING_METHOD:
        raise ValueError(f"model uses conditioning method {conditioning_method!r}; expected {CONDITIONING_METHOD!r}. "
                         "Retrain the model with the current code")
    return ChainTransformerMdn(components=int(artifact["components"]), embedding_dim=int(artifact["embedding_dim"]),
                               transformer_layers=int(artifact["transformer_layers"]),
                               attention_heads=int(artifact["attention_heads"]),
                               feedforward_dim=int(artifact["feedforward_dim"]), dropout=float(artifact["dropout"]),
                               hidden_size=int(artifact["hidden_size"]), hidden_layers=int(artifact["hidden_layers"]),
                               max_links=int(artifact["max_links"]))


def validate_condition(artifact: MdnArtifact, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate one chain condition and return its link distances and global features.

    Args:
        artifact (MdnArtifact): Trained model and normalization metadata.
        config (dict[str, Any]): Link distances and homogeneous chain features.

    Returns:
        tuple[np.ndarray, np.ndarray]: Link distances and globals ordered by GLOBAL_FEATURE_NAMES.
    """
    missing = [name for name in (LINK_DISTANCES_NAME, *GLOBAL_FEATURE_NAMES) if name not in config]
    if missing:
        raise ValueError(f"prediction config is missing required feature(s): {', '.join(missing)}")
    link_distances = np.asarray(config[LINK_DISTANCES_NAME], dtype=np.float32)
    if link_distances.ndim != 1 or link_distances.size == 0 or np.any(link_distances <= 0):
        raise ValueError(f"{LINK_DISTANCES_NAME} must be a nonempty list of positive values")
    if link_distances.size > int(artifact["max_links"]):
        raise ValueError(f"model supports at most {artifact['max_links']} links")
    global_features = np.asarray([config[name] for name in GLOBAL_FEATURE_NAMES], dtype=np.float32)
    if not np.all(np.isfinite(link_distances)) or not np.all(np.isfinite(global_features)):
        raise ValueError("prediction features must be finite")
    if "number_of_nodes" in config and int(config["number_of_nodes"]) != link_distances.size + 1:
        raise ValueError(f"number_of_nodes must equal len({LINK_DISTANCES_NAME}) + 1")
    return link_distances, global_features


def build_condition_sampler(artifact: MdnArtifact, device: torch.device) -> Callable[..., np.ndarray]:
    """Build the model once and return a sampler reusable across many chain conditions.

    Rebuilding the model per condition dominates the cost of a sweep over a whole testing set, so
    callers that score many conditions should build one sampler and call it repeatedly.

    Args:
        artifact (MdnArtifact): Trained model and normalization metadata.
        device (torch.device): PyTorch execution device.

    Returns:
        Callable[..., np.ndarray]: Callable taking (config, sample_count, seed) and returning rows
            of time in milliseconds and fidelity.
    """
    model = build_model_from_artifact(artifact).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    distance_mean = np.asarray(artifact["distance_mean"], dtype=np.float32)
    distance_std = np.asarray(artifact["distance_std"], dtype=np.float32)
    global_mean = np.asarray(artifact["global_mean"], dtype=np.float32)
    global_std = np.asarray(artifact["global_std"], dtype=np.float32)
    target_mean = np.asarray(artifact["target_mean"], dtype=np.float32)
    target_std = np.asarray(artifact["target_std"], dtype=np.float32)

    def sample(config: dict[str, Any], sample_count: int, seed: int) -> np.ndarray:
        """Generate paired time-fidelity samples for one repeater-chain condition."""
        link_distances, global_features = validate_condition(artifact, config)
        standardized_distances = (link_distances - distance_mean[0]) / distance_std[0]
        standardized_globals = (global_features - global_mean) / global_std

        torch.manual_seed(seed)
        with torch.no_grad():
            distance_tensor = torch.from_numpy(standardized_distances[None, :, None]).to(device)
            padding_mask = torch.zeros((1, link_distances.size), dtype=torch.bool, device=device)
            global_tensor = torch.from_numpy(standardized_globals[None]).to(device)
            distribution = mdn_distribution(model, distance_tensor, padding_mask, global_tensor)
            standardized_samples = distribution.sample((sample_count,))[:, 0].cpu().numpy()

        transformed_samples = standardized_samples * target_std + target_mean
        return inverse_transform_targets(transformed_samples)

    return sample


def sample_from_model(artifact: MdnArtifact, config: dict[str, Any], sample_count: int, seed: int,
                      device: torch.device) -> np.ndarray:
    """Generate paired time-fidelity samples for one repeater-chain condition.

    Args:
        artifact (MdnArtifact): Trained model and normalization metadata.
        config (dict[str, Any]): Link distances and homogeneous chain features.
        sample_count (int): Number of paired samples to generate.
        seed (int): PyTorch random seed.
        device (torch.device): PyTorch execution device.

    Returns:
        np.ndarray: Generated rows of time in milliseconds and fidelity.
    """
    return build_condition_sampler(artifact, device)(config, sample_count, seed)


def command_train(args: argparse.Namespace):
    """Load chain data, train the surrogate, and save its artifact.

    Args:
        args (argparse.Namespace): Parsed training arguments.
    """
    if args.embedding_dim % args.attention_heads != 0:
        raise ValueError("--embedding-dim must be divisible by --attention-heads")
    if args.max_samples_per_file is not None and args.max_samples_per_file < 1:
        raise ValueError("--max-samples-per-file must be positive")
    if not 0 < args.lr_reduction_factor < 1:
        raise ValueError("--lr-reduction-factor must be between 0 and 1")
    if args.lr_patience < 0:
        raise ValueError("--lr-patience cannot be negative")
    if args.lr_threshold < 0:
        raise ValueError("--lr-threshold cannot be negative")
    if not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("--min-learning-rate must be between 0 and --learning-rate")
    args.device = resolve_device(args.device)
    if args.device.type == "cuda":
        print(f"Using device: {args.device} ({torch.cuda.get_device_name(args.device)})")
    else:
        print(f"Using device: {args.device}")

    paths = iter_hdf5_paths(args.input)
    chain_data = load_chain_data(paths, args.max_samples_per_file, args.seed)
    training_raw, validation_raw = split_conditions(chain_data, args.validation_fraction, args.seed)
    normalization = calculate_normalization(training_raw)
    training = normalize_chain_data(training_raw, normalization)
    validation = normalize_chain_data(validation_raw, normalization)
    artifact, metrics = train_model(training, validation, normalization, args)
    model_path = args.output.with_name(f"{args.output.stem}_validation_nll={metrics['best_validation_nll']:.10f}{args.output.suffix}")
    save_model(model_path, artifact, training_raw, validation_raw, metrics, args)

    training_samples = sum(data.targets.shape[0] for data in training_raw)
    validation_samples = sum(data.targets.shape[0] for data in validation_raw)
    print(f"Used {len(paths)} HDF5 file(s)")
    print(f"Training conditions: {len(training_raw)}, paired samples: {training_samples}")
    print(f"Validation conditions: {len(validation_raw)}, paired samples: {validation_samples}")
    print(f"Global features: {', '.join(GLOBAL_FEATURE_NAMES)}")
    print(f"Targets: {TIME_PATH}, {FIDELITY_PATH}")
    print(f"Wrote model: {model_path}")
    print(f"Wrote metadata: {model_path.with_suffix('.json')}")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best-model NLL: train={metrics['best_train_nll']:.6g}, validation={metrics['best_validation_nll']:.6g}")


def command_predict(args: argparse.Namespace):
    """Load a chain surrogate and print generated time-fidelity samples.

    Args:
        args (argparse.Namespace): Parsed prediction arguments.
    """
    args.device = resolve_device(args.device)
    artifact = load_model(args.model, args.device)
    config = parse_config(args.config_json, args.config_file)
    samples = sample_from_model(artifact, config, args.sample_count, args.seed, args.device)
    print(json.dumps({"targets": list(TARGET_NAMES), "samples": samples.tolist()}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the training and prediction command-line interface.

    Returns:
        argparse.ArgumentParser: Configured top-level parser.
    """
    parser = argparse.ArgumentParser(description="Train or query a globally conditioned Transformer-MDN for repeater chains.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the repeater-chain surrogate.")
    train_parser.add_argument("--input", type=Path, nargs="+", default=[Path("chain/training/h5")],
                              help="HDF5 file(s) or directory tree(s). Defaults to chain/training/h5.")
    train_parser.add_argument("--output", type=Path, default=Path("models/chain_transformer_mdn.pt"))
    train_parser.add_argument("--components", type=int, default=12)
    train_parser.add_argument("--embedding-dim", type=int, default=64)
    train_parser.add_argument("--transformer-layers", type=int, default=2)
    train_parser.add_argument("--attention-heads", type=int, default=4)
    train_parser.add_argument("--feedforward-dim", type=int, default=128)
    train_parser.add_argument("--dropout", type=float, default=0.02)
    train_parser.add_argument("--hidden-size", type=int, default=64)
    train_parser.add_argument("--hidden-layers", type=int, default=2)
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--batch-size", type=int, default=32, help="Number of complete conditions per batch.")
    train_parser.add_argument("--learning-rate", type=float, default=0.0001)
    train_parser.add_argument("--lr-reduction-factor", type=float, default=0.5)
    train_parser.add_argument("--lr-patience", type=int, default=12)
    train_parser.add_argument("--lr-threshold", type=float, default=0.001)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=1e-5)
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--validation-fraction", type=float, default=0.2)
    train_parser.add_argument("--max-samples-per-file", type=int, default=2000)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--seed", type=int, default=7)
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--log-interval", type=int, default=1)
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser("predict", help="Generate paired time-fidelity samples.")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--config-json", help="JSON object containing link_distances and the eight global features.")
    predict_parser.add_argument("--config-file", type=Path, help="Path to a JSON chain configuration.")
    predict_parser.add_argument("--sample-count", type=int, default=5000)
    predict_parser.add_argument("--seed", type=int, default=7)
    predict_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    predict_parser.set_defaults(func=command_predict)
    return parser


def main():
    """Parse command-line arguments and run the selected command."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
