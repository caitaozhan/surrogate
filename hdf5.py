"""Convert linear simulation logs into HDF5 surrogate datasets.

The current logs contain scalar final-state fidelity and time-to-serve values,
so the generated HDF5 file uses the qnet ``log_format=fidelity`` convention.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPEED_OF_LIGHT_M_PER_PS = 2e-4
MICROSECOND_PS = 10**6
MILLISECOND_PS = 10**9
SPEED_OF_LIGHT_M_PER_MS = SPEED_OF_LIGHT_M_PER_PS * MILLISECOND_PS
MICROSECOND_MS = MICROSECOND_PS / MILLISECOND_PS

FIXED_PARAMETERS = {
    "link_distance": 5000.0,
    "memory_efficiency": 0.6,
    "optical_bsm_success_prob": 0.5,
    "detector_efficiency": 0.95,
    "attenuation": 0.0002,
    "speed_of_light_m_per_ms": SPEED_OF_LIGHT_M_PER_MS,
    "classical_delay_per_hop_ms": 20 * MICROSECOND_MS,
    "classical_delay_fixed_overhead_ms": 100 * MICROSECOND_MS,
}
FIXED_USER_OPTIONAL_TEXT = {
    "link type": "Meet in the middle",
    "entanglement gernation protocol": "Single heralded entanglement generation",
    "formalism": "Bell diagonal state",
}

LOG_PATTERN = re.compile(
    r"reservation=\|identity=(?P<identity>\d+); "
    r"initiator=(?P<initiator>[^;]+); "
    r"responder=(?P<responder>[^;]+); "
    r"path=(?P<path>\[.*?\]); "
    r"start_time=(?P<start_time>[\d,]+); "
    r"end_time=(?P<end_time>[\d,]+); "
    r"target_fidelity=(?P<target_fidelity>[-+]?\d*\.?\d+); "
    r"entanglement_number=(?P<entanglement_number>\d+); "
    r"memory_size=(?P<memory_size>\d+)\|, "
    r"time to serve=(?P<time_to_serve>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?), "
    r"fidelity=(?P<fidelity>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
FILENAME_PATTERN = re.compile(
    r"^linear_(?P<nodes>\d+)(?:_swap_prob=(?P<swap_prob>\d+(?:\.\d+)?))?\.log$"
)


@dataclass(frozen=True)
class ParsedLog:
    path: Path
    run_id: str
    node_count: int
    swapping_success_prob: float
    sample_id: np.ndarray
    reservation_start_time: np.ndarray
    reservation_end_time: np.ndarray
    time_to_serve: np.ndarray
    fidelity: np.ndarray


def _parse_int_with_commas(value: str) -> int:
    return int(value.replace(",", ""))


def _run_id(path: Path, node_count: int, swap_prob: float) -> str:
    if math.isnan(swap_prob):
        return f"linear_{node_count}_swap_prob_unknown"
    return f"linear_{node_count}_swap_prob_{swap_prob:g}".replace(".", "_")


def _swap_prob_description(swap_prob: float) -> str:
    if math.isnan(swap_prob):
        return "unspecified"
    return f"{swap_prob:g}"


def parse_linear_log(path: Path) -> ParsedLog:
    """Parse one ``linear_*`` log into numeric arrays."""
    filename_match = FILENAME_PATTERN.match(path.name)
    if not filename_match:
        raise ValueError(f"not a supported linear log name: {path.name}")

    node_count = int(filename_match.group("nodes"))
    swap_prob_text = filename_match.group("swap_prob")
    swapping_success_prob = float(swap_prob_text) if swap_prob_text else math.nan

    sample_id: list[int] = []
    reservation_start_time: list[int] = []
    reservation_end_time: list[int] = []
    time_to_serve: list[float] = []
    fidelity: list[float] = []

    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = LOG_PATTERN.search(line)
            if not match:
                continue

            sample_id.append(int(match.group("identity")))
            reservation_start_time.append(_parse_int_with_commas(match.group("start_time")))
            reservation_end_time.append(_parse_int_with_commas(match.group("end_time")))
            time_to_serve.append(float(match.group("time_to_serve")))
            fidelity.append(float(match.group("fidelity")))

    if not sample_id:
        raise ValueError(f"no final fidelity records found in {path}")

    return ParsedLog(
        path=path,
        run_id=_run_id(path, node_count, swapping_success_prob),
        node_count=node_count,
        swapping_success_prob=swapping_success_prob,
        sample_id=np.asarray(sample_id, dtype=np.uint64),
        reservation_start_time=np.asarray(reservation_start_time, dtype=np.uint64),
        reservation_end_time=np.asarray(reservation_end_time, dtype=np.uint64),
        time_to_serve=np.asarray(time_to_serve, dtype=np.float64),
        fidelity=np.asarray(fidelity, dtype=np.float64),
    )


def iter_linear_logs(log_dir: Path) -> list[Path]:
    """Return parseable linear log paths, ignoring notebooks and bottleneck logs."""
    return sorted(
        (path for path in log_dir.glob("linear_*.log") if FILENAME_PATTERN.match(path.name)),
        key=lambda path: (
            int(FILENAME_PATTERN.match(path.name).group("nodes")),  # type: ignore[union-attr]
            float(FILENAME_PATTERN.match(path.name).group("swap_prob") or -1),  # type: ignore[union-attr]
        ),
    )


def _enum_dtypes() -> tuple[np.dtype, np.dtype, np.dtype]:
    reference_state_dt = h5py.enum_dtype({"bell_pair": 0}, basetype="i1")
    log_format_dt = h5py.enum_dtype(
        {
            "fidelity": 0,
            "state_vector": 1,
            "werner_vector": 2,
            "density_matrix": 3,
            "pauli_observables": 4,
        },
        basetype="i1",
    )
    simulation_mode_dt = h5py.enum_dtype(
        {
            "repeated_single_shot": 0,
            "stateful": 1,
        },
        basetype="i1",
    )
    return reference_state_dt, log_format_dt, simulation_mode_dt


def write_one_hdf5(parsed: ParsedLog, output_path: Path) -> Path:
    """Write one parsed linear log using the requested qnet HDF5 schema."""
    reference_state_dt, log_format_dt, simulation_mode_dt = _enum_dtypes()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    description = (
        "Simulation of a linear topology quantum network with "
        f"{parsed.node_count} nodes and "
        f"{_swap_prob_description(parsed.swapping_success_prob)} swapping probability"
    )

    with h5py.File(output_path, "w") as h5_file:
        qnet = h5_file.create_group("qnet")
        qnet.attrs["format_version"] = np.uint64(1)
        qnet.attrs["format_version_minor"] = np.uint64(1)
        qnet.attrs.create("log_format", np.array(0, dtype=log_format_dt), dtype=log_format_dt)
        qnet.attrs.create("reference_state", np.array(0, dtype=reference_state_dt), dtype=reference_state_dt)
        qnet.attrs.create("simulation_mode", np.array(0, dtype=simulation_mode_dt), dtype=simulation_mode_dt)

        metadata = qnet.create_group("metadata")
        metadata.attrs["description"] = description
        metadata.attrs["simulator"] = "SeQUeNCe"

        sequence_metadata = metadata.create_group("SeQUeNCe")
        sequence_metadata.attrs["SeQUeNCe_format_version"] = np.uint64(1)

        user_optional = metadata.create_group("user_optional")
        user_optional.attrs["node_count"] = np.uint64(parsed.node_count)
        if math.isfinite(parsed.swapping_success_prob):
            user_optional.attrs["swapping_success_prob"] = parsed.swapping_success_prob
        for name, value in FIXED_PARAMETERS.items():
            user_optional.attrs[name] = value
        for name, value in FIXED_USER_OPTIONAL_TEXT.items():
            user_optional.attrs[name] = value

        simulation_log = qnet.create_group("simulation_log")
        simulation_log.create_dataset("time", data=parsed.time_to_serve, compression="gzip")
        simulation_log.create_dataset("state", data=parsed.fidelity, compression="gzip")

    return output_path


def write_hdf5_files(log_dir: Path, output_dir: Path) -> list[Path]:
    """Convert all supported linear logs into one HDF5 file per log."""
    log_paths = iter_linear_logs(log_dir)
    if not log_paths:
        raise FileNotFoundError(f"no linear_*.log files found under {log_dir}")

    parsed_logs = [parse_linear_log(path) for path in log_paths]
    return [write_one_hdf5(parsed, output_dir / f"{parsed.path.stem}.h5") for parsed in parsed_logs]


def plot_hdf5_file(hdf5_path: Path, plot_path: Path) -> Path:
    """Create side-by-side histograms for time-to-serve and fidelity."""
    with h5py.File(hdf5_path, "r") as h5_file:
        time_to_serve = h5_file["qnet/simulation_log/time"][:]
        fidelity = h5_file["qnet/simulation_log/state"][:]
        description = h5_file["qnet/metadata"].attrs["description"]

    mean_time_to_serve = float(np.mean(time_to_serve))
    mean_fidelity = float(np.mean(fidelity))

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].hist(time_to_serve, bins=50, color='blue', edgecolor="white", alpha=0.9)
    axes[0].axvline(mean_time_to_serve, color='red', linewidth=2, label=f"mean = {mean_time_to_serve:.3f} ms")
    axes[0].set_title("Time to Serve")
    axes[0].set_xlabel("Time to serve (ms)")
    axes[0].set_ylabel("Samples")
    axes[0].legend()

    axes[1].hist(fidelity, bins=50, color='brown', edgecolor="white", alpha=0.9)
    axes[1].axvline(mean_fidelity, color='darkgreen', linewidth=2, label=f"mean = {mean_fidelity:.6f}")
    axes[1].set_title("Fidelity")
    axes[1].set_xlabel("Fidelity")
    axes[1].set_ylabel("Samples")
    axes[1].legend()

    fig.suptitle(str(description), fontsize=11)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def plot_hdf5_files(hdf5_paths: list[Path], plot_dir: Path) -> list[Path]:
    """Create one histogram figure per HDF5 file."""
    return [plot_hdf5_file(path, plot_dir / f"{path.stem}.png") for path in hdf5_paths]


def main():
    parser = argparse.ArgumentParser(description="Convert linear_*.log files into HDF5.")
    parser.add_argument("--log-dir", type=Path, default=Path("log"), help="Directory containing linear_*.log files.")
    parser.add_argument("--output-dir", type=Path, default=Path("hdf5/sequence_linear"), help="Directory for one HDF5 file per linear log.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory for one PNG histogram figure per HDF5 file. Defaults to <output-dir>/plots.")
    args = parser.parse_args()

    output_paths = write_hdf5_files(args.log_dir, args.output_dir)
    print(f"Wrote {len(output_paths)} HDF5 files under {args.output_dir}")

    plot_dir = args.plot_dir or args.output_dir / "plots"
    plot_paths = plot_hdf5_files(output_paths, plot_dir)
    print(f"Wrote {len(plot_paths)} plot files under {plot_dir}")


if __name__ == "__main__":
    main()
