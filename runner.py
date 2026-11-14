"""Run main_link.py and main_chain.py over simulation parameter configurations."""

import numpy as np
import itertools
import time
from datetime import timedelta
from pathlib import Path
from subprocess import PIPE, Popen


CHAIN_PARAMETER_RANGES = {"number_of_nodes": (3, 17),
                          "link_distance": (1000, 20_000),
                          "memory_efficiency": (0.7, 1.0),
                          "detector_efficiency": (0.94, 1.0),
                          "attenuation": (0.0002, 0.0004),
                          "classical_delay_node": (0.06, 0.1),
                          "swapping_probability": (0.8, 1.0),
                          "initial_fidelity": (0.95, 1.0),
                          "coherence_time": (0.5, 3.0),
                          "classical_delay_hop": (0.02, 0.04)}


def build_link_task(command: list[str], parameter_grid: list[tuple[float, ...]], log_dir: str) -> list[list[str]]:
    """Build one link simulation command for each parameter combination."""
    tasks = []
    for values in parameter_grid:
        link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node = values
        task = command + ["-ld", str(link_distance), "-me", str(memory_efficiency), "-de", str(detector_efficiency),
                          "-a", str(attenuation), "-cdn", str(classical_delay_node), "-d", log_dir]
        tasks.append(task)
    return tasks


def build_chain_task(command: list[str], configurations: list[tuple], log_dir: str) -> list[list[str]]:
    """Build one chain simulation command for each parameter combination."""
    tasks = []
    for values in configurations:
        (link_distances, memory_efficiency, detector_efficiency, attenuation, classical_delay_node,
         swapping_probability, initial_fidelity, coherence_time, classical_delay_hop) = values
        task = command + ["-lds"] + [str(distance) for distance in link_distances]
        task += ["-me", str(memory_efficiency), "-de", str(detector_efficiency), "-a", str(attenuation),
                 "-cdn", str(classical_delay_node), "-sp", str(swapping_probability), "-if", str(initial_fidelity),
                 "-ct", str(coherence_time), "-cdh", str(classical_delay_hop), "-d", log_dir]
        tasks.append(task)
    return tasks


def sample_decimal(rng: np.random.Generator, value_range: tuple[float, float], decimal_places: int) -> float:
    """Sample uniformly from the decimal values in an inclusive range.

    Args:
        rng (np.random.Generator): NumPy random-number generator.
        value_range (tuple[float, float]): Inclusive minimum and maximum values.
        decimal_places (int): Number of digits retained after the decimal point.

    Returns:
        float: Randomly selected value at the requested precision.
    """
    scale = 10 ** decimal_places
    minimum = round(value_range[0] * scale)
    maximum = round(value_range[1] * scale)
    return float(rng.integers(minimum, maximum + 1) / scale)


def sample_chain_configurations(total_configurations: int, parameter_ranges: dict[str, tuple[float, float]],
                                seed: int) -> list[tuple]:
    """Randomly sample unique repeater-chain configurations from parameter ranges.

    Args:
        total_configurations (int): Number of unique configurations to generate.
        parameter_ranges (dict[str, tuple[float, float]]): Inclusive range for each chain parameter.
        seed (int): NumPy random seed.

    Returns:
        list[tuple]: Randomly sampled chain configurations.
    """
    if total_configurations < 1:
        raise ValueError("total_configurations must be positive")
    for name, value_range in parameter_ranges.items():
        if value_range[0] > value_range[1]:
            raise ValueError(f"invalid range for {name}: {value_range}")

    rng = np.random.default_rng(seed)
    seen = set()
    configurations = []
    while len(configurations) < total_configurations:
        nodes = int(rng.integers(int(parameter_ranges["number_of_nodes"][0]), int(parameter_ranges["number_of_nodes"][1]) + 1))
        link_distances = tuple(rng.integers(int(parameter_ranges["link_distance"][0]), int(parameter_ranges["link_distance"][1]) + 1,
                                            size=nodes - 1).tolist())
        configuration = (link_distances, 
                         sample_decimal(rng, parameter_ranges["memory_efficiency"], 2),
                         sample_decimal(rng, parameter_ranges["detector_efficiency"], 2),
                         sample_decimal(rng, parameter_ranges["attenuation"], 5),
                         sample_decimal(rng, parameter_ranges["classical_delay_node"], 3),
                         sample_decimal(rng, parameter_ranges["swapping_probability"], 2),
                         sample_decimal(rng, parameter_ranges["initial_fidelity"], 2),
                         sample_decimal(rng, parameter_ranges["coherence_time"], 2),
                         sample_decimal(rng, parameter_ranges["classical_delay_hop"], 3))
        if configuration not in seen:
            seen.add(configuration)
            configurations.append(configuration)
    return configurations


def get_output(p: Popen):
    """Print stderr and stdout from a completed process."""
    stderr = p.stderr.readlines()
    if stderr:
        for line in stderr:
            print(line)

    stdout = p.stdout.readlines()
    if stdout:
        for line in stdout:
            print(line)


def run_tasks(tasks: list[list[str]], parallel: int, runtime_log: str):
    """Run simulation commands with a fixed parallel-process limit."""
    pending = list(tasks)
    running: list[tuple[Popen, list[str], float]] = []
    start_time = time.perf_counter()
    runtime_file = open(runtime_log, "w")

    while pending or running:
        while pending and len(running) < parallel:
            task = pending.pop(0)
            task_start_time = time.perf_counter()
            process = Popen(task, stdout=PIPE, stderr=PIPE)
            running.append((process, task, task_start_time))
            print(f"Running {task}, {len(pending)} tasks left")

        finished = []
        for process, task, task_start_time in running:
            if process.poll() is None:
                continue
            finished.append((process, task, task_start_time))

        for process, task, task_start_time in finished:
            task_runtime = time.perf_counter() - task_start_time
            get_output(process)
            runtime_file.write(f"runtime={task_runtime:.3f} seconds, return_code={process.returncode}, task={task}\n")
            runtime_file.flush()
            running.remove((process, task, task_start_time))

        tasks_left = len(pending) + len(running)
        finished_count = len(tasks) - tasks_left
        if finished and tasks_left and finished_count >= parallel:
            elapsed_time = time.perf_counter() - start_time
            estimated_runtime_left = elapsed_time / finished_count * tasks_left
            print(f"Estimated runtime left: {timedelta(seconds=round(estimated_runtime_left))}")

        time.sleep(0.02)

    runtime_file.close()
    elapsed_time = time.perf_counter() - start_time
    print(f"Finished {len(tasks)} tasks in {elapsed_time:.2f} seconds")


def main_link_training():
    """Define the parameter grid and run all link simulations."""
    command = ["python", "main_link.py"]                               # 5400 in total
    link_distance = list(range(1000, 20001, 2000))                     # 10 values
    memory_efficiency = np.linspace(0.5, 1.0, 6).round(2).tolist()     # 6 values
    detector_efficiency = np.linspace(0.9, 1, 6).round(2).tolist()     # 6 values
    attenuation = np.linspace(0.0002, 0.0004, 3).round(5).tolist()     # 3 values
    classical_delay_node = np.linspace(0.02, 0.1, 5).round(3).tolist() # 5 values
    log_dir = "link/7.20.2026"
    runtime_log = "runtime.log"
    parallel = 7

    parameter_grid = list(itertools.product(link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node))
    tasks = build_link_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


def main_link_testing():
    """Define the parameter grid and run all link simulations."""
    command = ["python", "main_link.py"]                                  # 80 in total
    link_distance = list(range(2000, 20001, 4000))                        # 5 values
    memory_efficiency = np.linspace(0.55, 0.67, 2).round(2).tolist()      # 2 values
    detector_efficiency = np.linspace(0.91, 0.944, 2).round(2).tolist()   # 2 values
    attenuation = np.linspace(0.00025, 0.00033, 2).round(5).tolist()      # 2 values
    classical_delay_node = np.linspace(0.025, 0.077, 2).round(3).tolist() # 2 values
    log_dir = "link/testing"
    runtime_log = "runtime.log"
    parallel = 24

    parameter_grid = list(itertools.product(link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node))
    tasks = build_link_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


def main_chain(total_configurations: int, directory: str, seed: int):
    """Randomly sample repeater-chain configurations and run their simulations.

    Args:
        total_configurations (int): Number of unique configurations to simulate.
        directory (str): Directory for simulation data and the runtime log.
        seed (int): NumPy random seed.
    """
    command = ["python", "main_chain.py"]
    parallel = 45
    Path(directory).mkdir(parents=True, exist_ok=True)
    runtime_log = str(Path(directory) / "runtime.log")

    configurations = sample_chain_configurations(total_configurations, CHAIN_PARAMETER_RANGES, seed)
    tasks = build_chain_task(command, configurations, directory)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


if __name__ == "__main__":
    # main_link_training()
    # main_link_testing()
    main_chain(10_000, "chain/training", seed=0)
    main_chain(250, "chain/testing", seed=1)
