"""Run main_link.py / main_chain.py over a grid of simulation parameters."""

import math
import numpy as np
import itertools
import time
from datetime import timedelta
from subprocess import PIPE, Popen


def build_link_task(command: list[str], parameter_grid: list[tuple[float, ...]], log_dir: str) -> list[list[str]]:
    """Build one link simulation command for each parameter combination."""
    tasks = []
    for values in parameter_grid:
        link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node = values
        task = command + ["-ld", str(link_distance), "-me", str(memory_efficiency), "-de", str(detector_efficiency),
                          "-a", str(attenuation), "-cdn", str(classical_delay_node), "-d", log_dir]
        tasks.append(task)
    return tasks


def build_chain_task(command: list[str], parameter_grid: list[tuple[float, ...]], log_dir: str) -> list[list[str]]:
    """Build one chain simulation command for each parameter combination."""
    tasks = []
    for values in parameter_grid:
        (number_of_nodes, link_distance, memory_efficiency, detector_efficiency, attenuation,
         classical_delay_node, swapping_probability, initial_fidelity, coherence_time, classical_delay_hop) = values
        task = command + ["-n", str(number_of_nodes), "-ld", str(link_distance), "-me", str(memory_efficiency),
                          "-de", str(detector_efficiency), "-a", str(attenuation), "-cdn", str(classical_delay_node),
                          "-sp", str(swapping_probability), "-if", str(initial_fidelity), "-ct", str(coherence_time),
                          "-cdh", str(classical_delay_hop), "-d", log_dir]
        tasks.append(task)
    return tasks


def balanced_level_indices(number_of_levels: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `size` level indices so that every level is used equally often (+/- 1)."""
    repeats = math.ceil(size / number_of_levels)
    indices = np.tile(np.arange(number_of_levels), repeats)
    rng.shuffle(indices)
    return indices[:size]


def sample_parameter_grid(parameter_values: list[list], fraction: float, seed: int = 0) -> list[tuple]:
    """Latin-hypercube subsample of the full Cartesian product of `parameter_values`.

    Each parameter column is filled with its own levels in equal proportion and then
    shuffled independently, so every level of every parameter appears the same number
    of times while no two parameters stay correlated. Duplicate combinations are
    dropped and resampled, so the design contains `fraction` of the grid with no
    repeats. Sampling is deterministic for a given `seed`.
    """
    total = math.prod(len(values) for values in parameter_values)
    size = round(total * fraction)
    if not 0 < size <= total:
        raise ValueError(f"fraction={fraction} selects {size} of {total} combination(s)")

    rng = np.random.default_rng(seed)
    seen = set()
    grid = []
    while len(grid) < size:
        missing = size - len(grid)
        columns = [balanced_level_indices(len(values), missing, rng) for values in parameter_values]
        for level_indices in zip(*columns):
            combination = tuple(values[index] for values, index in zip(parameter_values, level_indices))
            if combination not in seen:
                seen.add(combination)
                grid.append(combination)
    print(f"Sampled {len(grid)} of {total} combination(s)")
    return grid


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


def main_chain_training():
    """Define the parameter grid and run all chain simulations."""
    command = ["python", "main_chain.py"]
                                                                       # 367416 in total, 11022 sampled
    number_of_nodes = [3, 5, 6, 9, 12, 15, 17]                         # 7 values
    swapping_probability = np.linspace(0.8, 1.0, 3).round(2).tolist()  # 3 values
    initial_fidelity = np.linspace(0.9, 1.0, 3).round(2).tolist()      # 3 values
    coherence_time = np.linspace(0.5, 3.0, 6).round(2).tolist()        # 6 values
    classical_delay_hop = [0.02, 0.03, 0.04]                           # 3 values
    
    link_distance = list(range(1000, 20001, 5000))                     # 4 values
    memory_efficiency = np.linspace(0.7, 1.0, 3).round(2).tolist()     # 3 values
    detector_efficiency = np.linspace(0.94, 1, 3).round(2).tolist()    # 3 values
    attenuation = np.linspace(0.0002, 0.0004, 3).round(5).tolist()     # 3 values
    classical_delay_node = np.linspace(0.06, 0.1, 3).round(3).tolist() # 3 values
    
    log_dir = "chain/training"
    runtime_log = "runtime_chain.log"
    parallel = 45
    fraction = 0.03
    seed = 0

    parameter_values = [number_of_nodes, link_distance, memory_efficiency, detector_efficiency,
                        attenuation, classical_delay_node, swapping_probability, initial_fidelity,
                        coherence_time, classical_delay_hop]
    parameter_grid = sample_parameter_grid(parameter_values, fraction, seed)
    tasks = build_chain_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


def main_chain_testing():
    """Define the parameter grid and run all chain simulations."""
    command = ["python", "main_chain.py"]
                                                                           # 256 in total
    number_of_nodes = [4, 5, 8, 10, 11, 13, 14, 16]                        # 8 values
    swapping_probability = np.linspace(0.85, 0.95, 2).round(2).tolist()    # 2 values
    initial_fidelity = [0.94, 0.98]                                        # 2 values
    coherence_time = [1.2, 2.4]                                            # 2 values
    classical_delay_hop = [0.025]                                          # 1 value

    link_distance = [3500, 13500]                                          # 2 values
    memory_efficiency = np.linspace(0.77, 0.87, 2).round(2).tolist()       # 2 values
    detector_efficiency = [0.88]                                           # 1 value
    attenuation = [0.00025]                                                # 1 value
    classical_delay_node = [0.07]                                          # 1 value

    log_dir = "chain/testing"
    runtime_log = "runtime_chain.log"
    parallel = 40

    parameter_grid = list(itertools.product(number_of_nodes, link_distance, memory_efficiency, detector_efficiency,
                                            attenuation, classical_delay_node, swapping_probability, initial_fidelity,
                                            coherence_time, classical_delay_hop))
    tasks = build_chain_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


if __name__ == "__main__":
    # main_link_training()
    # main_link_testing()
    main_chain_training()
    main_chain_testing()
