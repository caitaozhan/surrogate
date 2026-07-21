"""Run main_link.py over a grid of link parameters."""

import numpy as np
import itertools
import time
from datetime import timedelta
from subprocess import PIPE, Popen


def build_task(command: list[str], parameter_grid: list[tuple[float, ...]], log_dir: str) -> list[list[str]]:
    """Build one simulation command for each parameter combination."""
    tasks = []
    for values in parameter_grid:
        link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node = values
        task = command + ["-ld", str(link_distance), "-me", str(memory_efficiency), "-de", str(detector_efficiency), 
                          "-a", str(attenuation), "-cdn", str(classical_delay_node), "-d", log_dir]
        tasks.append(task)
    return tasks


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


def main():
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
    tasks = build_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


def main_testing():
    """Define the parameter grid and run all link simulations."""
    command = ["python", "main_link.py"]                                  # 80 in total
    link_distance = list(range(2000, 20001, 4000))                        # 5 values
    memory_efficiency = np.linspace(0.55, 0.67, 2).round(2).tolist()      # 2 values
    detector_efficiency = np.linspace(0.91, 0.944, 2).round(2).tolist()   # 2 values
    attenuation = np.linspace(0.00025, 0.00033, 2).round(5).tolist()      # 2 values
    classical_delay_node = np.linspace(0.025, 0.077, 2).round(3).tolist() # 2 values
    log_dir = "link/testing"
    runtime_log = "runtime.log"
    parallel = 7

    parameter_grid = list(itertools.product(link_distance, memory_efficiency, detector_efficiency, attenuation, classical_delay_node))
    tasks = build_task(command, parameter_grid, log_dir)
    print(f"Generated {len(tasks)} simulation task(s)")
    run_tasks(tasks, parallel, runtime_log)


if __name__ == "__main__":
    # main()
    main_testing()
