

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


TRUE_PI = math.pi
RACE_THREADS = (1, 2, 4, 8)
REDUCTION_THREADS = (1, 2, 4, 8, 16)
MODES = ("serial", "race", "critical", "reduction")


@dataclass
class Trial:
    mode: str
    threads: int
    trial: int
    pi: float
    elapsed_seconds: float
    error: float
    speedup: float = 0.0
    efficiency: float = 0.0
    lock_overhead_percent: float = 0.0


def integration_term(index: int, steps: int) -> float:
    step = 1.0 / steps
    x = (index + 0.5) * step
    return 4.0 / (1.0 + x * x)


def serial_pi(steps: int) -> float:
    total = sum(integration_term(index, steps) for index in range(steps))
    return total / steps


def bounds(steps: int, workers: int) -> list[tuple[int, int]]:
    base, remainder = divmod(steps, workers)
    result = []
    start = 0
    for worker in range(workers):
        size = base + (worker < remainder)
        result.append((start, start + size))
        start += size
    return result


def add_shared(start: int, end: int, steps: int, shared_sum, lock=None) -> None:
    for index in range(start, end):
        term = integration_term(index, steps)
        if lock is None:
            shared_sum.value += term
        else:
            with lock:
                shared_sum.value += term


def run_shared(steps: int, workers: int, critical: bool) -> tuple[float, float]:
    """Run the shared accumulator variant and include process join time."""
    shared_sum = mp.Value("d", 0.0, lock=False)
    lock = mp.Lock() if critical else None
    processes = []
    started = time.perf_counter()
    for start, end in bounds(steps, workers):
        process = mp.Process(target=add_shared, args=(start, end, steps, shared_sum, lock))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"worker process failed with exit code {process.exitcode}")
    elapsed = time.perf_counter() - started
    return shared_sum.value / steps, elapsed


def private_partial(bounds_and_steps: tuple[int, int, int]) -> float:
    start, end, steps = bounds_and_steps
    return sum(integration_term(index, steps) for index in range(start, end))


def run_reduction(steps: int, workers: int) -> tuple[float, float]:
    chunks = [(start, end, steps) for start, end in bounds(steps, workers)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        partials = list(executor.map(private_partial, chunks, chunksize=1))
    elapsed = time.perf_counter() - started
    return sum(partials) / steps, elapsed


def timed_trials(mode: str, steps: int, workers: int, repeats: int) -> list[Trial]:
    if mode == "serial":
        trials = []
        for trial_number in range(1, repeats + 1):
            started = time.perf_counter()
            pi = serial_pi(steps)
            elapsed = time.perf_counter() - started
            trials.append(Trial(mode, 1, trial_number, pi, elapsed, abs(pi - TRUE_PI)))
        return trials

    runner = {
        "race": lambda: run_shared(steps, workers, critical=False),
        "critical": lambda: run_shared(steps, workers, critical=True),
        "reduction": lambda: run_reduction(steps, workers),
    }[mode]
    trials = []
    for trial_number in range(1, repeats + 1):
        pi, elapsed = runner()
        trials.append(Trial(mode, workers, trial_number, pi, elapsed, abs(pi - TRUE_PI)))
    return trials


def write_csv(path: Path, trials: list[Trial]) -> None:
    fields = [
        "mode", "threads", "trial", "pi", "elapsed_seconds", "absolute_error",
        "speedup", "efficiency", "lock_overhead_percent",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({
                "mode": trial.mode,
                "threads": trial.threads,
                "trial": trial.trial,
                "pi": f"{trial.pi:.12f}",
                "elapsed_seconds": f"{trial.elapsed_seconds:.9f}",
                "absolute_error": f"{trial.error:.12e}",
                "speedup": f"{trial.speedup:.6f}",
                "efficiency": f"{trial.efficiency:.6f}",
                "lock_overhead_percent": f"{trial.lock_overhead_percent:.3f}",
            })


def create_plot(trials: list[Trial], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping lab2_speedup.png")
        return

    averages = {}
    for threads in REDUCTION_THREADS:
        values = [
            trial.elapsed_seconds
            for trial in trials
            if trial.mode == "reduction" and trial.threads == threads
        ]
        if values:
            averages[threads] = sum(values) / len(values)
    baseline = averages[1]
    thread_values = list(averages)
    speedups = [baseline / averages[threads] for threads in thread_values]
    plt.figure(figsize=(8, 5), dpi=180)
    plt.plot(thread_values, speedups, "o-", label="Measured reduction speedup")
    plt.plot(thread_values, thread_values, "--", label="Linear ideal speedup")
    plt.xlabel("Threads (P)")
    plt.ylabel("Speedup vs. P=1")
    plt.title("Lab 2: Reduction Strong Scaling")
    plt.xticks(thread_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def print_summary(trials: list[Trial], steps: int) -> None:
    print(f"N={steps:,}; true pi={TRUE_PI:.12f}")
    for mode in MODES:
        selected = [trial for trial in trials if trial.mode == mode]
        if not selected:
            continue
        print(f"\n{mode.title()} variant")
        for threads in sorted({trial.threads for trial in selected}):
            values = [trial for trial in selected if trial.threads == threads]
            average_time = sum(trial.elapsed_seconds for trial in values) / len(values)
            average_pi = sum(trial.pi for trial in values) / len(values)
            print(f"P={threads:>2}: pi={average_pi:.12f}, average={average_time:.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", default="lab2_results.csv")
    parser.add_argument("--plot", default="lab2_speedup.png")
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        raise SystemExit("--steps and --repeats must be positive")

    print("Running true serial baseline...")
    serial_trials = timed_trials("serial", args.steps, 1, args.repeats)
    all_trials = list(serial_trials)

    print("Running race quantification for P=1, 2, 4, 8...")
    for threads in RACE_THREADS:
        all_trials.extend(timed_trials("race", args.steps, threads, args.repeats))

    critical_steps = min(args.steps, 1_000_000)
    print(f"Running critical-section benchmark with N={critical_steps:,}...")
    critical_trials = timed_trials("critical", critical_steps, 4, args.repeats)
    critical_baseline = timed_trials("serial", critical_steps, 1, args.repeats)
    all_trials.extend(critical_trials)

    print("Running reduction strong scaling for P=1, 2, 4, 8, 16...")
    reduction_trials = []
    for threads in REDUCTION_THREADS:
        reduction_trials.extend(timed_trials("reduction", args.steps, threads, args.repeats))
    all_trials.extend(reduction_trials)

    baseline_time = (
        sum(trial.elapsed_seconds for trial in reduction_trials if trial.threads == 1)
        / len([trial for trial in reduction_trials if trial.threads == 1])
    )
    for trial in all_trials:
        if trial.mode == "reduction" and trial.threads in REDUCTION_THREADS:
            trial.speedup = baseline_time / trial.elapsed_seconds
            trial.efficiency = trial.speedup / trial.threads
    lock_baseline = sum(trial.elapsed_seconds for trial in critical_baseline) / len(critical_baseline)
    critical_avg = sum(trial.elapsed_seconds for trial in critical_trials) / len(critical_trials)
    for trial in critical_trials:
        trial.lock_overhead_percent = (critical_avg / lock_baseline - 1.0) * 100.0

    output = Path(args.output)
    write_csv(output, all_trials)
    create_plot(all_trials, Path(args.plot))
    print_summary(all_trials, args.steps)
    print(f"Wrote {output} and {args.plot}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
