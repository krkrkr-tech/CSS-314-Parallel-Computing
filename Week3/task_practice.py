from __future__ import annotations

import argparse
import random
import threading
import time
from dataclasses import dataclass


DEFAULT_PART1_ITERATIONS = 50_000_000
DEFAULT_PART3_ITERATIONS = 100_000_000
THREAD_COUNTS = (1, 2, 4, 8, 16, 32)


def inside_quarter_circle(rng: random.Random) -> bool:
	x = rng.random()
	y = rng.random()
	return x * x + y * y <= 1.0


def split_iterations(iterations: int, workers: int) -> list[int]:
	base, remainder = divmod(iterations, workers)
	return [base + (index < remainder) for index in range(workers)]


def run_phantom_bug(iterations: int, workers: int, seed: int) -> tuple[float, float]:
	total_hits = [0]
	counts = split_iterations(iterations, workers)

	def worker(worker_id: int, count: int) -> None:
		rng = random.Random(seed + worker_id)
		for _ in range(count):
			if inside_quarter_circle(rng):
				total_hits[0] += 1

	started = time.perf_counter()
	threads = [
		threading.Thread(target=worker, args=(worker_id, count))
		for worker_id, count in enumerate(counts)
	]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	elapsed_ms = (time.perf_counter() - started) * 1000
	return 4 * total_hits[0] / iterations, elapsed_ms


def run_synchronized(iterations: int, workers: int, seed: int) -> tuple[float, float]:
	total_hits = [0]
	counter_lock = threading.Lock()
	counts = split_iterations(iterations, workers)

	def worker(worker_id: int, count: int) -> None:
		rng = random.Random(seed + worker_id)
		for _ in range(count):
			if inside_quarter_circle(rng):
				with counter_lock:
					total_hits[0] += 1

	started = time.perf_counter()
	threads = [
		threading.Thread(target=worker, args=(worker_id, count))
		for worker_id, count in enumerate(counts)
	]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	elapsed_ms = (time.perf_counter() - started) * 1000
	return 4 * total_hits[0] / iterations, elapsed_ms


def run_single_thread(iterations: int, seed: int) -> tuple[float, float]:
	rng = random.Random(seed)
	started = time.perf_counter()
	total_hits = sum(inside_quarter_circle(rng) for _ in range(iterations))
	elapsed_ms = (time.perf_counter() - started) * 1000
	return 4 * total_hits / iterations, elapsed_ms


def run_reduction(iterations: int, workers: int, seed: int) -> tuple[float, float]:
	partial_hits = [0] * workers
	counts = split_iterations(iterations, workers)

	def worker(worker_id: int, count: int) -> None:
		rng = random.Random(seed + worker_id)
		partial_hits[worker_id] = sum(
			inside_quarter_circle(rng) for _ in range(count)
		)

	started = time.perf_counter()
	threads = [
		threading.Thread(target=worker, args=(worker_id, count))
		for worker_id, count in enumerate(counts)
	]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	total_hits = sum(partial_hits)
	elapsed_ms = (time.perf_counter() - started) * 1000
	return 4 * total_hits / iterations, elapsed_ms


@dataclass
class BenchmarkRow:
	workers: int
	runtime_ms: float
	speedup: float
	efficiency: float


def benchmark_reduction(iterations: int, seed: int) -> list[BenchmarkRow]:
	timings: list[tuple[int, float]] = []
	for workers in THREAD_COUNTS:
		pi, runtime_ms = run_reduction(iterations, workers, seed)
		timings.append((workers, runtime_ms))
		print(f"{workers:>7} {runtime_ms:>12.2f} ms    pi={pi:.7f}")

	baseline = timings[0][1]
	return [
		BenchmarkRow(
			workers=workers,
			runtime_ms=runtime_ms,
			speedup=baseline / runtime_ms,
			efficiency=baseline / runtime_ms / workers,
		)
		for workers, runtime_ms in timings
	]


def print_grid(rows: list[BenchmarkRow]) -> None:
	print("\nThreads  Runtime (ms)  Speedup vs. 1 thread  Efficiency")
	for row in rows:
		print(
			f"{row.workers:>7}  {row.runtime_ms:>12.2f}"
			f"  {row.speedup:>19.2f}x  {row.efficiency * 100:>9.2f}%"
		)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--part",
		choices=("1", "2", "3", "all"),
		default="all",
		help="experiment to run (default: all)",
	)
	parser.add_argument(
		"--iterations",
		type=int,
		default=None,
		help="override the iteration count; use a small value for a quick test",
	)
	parser.add_argument("--seed", type=int, default=230103217)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.iterations is not None and args.iterations <= 0:
		raise SystemExit("--iterations must be positive")

	part1_iterations = args.iterations or DEFAULT_PART1_ITERATIONS
	part3_iterations = args.iterations or DEFAULT_PART3_ITERATIONS

	if args.part in ("1", "all"):
		print("Part 1: unsynchronized shared counter, 5 runs")
		for run_number in range(1, 6):
			pi, runtime_ms = run_phantom_bug(part1_iterations, 4, args.seed + run_number)
			print(f"Run {run_number}: pi={pi:.7f}, runtime={runtime_ms:.2f} ms")

	if args.part in ("2", "all"):
		print("\nPart 2: synchronized shared counter vs. one thread")
		synchronized_pi, synchronized_ms = run_synchronized(
			part1_iterations, 4, args.seed
		)
		baseline_pi, baseline_ms = run_single_thread(part1_iterations, args.seed)
		print(f"Single thread: pi={baseline_pi:.7f}, runtime={baseline_ms:.2f} ms")
		print(f"Synchronized: pi={synchronized_pi:.7f}, runtime={synchronized_ms:.2f} ms")

	if args.part in ("3", "all"):
		print(f"\nPart 3: reduction benchmark ({part3_iterations:,} iterations)")
		rows = benchmark_reduction(part3_iterations, args.seed)
		print_grid(rows)


if __name__ == "__main__":
	main()
