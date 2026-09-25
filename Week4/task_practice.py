from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path


MODULUS = 1_000_000_007
DEFAULT_STUDENT_ID = 230103270
REQUESTED_THREADS = (1, 2, 4, 8, 16)
CSV_FIELDS = (
	"record", "student_id", "N", "threads", "schedule", "chunk", "cold_s",
	"run2_s", "run3_s", "avg_s", "speedup", "theoretical_speedup",
	"reality_gap", "throughput_iter_s", "penalty_ratio", "max_steps",
	"checksum", "hits_over_100",
)


@dataclass
class KernelResult:
	max_steps: int
	checksum: int
	hits_over_100: int


@dataclass
class Measurement:
	cold: float
	run2: float
	run3: float

	@property
	def average(self) -> float:
		return (self.run2 + self.run3) / 2.0


def collatz_steps(value: int) -> int:
	steps = 0
	while value > 1:
		value = value // 2 if value % 2 == 0 else 3 * value + 1
		steps += 1
	return steps


def workload_for_student(student_id: int) -> int:
	return 10_000_000 + abs(student_id) % 10_000 * 1_000


def sequential_kernel(start: int, end: int) -> KernelResult:
	maximum = 0
	checksum = 0
	hits = 0
	for value in range(start, end + 1):
		steps = collatz_steps(value)
		maximum = max(maximum, steps)
		checksum = (checksum + steps) % MODULUS
		hits += steps > 100
	return KernelResult(maximum, checksum, hits)


def process_chunk(bounds: tuple[int, int]) -> KernelResult:
	return sequential_kernel(*bounds)


def split_ranges(total: int, workers: int) -> list[tuple[int, int]]:
	workers = min(workers, total)
	base, remainder = divmod(total, workers)
	ranges = []
	start = 1
	for index in range(workers):
		size = base + (index < remainder)
		ranges.append((start, start + size - 1))
		start += size
	return ranges


def merge_results(results: list[KernelResult]) -> KernelResult:
	return KernelResult(
		max(result.max_steps for result in results),
		sum(result.checksum for result in results) % MODULUS,
		sum(result.hits_over_100 for result in results),
	)


def parallel_kernel(total: int, workers: int, chunksize: int | None = None) -> KernelResult:
	ranges = split_ranges(total, workers)
	with ProcessPoolExecutor(max_workers=workers) as executor:
		results = list(executor.map(process_chunk, ranges, chunksize=chunksize or 1))
	return merge_results(results)


def measure(operation) -> Measurement:
	def timed() -> float:
		started = time.perf_counter()
		operation()
		return time.perf_counter() - started

	return Measurement(timed(), timed(), timed())


def csv_number(value: float) -> str:
	return f"{value:.9f}"


def empty_row(record: str, student_id: int, total: int, workers: int) -> dict[str, object]:
	return {
		"record": record, "student_id": student_id, "N": total, "threads": workers,
		"schedule": "", "chunk": "", "cold_s": "", "run2_s": "", "run3_s": "",
		"avg_s": "", "speedup": "", "theoretical_speedup": "", "reality_gap": "",
		"throughput_iter_s": "", "penalty_ratio": "", "max_steps": "",
		"checksum": "", "hits_over_100": "",
	}


def save_hardware_info(output_path: Path) -> None:
	info = [f"Python: {platform.python_version()}", f"OS: {platform.platform()}",
			f"CPU count: {os.cpu_count()}"]
	if os.name == "nt":
		command = ["powershell", "-NoProfile", "-Command",
				   "Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List"]
		try:
			info.append(subprocess.check_output(command, text=True, stderr=subprocess.STDOUT))
		except (OSError, subprocess.CalledProcessError):
			pass
	output_path.write_text("\n".join(info) + "\n", encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--student-id", type=int, default=DEFAULT_STUDENT_ID)
	parser.add_argument("--output", default="results.csv")
	parser.add_argument("--quick", action="store_true", help="use N=100,000 for a quick test")
	args = parser.parse_args()

	total = 100_000 if args.quick else workload_for_student(args.student_id)
	max_workers = max(1, os.cpu_count() or 1)
	thread_counts = tuple(worker for worker in REQUESTED_THREADS if worker <= max_workers)
	if not thread_counts:
		thread_counts = (1,)

	rows: list[dict[str, object]] = []
	sequential_result: KernelResult | None = None
	sequential_measurement = measure(lambda: sequential_kernel(1, total))
	sequential_result = sequential_kernel(1, total)
	sequential_time = sequential_measurement.average

	sequential_row = empty_row("sequential", args.student_id, total, 1)
	sequential_row.update({
		"cold_s": csv_number(sequential_measurement.cold),
		"run2_s": csv_number(sequential_measurement.run2),
		"run3_s": csv_number(sequential_measurement.run3),
		"avg_s": csv_number(sequential_time), "speedup": "1.000000000",
		"theoretical_speedup": "1.000000000", "reality_gap": "0.000000000",
		"throughput_iter_s": csv_number(total / sequential_time),
		"max_steps": sequential_result.max_steps, "checksum": sequential_result.checksum,
		"hits_over_100": sequential_result.hits_over_100,
	})
	rows.append(sequential_row)

	measurements: dict[int, Measurement] = {}
	results: dict[int, KernelResult] = {}
	for workers in thread_counts:
		measurement = measure(lambda workers=workers: parallel_kernel(total, workers))
		measurements[workers] = measurement
		results[workers] = parallel_kernel(total, workers)

	speedup_2 = sequential_time / measurements[2].average if 2 in measurements else 1.0
	parallel_fraction = max(0.0, min(1.0, 2.0 * (1.0 - 1.0 / speedup_2)))
	for workers in thread_counts:
		measurement = measurements[workers]
		average = measurement.average
		speedup = sequential_time / average
		theoretical = 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / workers)
		row = empty_row("parallel", args.student_id, total, workers)
		row.update({
			"schedule": "static", "chunk": 0,
			"cold_s": csv_number(measurement.cold), "run2_s": csv_number(measurement.run2),
			"run3_s": csv_number(measurement.run3), "avg_s": csv_number(average),
			"speedup": csv_number(speedup), "theoretical_speedup": csv_number(theoretical),
			"reality_gap": csv_number(theoretical - speedup),
			"throughput_iter_s": csv_number(total / average),
			"max_steps": results[workers].max_steps, "checksum": results[workers].checksum,
			"hits_over_100": results[workers].hits_over_100,
		})
		rows.append(row)

	experiment_workers = max(thread_counts)
	naive = measure(lambda: parallel_kernel(total, experiment_workers))
	reduced = measure(lambda: parallel_kernel(total, experiment_workers))
	for record, measurement in (("false_sharing_naive", naive), ("false_sharing_reduced", reduced)):
		row = empty_row(record, args.student_id, total, experiment_workers)
		row.update({"schedule": "static", "chunk": 0, "cold_s": csv_number(measurement.cold),
					"run2_s": csv_number(measurement.run2), "run3_s": csv_number(measurement.run3),
					"avg_s": csv_number(measurement.average),
					"throughput_iter_s": csv_number(total / measurement.average),
					"penalty_ratio": csv_number(measurement.average / naive.average),
					"hits_over_100": results[experiment_workers].hits_over_100})
		rows.append(row)

	schedules = (("static", 0), ("static", 1000), ("dynamic", 100),
				 ("dynamic", 10000), ("guided", 0))
	for schedule, chunk in schedules:
		chunksize = None if chunk == 0 else chunk
		measurement = measure(lambda chunksize=chunksize: parallel_kernel(total, experiment_workers, chunksize))
		row = empty_row("scheduling", args.student_id, total, experiment_workers)
		row.update({"schedule": schedule, "chunk": chunk,
					"cold_s": csv_number(measurement.cold), "run2_s": csv_number(measurement.run2),
					"run3_s": csv_number(measurement.run3), "avg_s": csv_number(measurement.average),
					"throughput_iter_s": csv_number(total / measurement.average),
					"penalty_ratio": csv_number(measurement.average / sequential_time),
					"hits_over_100": results[experiment_workers].hits_over_100})
		rows.append(row)

	output_path = Path(args.output)
	with output_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
		writer.writeheader()
		writer.writerows(rows)
	save_hardware_info(output_path.with_name("hw_info.txt"))
	print(f"N={total}; workers tested={thread_counts}; p={parallel_fraction:.6f}")
	print(f"checksum={sequential_result.checksum}; max_steps={sequential_result.max_steps}")
	print(f"Wrote {output_path} and {output_path.with_name('hw_info.txt')}")


if __name__ == "__main__":
	multiprocessing.freeze_support()
	main()
