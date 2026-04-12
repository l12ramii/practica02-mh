import json
import math
import os
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.geneticAlgorithm import BINARY_GENES, INTEGER_GENES, LIMITS, genetic_algorithm
from src.gridSearch import grid_search
from src.randomSearch import random_search
from src.utils import evaluate_solution


# ------------------------------
# Configuracion del experimento
# ------------------------------
N_RUNS = 5
RANDOM_SEARCH_ITERS = 640
GA_MAX_EVALS = 640
GA_MODE = "generational"  # "generational" o "steady-state"

# Si no se conoce el optimo, dejar en None.
KNOWN_OPTIMUM = None
OPTIMUM_TOLERANCE = 1e-6

# Salida
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class RunResult:
	algorithm: str
	run_id: int
	best_params: list
	best_score: float
	history: list
	wall_time: float
	cpu_time: float
	cpu_pct: float
	peak_mem_mb: float


def _run_with_measurements(func, *args, **kwargs):
	"""Ejecuta un algoritmo y mide tiempo de pared, CPU y memoria de proceso principal."""
	tracemalloc.start()
	t0_wall = time.perf_counter()
	t0_cpu = time.process_time()

	best_params, best_score, history, _ = func(*args, **kwargs)

	wall_time = time.perf_counter() - t0_wall
	cpu_time = time.process_time() - t0_cpu
	_curr, peak_mem_bytes = tracemalloc.get_traced_memory()
	tracemalloc.stop()

	cpu_pct = (cpu_time / wall_time) * 100.0 if wall_time > 0 else 0.0
	peak_mem_mb = peak_mem_bytes / (1024 * 1024)

	return best_params, best_score, history, wall_time, cpu_time, cpu_pct, peak_mem_mb


def _iterate_running_best(history):
	return np.maximum.accumulate(np.asarray(history, dtype=float))


def _iterations_to_target(history, target):
	rb = _iterate_running_best(history)
	reached = np.where(rb >= target)[0]
	if len(reached) == 0:
		return len(history)
	return int(reached[0] + 1)


def _coefficient_of_variation(values):
	arr = np.asarray(values, dtype=float)
	mean = float(np.mean(arr))
	std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
	if math.isclose(mean, 0.0, abs_tol=1e-12):
		return 0.0
	return std / abs(mean)


def _param_neighbors(base_params, idx, step_ratio=0.1):
	"""Genera dos vecinos (+/-) para sensibilidad local de un parametro."""
	low = LIMITS[idx, 0]
	high = LIMITS[idx, 1]
	delta = (high - low) * step_ratio

	p_minus = list(base_params)
	p_plus = list(base_params)

	p_minus[idx] = max(low, float(base_params[idx]) - delta)
	p_plus[idx] = min(high, float(base_params[idx]) + delta)

	for p in (p_minus, p_plus):
		if idx in BINARY_GENES:
			p[idx] = int(np.clip(round(p[idx]), 0, 1))
		elif idx in INTEGER_GENES:
			p[idx] = int(round(p[idx]))
		else:
			p[idx] = float(p[idx])

	return p_minus, p_plus


def _estimate_param_sensitivity(best_params, baseline_score):
	"""
	Sensibilidad local por parametro:
	media de la perdida de fitness al perturbar +/-10% del rango.
	"""
	names = [
		"n_estimators",
		"max_depth",
		"min_samples_split",
		"min_samples_leaf",
		"max_features",
		"bootstrap",
		"criterion",
		"class_weight",
		"max_leaf_nodes",
		"min_impurity_decrease",
	]

	rows = []
	for idx, name in enumerate(names):
		p_minus, p_plus = _param_neighbors(best_params, idx)
		score_minus = evaluate_solution(p_minus)
		score_plus = evaluate_solution(p_plus)

		drop_minus = max(0.0, baseline_score - score_minus)
		drop_plus = max(0.0, baseline_score - score_plus)
		mean_drop = (drop_minus + drop_plus) / 2.0

		rows.append(
			{
				"param": name,
				"score_minus": score_minus,
				"score_plus": score_plus,
				"drop_mean": mean_drop,
			}
		)

	sens_df = pd.DataFrame(rows).sort_values("drop_mean", ascending=False).reset_index(drop=True)
	return sens_df


def _run_experiment():
	algorithms = {
		"random_search": lambda: random_search(n_iter=RANDOM_SEARCH_ITERS, patience=None, min_improvement=1e-4),
		"grid_search": lambda: grid_search(patience=None, min_improvement=1e-4),
		"genetic_algorithm": lambda: genetic_algorithm(
			mode=GA_MODE,
			max_evals=GA_MAX_EVALS,
			patience=None,
			min_improvement=1e-4,
		),
	}

	all_runs = []

	for alg_name, runner in algorithms.items():
		print(f"\n=== Ejecutando {alg_name} ({N_RUNS} repeticiones) ===")
		for run_id in range(1, N_RUNS + 1):
			print(f"\n[{alg_name}] ejecución {run_id}/{N_RUNS}")
			(
				best_params,
				best_score,
				history,
				wall_time,
				cpu_time,
				cpu_pct,
				peak_mem_mb,
			) = _run_with_measurements(runner)

			all_runs.append(
				RunResult(
					algorithm=alg_name,
					run_id=run_id,
					best_params=best_params,
					best_score=float(best_score),
					history=[float(x) for x in history],
					wall_time=float(wall_time),
					cpu_time=float(cpu_time),
					cpu_pct=float(cpu_pct),
					peak_mem_mb=float(peak_mem_mb),
				)
			)

	return all_runs


def _build_reports(all_runs):
	scores = np.array([r.best_score for r in all_runs], dtype=float)
	best_known = float(np.max(scores))
	reference_optimum = KNOWN_OPTIMUM if KNOWN_OPTIMUM is not None else best_known

	per_run_rows = []
	summary_rows = []
	convergence_rows = []

	for alg in sorted(set(r.algorithm for r in all_runs)):
		runs = [r for r in all_runs if r.algorithm == alg]
		best_scores = np.array([r.best_score for r in runs], dtype=float)

		distances = np.maximum(0.0, reference_optimum - best_scores)
		if KNOWN_OPTIMUM is not None:
			hits = int(np.sum(np.abs(best_scores - KNOWN_OPTIMUM) <= OPTIMUM_TOLERANCE))
		else:
			hits = int(np.sum(np.abs(best_scores - best_known) <= OPTIMUM_TOLERANCE))

		target_95 = 0.95 * reference_optimum
		iters_to_95 = [_iterations_to_target(r.history, target_95) for r in runs]

		summary_rows.append(
			{
				"algorithm": alg,
				"runs": len(runs),
				"mean_best_score": float(np.mean(best_scores)),
				"std_best_score": float(np.std(best_scores, ddof=1)) if len(runs) > 1 else 0.0,
				"min_best_score": float(np.min(best_scores)),
				"max_best_score": float(np.max(best_scores)),
				"cv_best_score": float(_coefficient_of_variation(best_scores)),
				"optimum_reference": float(reference_optimum),
				"times_reached_reference": hits,
				"mean_distance_to_reference": float(np.mean(distances)),
				"max_distance_to_reference": float(np.max(distances)),
				"mean_iter_to_95pct_reference": float(np.mean(iters_to_95)),
				"std_iter_to_95pct_reference": float(np.std(iters_to_95, ddof=1)) if len(runs) > 1 else 0.0,
				"mean_wall_time_s": float(np.mean([r.wall_time for r in runs])),
				"std_wall_time_s": float(np.std([r.wall_time for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
				"mean_cpu_time_s": float(np.mean([r.cpu_time for r in runs])),
				"mean_cpu_usage_pct": float(np.mean([r.cpu_pct for r in runs])),
				"mean_peak_mem_mb_main_process": float(np.mean([r.peak_mem_mb for r in runs])),
			}
		)

		max_len = max(len(r.history) for r in runs)
		rb_series = []
		for r in runs:
			rb = _iterate_running_best(r.history)
			if len(rb) < max_len:
				pad = np.full(max_len - len(rb), rb[-1])
				rb = np.concatenate([rb, pad])
			rb_series.append(rb)
		rb_mat = np.vstack(rb_series)

		for i in range(max_len):
			convergence_rows.append(
				{
					"algorithm": alg,
					"iteration": i + 1,
					"mean_running_best": float(np.mean(rb_mat[:, i])),
					"std_running_best": float(np.std(rb_mat[:, i], ddof=1)) if len(runs) > 1 else 0.0,
				}
			)

		for r in runs:
			per_run_rows.append(
				{
					"algorithm": r.algorithm,
					"run_id": r.run_id,
					"best_score": r.best_score,
					"wall_time_s": r.wall_time,
					"cpu_time_s": r.cpu_time,
					"cpu_usage_pct": r.cpu_pct,
					"peak_mem_mb_main_process": r.peak_mem_mb,
					"history_len": len(r.history),
					"distance_to_reference": max(0.0, reference_optimum - r.best_score),
					"best_params": json.dumps(r.best_params),
				}
			)

	summary_df = pd.DataFrame(summary_rows).sort_values("mean_best_score", ascending=False)
	per_run_df = pd.DataFrame(per_run_rows)
	convergence_df = pd.DataFrame(convergence_rows)

	top_run = max(all_runs, key=lambda x: x.best_score)
	print("Calculando sensibilidad de parámetros...")
	sensitivity_df = _estimate_param_sensitivity(top_run.best_params, top_run.best_score)

	return summary_df, per_run_df, convergence_df, sensitivity_df, reference_optimum, best_known, top_run


def _save_summary_bar_charts(summary_df):
	"""Genera una grafica de barras por estadistica con una barra por algoritmo."""
	charts_dir = os.path.join(OUTPUT_DIR, "graficas_barras")
	os.makedirs(charts_dir, exist_ok=True)

	# Excluir metricas solicitadas y todas las relacionadas con "reference".
	excluded_metrics = {
		"runs",
		"mean_best_score",
		"cv_best_score",
		"mean_wall_time_s",
		"std_wall_time_s",
		"mean_peak_mem_mb_main_process",
		"min_best_score",
		"max_best_score",
	}

	metric_columns = [
		col for col in summary_df.columns
		if col != "algorithm"
		and pd.api.types.is_numeric_dtype(summary_df[col])
		and col not in excluded_metrics
		and "reference" not in col
	]

	algorithm_labels = summary_df["algorithm"].tolist()
	generated_files = []

	# Grafica combinada: minimo, media y maximo por algoritmo.
	if {"min_best_score", "mean_best_score", "max_best_score"}.issubset(summary_df.columns):
		min_values = summary_df["min_best_score"].astype(float).to_numpy()
		mean_values = summary_df["mean_best_score"].astype(float).to_numpy()
		max_values = summary_df["max_best_score"].astype(float).to_numpy()
		x = np.arange(len(algorithm_labels))
		width = 0.26

		fig, ax = plt.subplots(figsize=(9, 5))
		bars_min = ax.bar(x - width, min_values, width, label="min_best_score")
		bars_mean = ax.bar(x, mean_values, width, label="mean_best_score")
		bars_max = ax.bar(x + width, max_values, width, label="max_best_score")

		ax.set_title("min - med - max mejores resultados por algoritmo")
		ax.set_xlabel("Algoritmo")
		ax.set_ylabel("best_score")
		ax.set_xticks(x)
		ax.set_xticklabels(algorithm_labels)
		ax.grid(axis="y", alpha=0.3)
		ax.legend()

		for bars in (bars_min, bars_mean, bars_max):
			for bar in bars:
				val = bar.get_height()
				offset = max(abs(val) * 0.01, 1e-6)
				ax.text(
					bar.get_x() + bar.get_width() / 2,
					val + offset,
					f"{val:.4f}",
					ha="center",
					va="bottom",
					fontsize=9,
				)

		fig.tight_layout()
		min_max_path = os.path.join(charts_dir, "min_med_max_mejores.png")
		fig.savefig(min_max_path, dpi=130)
		plt.close(fig)
		generated_files.append(min_max_path)

	for metric in metric_columns:
		values = summary_df[metric].astype(float).to_numpy()
		fig, ax = plt.subplots(figsize=(8, 5))
		bars = ax.bar(algorithm_labels, values)

		ax.set_title(f"{metric} por algoritmo")
		ax.set_xlabel("Algoritmo")
		ax.set_ylabel(metric)
		ax.grid(axis="y", alpha=0.3)

		for bar, val in zip(bars, values):
			height = bar.get_height()
			offset = max(abs(val) * 0.01, 1e-6)
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				height + offset,
				f"{val:.4f}",
				ha="center",
				va="bottom",
				fontsize=9,
			)

		fig.tight_layout()
		safe_name = metric.replace("%", "pct").replace("/", "_")
		chart_path = os.path.join(charts_dir, f"{safe_name}.png")
		fig.savefig(chart_path, dpi=130)
		plt.close(fig)
		generated_files.append(chart_path)

	return generated_files


def _save_combined_convergence_chart(convergence_df):
	"""Genera una grafica con las convergencias medias de todos los algoritmos."""
	charts_dir = os.path.join(OUTPUT_DIR, "graficas_barras")
	os.makedirs(charts_dir, exist_ok=True)

	fig, ax = plt.subplots(figsize=(10, 6))
	for algorithm in sorted(convergence_df["algorithm"].unique()):
		df_alg = convergence_df[convergence_df["algorithm"] == algorithm].sort_values("iteration")
		ax.plot(
			df_alg["iteration"].to_numpy(),
			df_alg["mean_running_best"].to_numpy(),
			label=algorithm,
			linewidth=2,
		)

	ax.set_title("Convergencia media por algoritmo")
	ax.set_xlabel("Iteración")
	ax.set_ylabel("Running best medio")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()

	chart_path = os.path.join(charts_dir, "convergencia_comparada.png")
	fig.savefig(chart_path, dpi=130)
	plt.close(fig)
	return chart_path


def _save_outputs(summary_df, per_run_df, convergence_df, sensitivity_df, reference_optimum, best_known, top_run):
	json_out = os.path.join(OUTPUT_DIR, "analisis_rendimiento_resumen.json")
	print("Generando gráficas de barras...")
	charts = _save_summary_bar_charts(summary_df)
	print("Generando gráfica de convergencia comparada...")
	convergence_chart = _save_combined_convergence_chart(convergence_df)

	payload = {
		"config": {
			"n_runs": N_RUNS,
			"random_search_iters": RANDOM_SEARCH_ITERS,
			"ga_max_evals": GA_MAX_EVALS,
			"ga_mode": GA_MODE,
			"known_optimum": KNOWN_OPTIMUM,
			"optimum_tolerance": OPTIMUM_TOLERANCE,
		},
		"reference_optimum_used": float(reference_optimum),
		"best_known_found": float(best_known),
		"best_overall_run": {
			"algorithm": top_run.algorithm,
			"run_id": top_run.run_id,
			"best_score": float(top_run.best_score),
			"best_params": top_run.best_params,
		},
		"summary_by_algorithm": summary_df.to_dict(orient="records"),
		"parameter_sensitivity": sensitivity_df.to_dict(orient="records"),
		"notes": [
			"peak_mem_mb_main_process mide memoria del proceso principal.",
			"En algoritmos multiproceso (random/grid) puede infraestimar memoria total.",
			"Si KNOWN_OPTIMUM es None, se usa best_known_found como referencia de optimo.",
		],
	}

	with open(json_out, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2, ensure_ascii=False)

	print("\n=== Archivos generados ===")
	print(f"- {os.path.join(OUTPUT_DIR, 'graficas_barras')}")
	print(f"- Total de graficas: {len(charts) + 1}")
	print(f"- {convergence_chart}")
	print(f"- {json_out}")


def main():
	print("Iniciando análisis de rendimiento...")
	all_runs = _run_experiment()
	(
		summary_df,
		per_run_df,
		convergence_df,
		sensitivity_df,
		reference_optimum,
		best_known,
		top_run,
	) = _build_reports(all_runs)

	_save_outputs(
		summary_df,
		per_run_df,
		convergence_df,
		sensitivity_df,
		reference_optimum,
		best_known,
		top_run,
	)

	print("\n=== Resumen rápido ===")
	print(summary_df[["algorithm", "mean_best_score", "std_best_score", "mean_wall_time_s"]].to_string(index=False))
	print("\nAnálisis completado.")


if __name__ == "__main__":
	main()
