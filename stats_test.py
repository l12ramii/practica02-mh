import json
import os

import pandas as pd

try:
	from statds.no_parametrics import bonferroni, friedman, holm
except ImportError as exc:
	friedman = None
	bonferroni = None
	holm = None
	STATDS_IMPORT_ERROR = exc
else:
	STATDS_IMPORT_ERROR = None


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PER_RUN_RESULTS_CSV = os.path.join(OUTPUT_DIR, "per_run_results.csv")
PER_RUN_RESULTS_JSON = os.path.join(OUTPUT_DIR, "per_run_results.json")
STATISTICAL_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "analisis_estadistico_resumen.json")
STATISTICAL_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "analisis_estadistico_resumen.csv")
FIG_BONFERRONI_PATH = os.path.join(OUTPUT_DIR, "graficas_barras", "friedman_bonferroni_dunn.png")
FIG_HOLM_PATH = os.path.join(OUTPUT_DIR, "graficas_barras", "friedman_holm.png")

ALPHA = 0.05
TARGET_ALGORITHMS = ["genetic_algorithm", "grid_search", "random_search"]
CONTROL_ALGORITHM = "genetic_algorithm"
TARGET_PAIRS = [
	("genetic_algorithm", "grid_search"),
	("genetic_algorithm", "random_search"),
]


def _ensure_statds_available():
	if STATDS_IMPORT_ERROR is not None:
		raise ImportError(
			"No se pudo importar StaTDS. Instala la dependencia con 'pip install statds' "
			"o ejecuta 'pip install -r requirements.txt'."
		) from STATDS_IMPORT_ERROR


def _load_per_run_results():
	if os.path.exists(PER_RUN_RESULTS_CSV):
		df = pd.read_csv(PER_RUN_RESULTS_CSV)
	elif os.path.exists(PER_RUN_RESULTS_JSON):
		with open(PER_RUN_RESULTS_JSON, "r", encoding="utf-8") as handle:
			df = pd.DataFrame(json.load(handle))
	else:
		raise FileNotFoundError(
			"No se encontraron resultados por ejecución. Ejecuta primero 'analisis_rendimiento.py' "
			"para generar 'outputs/per_run_results.csv'."
		)

	required_columns = {"algorithm", "run_id", "best_score"}
	missing_columns = required_columns.difference(df.columns)
	if missing_columns:
		raise ValueError(f"El fichero de resultados no contiene las columnas requeridas: {sorted(missing_columns)}")

	df = df.copy()
	df["algorithm"] = df["algorithm"].astype(str)
	df["run_id"] = pd.to_numeric(df["run_id"], errors="raise").astype(int)
	df["best_score"] = pd.to_numeric(df["best_score"], errors="raise").astype(float)
	return df


def _prepare_friedman_dataset(per_run_df):
	filtered = per_run_df[per_run_df["algorithm"].isin(TARGET_ALGORITHMS)].copy()
	if filtered.empty:
		raise ValueError("No hay corridas para los algoritmos objetivo del análisis estadístico.")

	pivot = filtered.pivot_table(index="run_id", columns="algorithm", values="best_score", aggfunc="mean")
	missing_algorithms = [algorithm for algorithm in TARGET_ALGORITHMS if algorithm not in pivot.columns]
	if missing_algorithms:
		raise ValueError(f"Faltan algoritmos para Friedman: {missing_algorithms}")

	pivot = pivot[TARGET_ALGORITHMS].sort_index()
	
	# Si Grid Search es determinístico (1 corrida), replicar su valor en todas las filas.
	if "grid_search" in pivot.columns and pivot["grid_search"].isna().any():
		grid_search_value = pivot["grid_search"].dropna().iloc[0] if not pivot["grid_search"].dropna().empty else None
		if grid_search_value is not None:
			pivot["grid_search"] = grid_search_value
	
	if pivot.isna().any().any():
		raise ValueError("El conjunto para Friedman contiene valores nulos tras agrupar por run_id.")

	# StaTDS requiere la primera columna como identificador de bloque.
	return pivot.reset_index().rename(columns={"run_id": "block"})


def _serialize(value):
	if isinstance(value, pd.DataFrame):
		return value.to_dict(orient="records")
	if isinstance(value, pd.Series):
		return value.to_dict()
	if isinstance(value, dict):
		return {str(key): _serialize(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_serialize(item) for item in value]
	if hasattr(value, "item"):
		try:
			return value.item()
		except Exception:
			return value
	return value


def _extract_pairwise_results(posthoc_df, method_name):
	pairwise_rows = []
	if posthoc_df is None or posthoc_df.empty:
		return pairwise_rows

	for _, row in posthoc_df.iterrows():
		comparison = str(row.get("Comparison", ""))
		parts = comparison.split(" vs ")
		if len(parts) != 2:
			continue

		algorithm_a, algorithm_b = parts[0].strip(), parts[1].strip()
		pairwise_rows.append(
			{
				"method": method_name,
				"algorithm_a": algorithm_a,
				"algorithm_b": algorithm_b,
				"statistic_z": float(row.get("Statistic (Z)")),
				"p_value": float(row.get("p-value")),
				"adjusted_alpha": float(row.get("Adjusted alpha")),
				"adjusted_p_value": float(row.get("Adjusted p-value")),
				"result": str(row.get("Results", "")),
				"significant_difference": str(row.get("Results", "")).strip().lower() == "reject h0",
			}
		)

	return pairwise_rows


def run_statistical_analysis(alpha=ALPHA):
	_ensure_statds_available()
	per_run_df = _load_per_run_results()
	friedman_dataset = _prepare_friedman_dataset(per_run_df)

	friedman_rankings, statistic, p_value, critical_value, hypothesis = friedman(
		friedman_dataset,
		alpha,
		minimize=False,
	)

	result = {
		"alpha": float(alpha),
		"n_blocks": int(friedman_dataset.shape[0]),
		"algorithms": TARGET_ALGORITHMS,
		"control_algorithm": CONTROL_ALGORITHM,
		"friedman": {
			"statistic": _serialize(statistic),
			"p_value": _serialize(p_value),
			"critical_value": _serialize(critical_value),
			"hypothesis": str(hypothesis),
			"rankings": _serialize(friedman_rankings),
		},
		"post_hoc": {
			"applied": False,
			"methods": [],
			"pairwise_comparisons": {
				"bonferroni_dunn": [],
				"holm": [],
			},
		},
	}

	if str(hypothesis).lower().startswith("reject") or float(p_value) < alpha:
		bonferroni_df, bonferroni_fig = bonferroni(
			friedman_rankings,
			friedman_dataset.shape[0],
			alpha,
			control=CONTROL_ALGORITHM,
			type_rank="Friedman",
		)
		holm_df, holm_fig = holm(
			friedman_rankings,
			friedman_dataset.shape[0],
			alpha,
			control=CONTROL_ALGORITHM,
			type_rank="Friedman",
		)
		bonferroni_rows = _extract_pairwise_results(bonferroni_df, "bonferroni_dunn")
		holm_rows = _extract_pairwise_results(holm_df, "holm")

		os.makedirs(os.path.dirname(FIG_BONFERRONI_PATH), exist_ok=True)
		bonferroni_fig.savefig(FIG_BONFERRONI_PATH, dpi=130, bbox_inches="tight")
		holm_fig.savefig(FIG_HOLM_PATH, dpi=130, bbox_inches="tight")

		result["post_hoc"] = {
			"applied": True,
			"methods": ["bonferroni_dunn", "holm"],
			"pairwise_comparisons": {
				"bonferroni_dunn": bonferroni_rows,
				"holm": holm_rows,
			},
			"tables": {
				"bonferroni_dunn": _serialize(bonferroni_df),
				"holm": _serialize(holm_df),
			},
			"figures": {
				"bonferroni_dunn": FIG_BONFERRONI_PATH,
				"holm": FIG_HOLM_PATH,
			},
		}

	return result, friedman_dataset


def save_results(result):
	os.makedirs(OUTPUT_DIR, exist_ok=True)
	with open(STATISTICAL_OUTPUT_JSON, "w", encoding="utf-8") as handle:
		json.dump(_serialize(result), handle, indent=2, ensure_ascii=False)
	rows = []
	rows.extend(result.get("post_hoc", {}).get("pairwise_comparisons", {}).get("bonferroni_dunn", []))
	rows.extend(result.get("post_hoc", {}).get("pairwise_comparisons", {}).get("holm", []))
	pd.DataFrame(rows).to_csv(
		STATISTICAL_OUTPUT_CSV,
		index=False,
	)


def main():
	result, friedman_dataset = run_statistical_analysis(ALPHA)
	save_results(result)

	print("=== Test de Friedman ===")
	print(friedman_dataset.to_string(index=False))
	print(f"Hipótesis: {result['friedman']['hypothesis']}")
	print(f"Estadístico: {result['friedman']['statistic']}")
	print(f"p-value: {result['friedman']['p_value']}")

	if result["post_hoc"]["applied"]:
		print("\n=== Post-hoc one-vs-all (control: genetic_algorithm) ===")
		print("Bonferroni-Dunn:")
		for comparison in result["post_hoc"]["pairwise_comparisons"]["bonferroni_dunn"]:
			status = "sí" if comparison["significant_difference"] else "no"
			print(
				f"{comparison['algorithm_a']} vs {comparison['algorithm_b']}: "
				f"p_ajustada={comparison['adjusted_p_value']:.6f} | significativa: {status}"
			)
		print("Holm:")
		for comparison in result["post_hoc"]["pairwise_comparisons"]["holm"]:
			status = "sí" if comparison["significant_difference"] else "no"
			print(
				f"{comparison['algorithm_a']} vs {comparison['algorithm_b']}: "
				f"p_ajustada={comparison['adjusted_p_value']:.6f} | significativa: {status}"
			)
	else:
		print("\nNo se aplicó post-hoc porque Friedman no rechazó la hipótesis nula.")

	print(f"\nResultados guardados en {STATISTICAL_OUTPUT_JSON}")


if __name__ == "__main__":
	main()