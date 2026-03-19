from pathlib import Path
from ps_crosstalk_analysis import save_crosstalk_matrix

data_dir = "measurements/v2pi_batch_ps_scan_results_run_010"
all_ps_taps = list(range(9, 17))
per_ps_csvs = {
    tap: Path(f"{data_dir}/ps_crosstalk_ps_{tap}.csv") for tap in all_ps_taps
}

save_crosstalk_matrix(
    per_ps_csv_paths=per_ps_csvs,
    resistance_ohm=600.0,
    output_path=Path(f"{data_dir}/ps_crosstalk_matrix.csv"),
)
