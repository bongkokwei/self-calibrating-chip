# Self-Calibrating Photonic FIR Filter

Python implementation of the self-calibration pipeline for a 16-tap Si₃N₄ programmable photonic FIR filter (LioniX TriPleX platform, ~160 GHz FSR), based on [Xu et al. 2022](https://doi.org/10.1038/s41566-022-01020-z).

## Prerequisites

- Luna OVA connected and reachable at the IP in `config/calibration_config.yaml`
- Voltage controller on the configured COM port (default `COM3`)
- EDFA controller on the configured COM port (default `COM6`)
- TEC temperature stabilisation active

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Calibration

Edit `config/calibration_config.yaml` to set your target filter response, then run:

```bash
python experiment/run_calibration.py
# or with a custom config:
python experiment/run_calibration.py path/to/config.yaml
```

The script runs in **sequential mode** by default: amplitude calibration (MZI power splitting ratios) converges first, then phase calibration (PS tap phases via Kramers-Kronig recovery) activates. Results and plots are saved to `measurements/`.

### Key config flags

| Flag | Description |
|---|---|
| `calibration.sequential_mode` | `true` = amplitude-first then phase (recommended) |
| `calibration.use_gap_method` | `true` = use gap method instead of KK phase recovery |
| `calibration.ps_crosstalk_matrix_path` | Path to PS thermal crosstalk matrix CSV |

## Characterisation Scripts

| Script | Purpose |
|---|---|
| `experiment/batch_mzi_scan.py` | Characterise V₂π and φ_init for all MZIs |
| `experiment/batch_ps_scan.py` | Characterise V₂π and φ_init for all phase shifters |
| `experiment/run_ps_batch_scan.py` | Batch PS crosstalk analysis across all signal taps |
| `experiment/easy_voltage_control.py` | Manual voltage control via CLI |

## Reference

Xu et al. (2022). Self-calibrating programmable photonic integrated circuits. *Nature Photonics*, 16, 595–602.