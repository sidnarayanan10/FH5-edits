# FH5 Aero & Agility Telemetry Optimizer

A closed-loop vehicle dynamics calculator and live telemetry analyzer for **Forza Horizon 5**. This project models aerodynamic downforce balance and mechanical suspension setups in Python, captures real-time physics data via FH5's UDP **Data Out** stream, and validates whether calculated tuning changes measurably improve cornering grip and turn-in agility.

---

## Overview

Most vehicle setups rely on trial-and-error slider adjustments without quantitative feedback. This tool treats game tuning as a three-stage engineering test loop:

1. **Calculate (`tune_calculator.py`):** Computes target front/rear downforce, anti-roll bar (ARB) stiffness, spring rates, and alignment geometry based on vehicle mass, weight distribution, drivetrain, and target circuit profile.
2. **Capture (`telemetry_logger.py`):** Listens to Forza Horizon 5's local UDP telemetry stream (`324-byte` Dash packet format) and records vehicle dynamics—speed, lateral G-force, yaw rate, steering input, tire slip ratios, and normalized suspension travel—to CSV logs.
3. **Validate (`analyze_run.py`):** Compares baseline (stock) runs against tuned runs to quantify improvements in lateral grip, high-speed aerodynamic load, steering-to-yaw agility, and suspension bottom-out margins.

---

## Physics & Tuning Methodology

### 1. Aerodynamic Downforce & Balance
Aerodynamic downforce scales with the square of velocity ($F_d \propto v^2$). While maximizing rear downforce increases high-speed cornering stability, excessive rear wing angle without matching front splitter load shifts the aerodynamic center of pressure rearward, inducing high-speed understeer.
* **High-Agility / Technical Mode:** Maximizes front downforce (`100%` of front aero range) and balances rear downforce relative to front weight percentage to sharpen turn-in response.
* **High-Speed Mode:** Increases rear downforce margin to stabilize high-velocity sweepers while maintaining positive front turn-in bite.

### 2. Mechanical Roll Stiffness & Weight Distribution
Baseline anti-roll bars and spring rates are scaled proportionally to front/rear mass distribution:

$$
\text{Setting} = (\text{Max} - \text{Min}) \times W_{\text{frac}} + \text{Min}
$$

* **Agility Bias:** Slightly stiffening the rear ARB relative to the front promotes rotational yaw response during corner entry.
* **Aero Load Compensation:** Increasing downforce adds vertical load onto the suspension at high speed. Spring rates are scaled upward alongside total downforce targets to prevent normalized suspension travel from reaching `1.0` (bottoming out) in high-speed compression zones.

---

## Project Structure

```text
fh5-aero-telemetry-optimizer/
├── main.py                # Unified CLI for calculating tunes, recording UDP data, and running analysis
├── tune_calculator.py     # Physics-based aero, ARB, spring, damping, and camber calculator
├── telemetry_logger.py    # Live UDP socket listener & FH5 Dash packet parser
├── analyze_run.py         # Statistical comparison & matplotlib visualization generator
├── requirements.txt       # Python dependencies (numpy, pandas, matplotlib)
├── data/                  # Recorded and sample telemetry CSV logs
└── output/                # Generated comparison plots and validation summaries
```

---

## Usage & CLI Commands

Open PowerShell in the project root directory to run the following workflow commands.

### 1. Calculate a Tune

**Command Format:**
```powershell
python main.py calculate `
  --weight <FINAL_WEIGHT> `
  --front-weight <FRONT_PCT> `
  --drivetrain FWD `
  --front-aero <MIN> <MAX> `
  --rear-aero <MIN> <MAX> `
  --front-arb <MIN> <MAX> `
  --rear-arb <MIN> <MAX> `
  --front-springs <MIN> <MAX> `
  --rear-springs <MIN> <MAX> `
  --ride-height <MIN> <MAX> `
  --profile technical
```

**Example:**
```powershell
python main.py calculate `
  --weight 3035lbs `
  --front-weight 65 `
  --drivetrain FWD `
  --front-aero 209 399 `
  --rear-aero 194 383 `
  --front-arb 1 65 `
  --rear-arb 1 65 `
  --front-springs 314.5 1572.4 `
  --rear-springs 314.5 1572.4 `
  --ride-height 6.2 7.3 `
  --profile technical
```

### 2. Record Telemetry Data from FH5
Start the UDP listener before beginning your test run in Forza Horizon 5:
```powershell
python main.py record --host 0.0.0.0 --port 20044
```

### 3. Analyze & Validate Runs
Compare your baseline and tuned telemetry CSV logs after data collection:
```powershell
python main.py analyze --baseline data/CivicRun1_tuned.csv --tuned data/CivicRun1_baseline.csv
```

---

## Contributing / Pushing Changes to Git

To save and push your updates or new telemetry logs to the repository:

```bash
git add .
git commit -m "Describe your changes here"
git push
```