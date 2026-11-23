"""Compare two FH5 telemetry logs (baseline vs tuned) with a summary table and plots."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from telemetry_logger import CSV_COLUMNS, PROJECT_DIR, ensure_dir

OUTPUT_DIR = PROJECT_DIR / "output"
DEFAULT_OUTPUT = OUTPUT_DIR / "telemetry_comparison.png"

HIGH_SPEED_MPS = 45.0
BOTTOM_OUT_LEVEL = 0.98
MIN_MOVING_SPEED_MPS = 5.0
RAD_TO_DEG = 180.0 / math.pi

WHEELS = ("fl", "fr", "rl", "rr")
SLIP_COLUMNS = [f"tire_slip_ratio_{w}" for w in WHEELS]
SUSP_COLUMNS = [f"susp_travel_norm_{w}" for w in WHEELS]

BASELINE_COLOR = "#d1495b"
TUNED_COLOR = "#00798c"


def load_telemetry(path):
    """Read a telemetry CSV and check that all required columns and enough rows exist."""
    if not path.is_file():
        raise FileNotFoundError(f"telemetry file not found: {path}")
    frame = pd.read_csv(path)
    missing = [c for c in CSV_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing column(s): {', '.join(missing)}")
    frame = frame[CSV_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < 100:
        raise ValueError(f"{path.name} has only {len(frame)} usable rows; need at least 100")
    return frame.reset_index(drop=True)


def get_cornering_mask(frame, threshold_g):
    """True for rows where the car is moving and cornering hard enough."""
    return (
        (frame["lat_accel_g"].abs() >= threshold_g) & (frame["speed_mps"] > MIN_MOVING_SPEED_MPS)
    ).to_numpy()


def estimate_sample_period_s(frame):
    """Find the typical time gap between rows in seconds (defaults to 1/60s)."""
    median_ms = float(np.median(np.diff(frame["timestamp_ms"].to_numpy())))
    if 0.0 < median_ms < 500.0:
        return median_ms / 1000.0
    return 1.0 / 60.0


def estimate_turn_in_lag_ms(steer, yaw, dt_s, max_lag_s=0.6):
    """Measure how many milliseconds yaw rate lags behind steering input."""
    n = len(steer)
    max_lag = int(max_lag_s / dt_s)
    if n < 4 * max_lag or np.std(steer) == 0 or np.std(yaw) == 0:
        return float("nan")

    # Normalize both signals so we can compare their shapes
    s = (steer - steer.mean()) / steer.std()
    y = (yaw - yaw.mean()) / yaw.std()

    # Test every possible lag from 0 up to max_lag steps
    corr = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        corr[lag] = np.mean(s[: n - lag] * y[lag:])

    # Flip sign if steering and yaw use opposite +/- conventions
    if corr[np.argmax(np.abs(corr))] < 0:
        corr = -corr

    peak = int(np.argmax(corr))
    offset = 0.0
    if 0 < peak < max_lag:
        denom = corr[peak - 1] - 2.0 * corr[peak] + corr[peak + 1]
        if denom != 0:
            offset = 0.5 * (corr[peak - 1] - corr[peak + 1]) / denom

    return float((peak + offset) * dt_s * 1000.0)


def compute_metrics(frame, corner_threshold_g=0.5):
    """Calculate all summary stats for one run and return them as a dictionary."""
    corner = get_cornering_mask(frame, corner_threshold_g)
    lat = frame["lat_accel_g"].abs().to_numpy()
    speed = frame["speed_mps"].to_numpy()

    peak_lat = float(lat.max())
    p95_lat = float(np.percentile(lat[corner], 95)) if corner.any() else float("nan")

    high_speed = corner & (speed > HIGH_SPEED_MPS)
    hs_speed = float(speed[high_speed].mean()) if high_speed.any() else float("nan")

    steer = frame["steer_input"].to_numpy()
    yaw_deg = frame["yaw_rate"].to_numpy() * RAD_TO_DEG
    steering_active = (np.abs(steer) > 0.05) & (speed > MIN_MOVING_SPEED_MPS)
    if steering_active.any():
        gain = float(
            np.sum(np.abs(steer[steering_active]) * np.abs(yaw_deg[steering_active]))
            / np.sum(steer[steering_active] ** 2)
        )
    else:
        gain = float("nan")

    lag_ms = estimate_turn_in_lag_ms(steer, yaw_deg, estimate_sample_period_s(frame))
    avg_slip = float(frame.loc[corner, SLIP_COLUMNS].abs().to_numpy().mean()) if corner.any() else float("nan")

    susp_peaks = {w: float(frame[f"susp_travel_norm_{w}"].max()) for w in WHEELS}
    susp_all = frame[SUSP_COLUMNS].to_numpy()
    susp_peak = max(susp_peaks.values())

    return {
        "samples": len(frame),
        "peak_lat_g": peak_lat,
        "p95_lat_g": p95_lat,
        "high_speed_corner_speed_mps": hs_speed,
        "high_speed_samples": int(high_speed.sum()),
        "yaw_gain_deg_per_steer": gain,
        "peak_yaw_rate_deg": float(np.abs(yaw_deg).max()),
        "turn_in_lag_ms": lag_ms,
        "avg_corner_slip": avg_slip,
        "susp_peak_by_wheel": susp_peaks,
        "susp_peak": susp_peak,
        "bottom_out_samples": int((susp_all.max(axis=1) >= BOTTOM_OUT_LEVEL).sum()),
        "bottomed_out": susp_peak >= BOTTOM_OUT_LEVEL,
    }


def fmt(value, decimals, unit=""):
    return "n/a" if math.isnan(value) else f"{value:.{decimals}f}{unit}"


def verdict(base, tuned, higher_is_better):
    if higher_is_better is None or math.isnan(base) or math.isnan(tuned):
        return ""
    scale = max(abs(base), 1e-9)
    if abs(tuned - base) / scale < 0.01:
        return "same"
    improved = (tuned > base) == higher_is_better
    return "better" if improved else "worse"


def format_summary(base, tuned, baseline_name, tuned_name):
    rows = [
        ("Peak lateral G", base["peak_lat_g"], tuned["peak_lat_g"], 2, " g", True),
        ("95th-pct lateral G (cornering)", base["p95_lat_g"], tuned["p95_lat_g"], 2, " g", True),
        (f"High-speed corner speed (>{HIGH_SPEED_MPS:.0f} m/s)", base["high_speed_corner_speed_mps"],
         tuned["high_speed_corner_speed_mps"], 1, " m/s", True),
        ("Yaw gain (deg/s per full steer)", base["yaw_gain_deg_per_steer"], tuned["yaw_gain_deg_per_steer"],
         1, "", True),
        ("Peak yaw rate", base["peak_yaw_rate_deg"], tuned["peak_yaw_rate_deg"], 1, " deg/s", True),
        ("Steering-to-yaw lag", base["turn_in_lag_ms"], tuned["turn_in_lag_ms"], 0, " ms", False),
        ("Avg |tire slip| in corners", base["avg_corner_slip"], tuned["avg_corner_slip"], 3, "", False),
        ("Suspension travel peak", base["susp_peak"], tuned["susp_peak"], 3, "", None),
    ]

    label_w = max(len(r[0]) for r in rows) + 2
    col_w = 14
    header = (
        f"{'Metric':<{label_w}}{baseline_name[:col_w - 1]:>{col_w}}{tuned_name[:col_w - 1]:>{col_w}}"
        f"{'Change':>{col_w}}  Verdict"
    )
    lines = ["", "TELEMETRY COMPARISON", "=" * len(header), header, "-" * len(header)]
    for label, b, t, decimals, unit, higher_better in rows:
        change = "n/a" if math.isnan(b) or math.isnan(t) else f"{t - b:+.{decimals}f}"
        lines.append(
            f"{label:<{label_w}}{fmt(b, decimals, unit):>{col_w}}{fmt(t, decimals, unit):>{col_w}}"
            f"{change:>{col_w}}  {verdict(b, t, higher_better)}"
        )

    def bottom_out_text(m):
        if m["bottomed_out"]:
            return f"BOTTOMED OUT ({m['bottom_out_samples']} samples >= {BOTTOM_OUT_LEVEL})"
        return f"OK (margin {BOTTOM_OUT_LEVEL - m['susp_peak']:.3f} below {BOTTOM_OUT_LEVEL})"

    lines += [
        "-" * len(header),
        f"Bottom-out check ({baseline_name}): {bottom_out_text(base)}",
        f"Bottom-out check ({tuned_name}): {bottom_out_text(tuned)}",
        "Peak travel per wheel (FL/FR/RL/RR): "
        + " | ".join(
            f"{name}: " + "/".join(f"{m['susp_peak_by_wheel'][w]:.2f}" for w in WHEELS)
            for name, m in ((baseline_name, base), (tuned_name, tuned))
        ),
        f"High-speed cornering samples: {baseline_name} {base['high_speed_samples']}, "
        f"{tuned_name} {tuned['high_speed_samples']}",
        "=" * len(header),
    ]
    return "\n".join(lines)


def smooth_density_curve(samples, grid, max_samples=8000):
    """Draw a smooth bell-curve density line over a histogram (Gaussian KDE)."""
    if len(samples) > max_samples:
        samples = np.random.default_rng(0).choice(samples, max_samples, replace=False)
    std = samples.std()
    iqr = np.subtract(*np.percentile(samples, [75, 25]))
    spread = min(std, iqr / 1.34) if iqr > 0 else std
    bandwidth = max(0.9 * spread * len(samples) ** -0.2, 1e-3)
    z = (grid[:, None] - samples[None, :]) / bandwidth
    return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(samples) * bandwidth * math.sqrt(2.0 * math.pi))


def save_comparison_figure(
    baseline, tuned, base_metrics, tuned_metrics, output_path,
    baseline_name="Baseline", tuned_name="Tuned", corner_threshold_g=0.5,
):
    ensure_dir(output_path.parent)
    frames = (baseline, tuned)
    names = (baseline_name, tuned_name)
    metrics = (base_metrics, tuned_metrics)
    colors = (BASELINE_COLOR, TUNED_COLOR)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Lateral G distribution
    ax1 = axes[0, 0]
    all_lat = np.concatenate([f["lat_accel_g"].abs().to_numpy() for f in frames])
    upper = max(1.0, float(np.percentile(all_lat, 99.9)) * 1.05)
    grid = np.linspace(corner_threshold_g, upper, 300)
    for frame, name, color in zip(frames, names, colors):
        lat = frame["lat_accel_g"].abs().to_numpy()
        lat = lat[lat >= corner_threshold_g]
        if len(lat) >= 10:
            ax1.hist(lat, bins=40, range=(corner_threshold_g, upper), density=True, alpha=0.30, color=color)
            ax1.plot(grid, smooth_density_curve(lat, grid), color=color, linewidth=2,
                     label=f"{name} (p95 {np.percentile(lat, 95):.2f} g)")
    ax1.set_title("1. Lateral G-Force Distribution (cornering)")
    ax1.set_xlabel("|Lateral acceleration| (g)")
    ax1.set_ylabel("Density")
    ax1.legend()

    # Panel 2: Speed vs Lateral G
    ax2 = axes[0, 1]
    ax2.axvspan(HIGH_SPEED_MPS, 100, color="#f2e8cf", alpha=0.5, label=f"aero regime (>{HIGH_SPEED_MPS:.0f} m/s)")
    max_speed = 0.0
    for frame, name, color in zip(frames, names, colors):
        corner = get_cornering_mask(frame, corner_threshold_g)
        speed = frame["speed_mps"].to_numpy()[corner]
        lat = frame["lat_accel_g"].abs().to_numpy()[corner]
        if len(speed) == 0:
            continue
        max_speed = max(max_speed, float(speed.max()))
        keep = np.random.default_rng(1).choice(len(speed), min(len(speed), 4000), replace=False)
        ax2.scatter(speed[keep], lat[keep], s=4, alpha=0.25, color=color)

        edges = np.arange(speed.min(), speed.max() + 4.0, 4.0)
        centers, envelope = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (speed >= lo) & (speed < hi)
            if in_bin.sum() >= 20:
                centers.append((lo + hi) / 2.0)
                envelope.append(np.percentile(lat[in_bin], 95))
        ax2.plot(centers, envelope, color=color, linewidth=2.5, label=f"{name} grip envelope (p95)")
    ax2.set_xlim(0, max(max_speed * 1.05, HIGH_SPEED_MPS + 5))
    ax2.set_title("2. High-Speed Cornering Grip")
    ax2.set_xlabel("Speed (m/s)")
    ax2.set_ylabel("|Lateral acceleration| (g)")
    ax2.legend(loc="upper left")

    # Panel 3: Steering vs Yaw Rate
    ax3 = axes[1, 0]
    max_steer = 0.0
    for frame, name, color, m in zip(frames, names, colors, metrics):
        steer = frame["steer_input"].abs().to_numpy()
        yaw = frame["yaw_rate"].abs().to_numpy() * RAD_TO_DEG
        active = (steer > 0.05) & (frame["speed_mps"].to_numpy() > MIN_MOVING_SPEED_MPS)
        if not active.any():
            continue
        steer, yaw = steer[active], yaw[active]
        max_steer = max(max_steer, float(steer.max()))
        keep = np.random.default_rng(2).choice(len(steer), min(len(steer), 4000), replace=False)
        ax3.scatter(steer[keep], yaw[keep], s=4, alpha=0.25, color=color)
        xs = np.linspace(0, steer.max(), 2)
        ax3.plot(xs, m["yaw_gain_deg_per_steer"] * xs, color=color, linewidth=2.5,
                 label=f"{name}: {m['yaw_gain_deg_per_steer']:.1f} deg/s per steer, lag {fmt(m['turn_in_lag_ms'], 0)} ms")
    ax3.set_title("3. Turn-In Agility: Steering vs Yaw Rate")
    ax3.set_xlabel("|Steering input| (1.0 = full lock)")
    ax3.set_ylabel("|Yaw rate| (deg/s)")
    ax3.set_xlim(0, max(max_steer * 1.05, 0.1))
    ax3.legend(loc="upper left")

    # Panel 4: Suspension travel by wheel
    ax4 = axes[1, 1]
    positions = np.arange(len(WHEELS))
    width = 0.38
    for i, (frame, name, color, m) in enumerate(zip(frames, names, colors, metrics)):
        offset = (i - 0.5) * width
        peaks = [m["susp_peak_by_wheel"][w] for w in WHEELS]
        p99 = [float(np.percentile(frame[f"susp_travel_norm_{w}"], 99)) for w in WHEELS]
        ax4.bar(positions + offset, peaks, width, color=color, alpha=0.75, label=f"{name} peak")
        ax4.scatter(positions + offset, p99, color="black", marker="_", s=250, zorder=3,
                    label="99th percentile" if i == 0 else None)
        for x, peak in zip(positions + offset, peaks):
            ax4.text(x, peak + 0.01, f"{peak:.2f}", ha="center", va="bottom", fontsize=8)
    ax4.axhline(BOTTOM_OUT_LEVEL, color="red", linestyle="--", linewidth=1.5, label=f"bottom-out ({BOTTOM_OUT_LEVEL})")
    ax4.set_xticks(positions)
    ax4.set_xticklabels([w.upper() for w in WHEELS])
    ax4.set_ylim(0, 1.1)
    ax4.set_title("4. Suspension Travel by Corner (1.0 = bottomed)")
    ax4.set_xlabel("Wheel")
    ax4.set_ylabel("Normalized travel")
    ax4.legend(loc="lower right", fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"FH5 Telemetry Comparison: {baseline_name} vs {tuned_name}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def analyze(baseline_path, tuned_path, output_path=DEFAULT_OUTPUT, corner_threshold_g=0.5):
    baseline = load_telemetry(baseline_path)
    tuned = load_telemetry(tuned_path)
    base_metrics = compute_metrics(baseline, corner_threshold_g)
    tuned_metrics = compute_metrics(tuned, corner_threshold_g)

    baseline_name, tuned_name = "Baseline", "Tuned"
    print(format_summary(base_metrics, tuned_metrics, baseline_name, tuned_name))
    saved = save_comparison_figure(
        baseline, tuned, base_metrics, tuned_metrics, output_path,
        baseline_name, tuned_name, corner_threshold_g,
    )
    print(f"\nSaved comparison figure -> {saved}")
    return base_metrics, tuned_metrics


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        prog="analyze_run.py",
        description="Compare baseline vs tuned FH5 telemetry CSVs.",
        add_help=add_help,
    )
    parser.add_argument("--baseline", type=Path, required=True, help="CSV recorded with the baseline setup.")
    parser.add_argument("--tuned", type=Path, required=True, help="CSV recorded with the tuned setup.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to save the PNG (default output/telemetry_comparison.png).")
    parser.add_argument("--corner-threshold-g", type=float, default=0.5,
                        help="Minimum |lateral G| for a sample to count as cornering (default 0.5).")
    return parser


def run(args):
    try:
        analyze(args.baseline, args.tuned, args.output, args.corner_threshold_g)
    except (FileNotFoundError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
