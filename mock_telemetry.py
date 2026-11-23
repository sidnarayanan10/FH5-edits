"""Creates fake Forza telemetry CSVs so you can test the charts without the game."""

import csv
import math
import numpy as np

from telemetry_logger import CSV_COLUMNS, GRAVITY_MPS2, ensure_dir

SAMPLE_RATE_HZ = 60
DISTANCE_STEP_M = 1.0
CURVATURE_SMOOTHING_M = 31
WHEELBASE_M = 2.7
MAX_ROAD_WHEEL_ANGLE_RAD = 0.25
GEAR_SPEEDS = [0.0, 14.0, 26.0, 38.0, 52.0, 66.0, 84.0]
WHEELS = ("fl", "fr", "rl", "rr")

STOCK_CAR = {
    "name": "stock",
    "mu0": 1.15,
    "k_aero": 2.0e-5,
    "understeer_k": 2.2e-4,
    "yaw_tau_s": 0.22,
    "roll_rad_per_g": 0.040,
    "corner_slip": 0.050,
    "drive_slip": 0.10,
    "susp_static": 0.45,
    "susp_aero": 3.0e-5,
    "susp_roll_per_g": 0.090,
    "top_speed_mps": 74.0,
    "accel_mps2": 7.0,
}

TUNED_CAR = {
    "name": "tuned",
    "mu0": 1.20,
    "k_aero": 9.0e-5,
    "understeer_k": 0.8e-4,
    "yaw_tau_s": 0.13,
    "roll_rad_per_g": 0.026,
    "corner_slip": 0.036,
    "drive_slip": 0.07,
    "susp_static": 0.42,
    "susp_aero": 5.8e-5,
    "susp_roll_per_g": 0.065,
    "top_speed_mps": 77.0,
    "accel_mps2": 7.5,
}

# Each tuple is (length_m, turn_radius_m, direction): radius 0 = straight, +1 = right, -1 = left
TRACK = [
    (500, 0, 1),
    (300, 300, +1),
    (250, 0, 1),
    (170, 90, -1),
    (130, 60, +1),
    (450, 0, 1),
    (80, 28, -1),
    (400, 0, 1),
    (350, 220, +1),
    (200, 150, -1),
    (300, 0, 1),
    (400, 400, -1),
    (200, 0, 1),
    (180, 120, +1),
    (200, 0, 1),
]


def build_curvature(track, step=DISTANCE_STEP_M):
    """Turn track segments into a smooth curve (1 / radius) every 1 meter."""
    pieces = []
    for length_m, radius_m, direction in track:
        count = max(1, int(round(length_m / step)))
        turn_sharpness = direction / radius_m if radius_m > 0 else 0.0
        pieces.append(np.full(count, turn_sharpness))
    kappa = np.concatenate(pieces)

    half = CURVATURE_SMOOTHING_M // 2
    wrapped = np.concatenate([kappa[-half:], kappa, kappa[:half]])
    window = np.ones(2 * half + 1) / (2 * half + 1)
    return np.convolve(wrapped, window, mode="valid")


def simulate_speed_profile(kappa, car, step=DISTANCE_STEP_M):
    """Find the fastest speed around the track given corner grip, braking, and engine power."""
    n = len(kappa)
    abs_kappa = np.abs(kappa)
    aero_term = GRAVITY_MPS2 * car["k_aero"]

    # Step 1: Maximum speed allowed in each corner
    speed_limit = np.full(n, car["top_speed_mps"])
    sharp_enough = (abs_kappa - aero_term) > 1e-6
    speed_limit[sharp_enough] = np.minimum(
        car["top_speed_mps"],
        np.sqrt(GRAVITY_MPS2 * car["mu0"] / (abs_kappa[sharp_enough] - aero_term)),
    )

    # Step 2: Slow down early enough before corners (backward pass) and speed up on straights (forward pass)
    speed = speed_limit.copy()
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            next_speed = speed[(i + 1) % n]
            brake_decel = 0.9 * GRAVITY_MPS2 * (car["mu0"] + car["k_aero"] * next_speed ** 2)
            speed[i] = min(speed[i], math.sqrt(next_speed ** 2 + 2.0 * brake_decel * step))

        for i in range(n):
            prev_speed = speed[i - 1]
            accel = max(0.5, car["accel_mps2"] * (1.0 - (prev_speed / car["top_speed_mps"]) ** 2))
            speed[i] = min(speed[i], math.sqrt(prev_speed ** 2 + 2.0 * accel * step))

    return speed


def smooth_noise(rng, n, sigma, window=5):
    raw = rng.normal(size=n + window - 1)
    smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
    return smoothed * math.sqrt(window) * sigma


def apply_lag(target, tau_s, dt):
    """Make a signal react gradually instead of jumping instantly."""
    alpha = dt / (tau_s + dt)
    out = np.empty_like(target)
    out[0] = target[0]
    for i in range(1, len(target)):
        out[i] = out[i - 1] + alpha * (target[i] - out[i - 1])
    return out


def drive_laps(car, laps, rng):
    """Resample the distance-based lap into 60 Hz time-based samples."""
    kappa = build_curvature(TRACK)
    base_speed = simulate_speed_profile(kappa, car)
    distance_grid = np.arange(len(kappa)) * DISTANCE_STEP_M

    all_speeds = []
    all_kappas = []
    for _ in range(laps):
        lap_speed = base_speed * rng.uniform(0.97, 1.0)
        time_at_meter = np.concatenate([[0.0], np.cumsum(DISTANCE_STEP_M / lap_speed)])
        sample_times = np.arange(0.0, time_at_meter[-1], 1.0 / SAMPLE_RATE_HZ)
        sample_distances = np.interp(sample_times, time_at_meter, np.arange(len(time_at_meter)) * DISTANCE_STEP_M)
        all_speeds.append(np.interp(sample_distances, distance_grid, lap_speed))
        all_kappas.append(np.interp(sample_distances, distance_grid, kappa))

    return np.concatenate(all_speeds), np.concatenate(all_kappas)


def add_kerb_hits(susp, n, laps, rng):
    """Add short suspension bumps to simulate hitting kerbs."""
    index = np.arange(n)
    for _ in range(5 * laps):
        wheel = WHEELS[int(rng.integers(len(WHEELS)))]
        center = int(rng.integers(n))
        bump_height = rng.uniform(0.08, 0.16)
        susp[wheel] += bump_height * np.exp(-0.5 * ((index - center) / 4.0) ** 2)


def synthesize_channels(car, laps, seed):
    """Build all 20 telemetry columns for a fake driving session."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / SAMPLE_RATE_HZ
    v, kappa = drive_laps(car, laps, rng)
    n = len(v)

    # Steering, yaw rate, and sideways G-force
    steer_cmd = kappa * WHEELBASE_M * (1.0 + car["understeer_k"] * v ** 2) / MAX_ROAD_WHEEL_ANGLE_RAD
    steer = np.clip(steer_cmd + smooth_noise(rng, n, 0.012), -1.0, 1.0)
    yaw_rate = apply_lag(v * kappa, car["yaw_tau_s"], dt) + smooth_noise(rng, n, 0.01)
    lat_g = v * yaw_rate / GRAVITY_MPS2 + smooth_noise(rng, n, 0.02)

    # Forward acceleration, pedals, gear, and RPM
    accel = np.convolve(np.gradient(v, dt), np.ones(5) / 5, mode="same")
    long_g = accel / GRAVITY_MPS2 + smooth_noise(rng, n, 0.02)
    brake = np.where(accel < -1.0, np.clip((-accel - 1.0) / 9.0, 0.0, 1.0), 0.0)
    throttle = np.where(
        accel > -1.0,
        np.clip(0.55 + 0.45 * v / car["top_speed_mps"] + accel / 4.0, 0.0, 1.0),
        0.0,
    )

    breakpoints = np.array(GEAR_SPEEDS)
    gear = np.clip(np.searchsorted(breakpoints, v, side="right"), 1, len(breakpoints) - 1)
    low_speed = breakpoints[gear - 1]
    high_speed = breakpoints[gear]
    rpm = 3500.0 + 4200.0 * np.clip((v - low_speed) / (high_speed - low_speed), 0.0, 1.0)

    # Pitch, roll, and tire slip (RWD: rear tires get engine wheelspin)
    pitch = -0.012 * long_g + smooth_noise(rng, n, 0.002)
    roll = -car["roll_rad_per_g"] * lat_g + smooth_noise(rng, n, 0.002)

    cornering_slip = car["corner_slip"] * np.abs(lat_g)
    wheelspin = car["drive_slip"] * throttle * np.clip(1.0 - v / 70.0, 0.0, 1.0)
    braking_slip = -0.07 * brake
    slip = {}
    for w in WHEELS:
        rear_wheel = w in ("rl", "rr")
        slip[w] = cornering_slip + braking_slip + (wheelspin if rear_wheel else 0.0) + rng.normal(0.0, 0.008, n)

    # Suspension travel (0.0 = extended, 1.0 = bottomed out)
    side_sign = {"fl": -1.0, "fr": 1.0, "rl": -1.0, "rr": 1.0}
    aero_share = {"fl": 0.9, "fr": 0.9, "rl": 1.1, "rr": 1.1}
    pitch_sign = {"fl": 1.0, "fr": 1.0, "rl": -1.0, "rr": -1.0}
    susp = {}
    for w in WHEELS:
        susp[w] = (
            car["susp_static"]
            + car["susp_aero"] * aero_share[w] * v ** 2
            - side_sign[w] * car["susp_roll_per_g"] * lat_g
            + pitch_sign[w] * 0.040 * (-long_g)
            + rng.normal(0.0, 0.004, n)
        )
    add_kerb_hits(susp, n, laps, rng)
    for w in WHEELS:
        susp[w] = np.clip(susp[w], 0.0, 1.0)

    timestamp_ms = (1_000_000 + np.arange(n) * 1000.0 / SAMPLE_RATE_HZ).round().astype(np.int64)

    channels = {
        "timestamp_ms": timestamp_ms,
        "speed_mps": v,
        "yaw_rate": yaw_rate,
        "pitch": pitch,
        "roll": roll,
        "lat_accel_g": lat_g,
        "long_accel_g": long_g,
        "steer_input": steer,
        "throttle": throttle,
        "brake": brake,
        "gear": gear.astype(np.int64),
        "rpm": rpm,
    }
    for w in WHEELS:
        channels[f"tire_slip_ratio_{w}"] = slip[w]
    for w in WHEELS:
        channels[f"susp_travel_norm_{w}"] = susp[w]
    return channels


def write_channels_csv(path, channels):
    columns = [channels[name] for name in CSV_COLUMNS]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in zip(*columns):
            writer.writerow(
                [int(val) if isinstance(val, (np.integer, int)) else f"{val:.5f}" for val in row]
            )


def generate_mock_logs(data_dir, laps=3, seed=5):
    ensure_dir(data_dir)
    paths = []
    for offset, (car, filename) in enumerate(((STOCK_CAR, "stock_laps.csv"), (TUNED_CAR, "tuned_laps.csv"))):
        channels = synthesize_channels(car, laps=laps, seed=seed + offset)
        path = data_dir / filename
        write_channels_csv(path, channels)
        print(f"Wrote {len(channels['speed_mps'])} samples ({laps} laps, {car['name']} setup) -> {path}")
        paths.append(path)
    return paths[0], paths[1]
