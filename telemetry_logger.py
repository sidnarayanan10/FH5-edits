"""Forza Horizon 5 UDP telemetry logger.

Listens for 324-byte "Dash" packets from FH5 and saves useful driving data to a CSV file.
Can also generate fake telemetry logs with --generate-mock so you can test without the game.
"""

import argparse
import csv
import socket
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 20044

PACKET_SIZE = 324
GRAVITY_MPS2 = 9.80665
STEER_FULL_SCALE = 127.0
PEDAL_FULL_SCALE = 255.0

CSV_COLUMNS = [
    "timestamp_ms",
    "speed_mps",
    "yaw_rate",
    "pitch",
    "roll",
    "lat_accel_g",
    "long_accel_g",
    "steer_input",
    "throttle",
    "brake",
    "gear",
    "rpm",
    "tire_slip_ratio_fl",
    "tire_slip_ratio_fr",
    "tire_slip_ratio_rl",
    "tire_slip_ratio_rr",
    "susp_travel_norm_fl",
    "susp_travel_norm_fr",
    "susp_travel_norm_rl",
    "susp_travel_norm_rr",
]


def make_wheel_fields(prefix, code="f"):
    return [(f"{prefix}_{w}", code) for w in ("fl", "fr", "rl", "rr")]


def make_xyz_fields(prefix):
    return [(f"{prefix}_{axis}", "f") for axis in ("x", "y", "z")]


# Exact order of data inside FH5's 324-byte Dash UDP packet (little-endian).
# None means padding bytes we skip.
PACKET_LAYOUT = [
    # Sled section (bytes 0-231)
    ("is_race_on", "i"),
    ("timestamp_ms", "I"),
    ("engine_max_rpm", "f"),
    ("engine_idle_rpm", "f"),
    ("current_engine_rpm", "f"),
    *make_xyz_fields("accel"),
    *make_xyz_fields("velocity"),
    *make_xyz_fields("angular_velocity"),
    ("yaw", "f"),
    ("pitch", "f"),
    ("roll", "f"),
    *make_wheel_fields("susp_travel_norm"),
    *make_wheel_fields("tire_slip_ratio"),
    *make_wheel_fields("wheel_rotation_speed"),
    *make_wheel_fields("wheel_on_rumble_strip", "i"),
    *make_wheel_fields("wheel_in_puddle_depth"),
    *make_wheel_fields("surface_rumble"),
    *make_wheel_fields("tire_slip_angle"),
    *make_wheel_fields("tire_combined_slip"),
    *make_wheel_fields("susp_travel_m"),
    ("car_ordinal", "i"),
    ("car_class", "i"),
    ("car_performance_index", "i"),
    ("drivetrain_type", "i"),
    ("num_cylinders", "i"),
    # 12-byte Horizon-only gap (bytes 232-243)
    (None, "12x"),
    # Dash section (bytes 244-322)
    *make_xyz_fields("position"),
    ("speed", "f"),
    ("power", "f"),
    ("torque", "f"),
    *make_wheel_fields("tire_temp"),
    ("boost", "f"),
    ("fuel", "f"),
    ("distance_traveled", "f"),
    ("best_lap", "f"),
    ("last_lap", "f"),
    ("current_lap", "f"),
    ("current_race_time", "f"),
    ("lap_number", "H"),
    ("race_position", "B"),
    ("accel_pedal", "B"),
    ("brake_pedal", "B"),
    ("clutch", "B"),
    ("handbrake", "B"),
    ("gear", "B"),
    ("steer", "b"),
    ("normalized_driving_line", "b"),
    ("normalized_ai_brake_difference", "b"),
    # 1 trailing pad byte -> 324 bytes total
    (None, "x"),
]

PACKET_STRUCT = struct.Struct("<" + "".join(code for _, code in PACKET_LAYOUT))
FIELD_NAMES = [name for name, _ in PACKET_LAYOUT if name is not None]
assert PACKET_STRUCT.size == PACKET_SIZE


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_packet(data):
    """Unpack one 324-byte UDP packet into a dictionary, or return None if invalid."""
    if len(data) != PACKET_SIZE:
        return None
    try:
        values = PACKET_STRUCT.unpack(data)
    except struct.error:
        return None
    return dict(zip(FIELD_NAMES, values))


def to_log_row(raw):
    """Convert raw packet numbers into clean CSV units."""
    steer = max(-1.0, min(1.0, raw["steer"] / STEER_FULL_SCALE))
    row = {
        "timestamp_ms": int(raw["timestamp_ms"]),
        "speed_mps": raw["speed"],
        "yaw_rate": raw["angular_velocity_y"],
        "pitch": raw["pitch"],
        "roll": raw["roll"],
        "lat_accel_g": raw["accel_x"] / GRAVITY_MPS2,
        "long_accel_g": raw["accel_z"] / GRAVITY_MPS2,
        "steer_input": steer,
        "throttle": raw["accel_pedal"] / PEDAL_FULL_SCALE,
        "brake": raw["brake_pedal"] / PEDAL_FULL_SCALE,
        "gear": int(raw["gear"]),
        "rpm": raw["current_engine_rpm"],
        "tire_slip_ratio_fl": raw["tire_slip_ratio_fl"],
        "tire_slip_ratio_fr": raw["tire_slip_ratio_fr"],
        "tire_slip_ratio_rl": raw["tire_slip_ratio_rl"],
        "tire_slip_ratio_rr": raw["tire_slip_ratio_rr"],
        "susp_travel_norm_fl": raw["susp_travel_norm_fl"],
        "susp_travel_norm_fr": raw["susp_travel_norm_fr"],
        "susp_travel_norm_rl": raw["susp_travel_norm_rl"],
        "susp_travel_norm_rr": raw["susp_travel_norm_rr"],
    }
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}


def record(host=DEFAULT_HOST, port=DEFAULT_PORT, output_dir=DATA_DIR, duration_s=None, socket_timeout_s=1.0):
    """Listen on UDP and write active driving frames to a timestamped CSV."""
    ensure_dir(output_dir)
    csv_path = output_dir / f"telemetry_{datetime.now():%Y%m%d_%H%M%S}.csv"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError as exc:
        print(f"error: could not create UDP socket: {exc}", file=sys.stderr)
        return None

    with sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(socket_timeout_s)
        try:
            sock.bind((host, port))
        except OSError as exc:
            print(
                f"error: could not bind UDP {host}:{port} ({exc}). "
                "Is another telemetry tool already using this port?",
                file=sys.stderr,
            )
            return None

        print(f"Listening for FH5 Data Out on udp://{host}:{port}")
        print(f"Logging to {csv_path}")
        if duration_s is None:
            print("Press Ctrl+C to stop.")
        else:
            print(f"Recording for {duration_s:.0f}s (Ctrl+C to stop early).")

        rows_written = 0
        packets_seen = 0
        packets_ignored = 0
        started = time.monotonic()
        last_status = started

        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            try:
                while duration_s is None or (time.monotonic() - started) < duration_s:
                    try:
                        data, _ = sock.recvfrom(2048)
                    except socket.timeout:
                        if time.monotonic() - last_status >= 5.0:
                            print("  ...waiting for packets (is Data Out enabled and the race running?)")
                            last_status = time.monotonic()
                        continue
                    except ConnectionResetError:
                        continue

                    packets_seen += 1
                    raw = parse_packet(data)
                    if raw is None:
                        packets_ignored += 1
                        if packets_ignored == 1:
                            print(
                                f"  warning: ignoring a {len(data)}-byte packet "
                                f"(expected {PACKET_SIZE}). Set Data Out format to 'Dash'."
                            )
                        continue

                    if raw["is_race_on"] != 1:
                        continue

                    writer.writerow(to_log_row(raw))
                    rows_written += 1

                    if rows_written % 120 == 0:
                        handle.flush()
                    if time.monotonic() - last_status >= 5.0:
                        print(f"  {rows_written} samples logged ({raw['speed']:.1f} m/s)")
                        last_status = time.monotonic()
            except KeyboardInterrupt:
                print("\nStopping (Ctrl+C).")

    print(f"Done: {rows_written} samples from {packets_seen} packets -> {csv_path}")
    if rows_written == 0:
        print("note: no samples were logged. Check the Data Out IP/port and that a race is active.")
    return csv_path


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        prog="telemetry_logger.py",
        description="Log Forza Horizon 5 UDP Data Out telemetry to CSV, or generate mock logs.",
        add_help=add_help,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"IP to bind (default {DEFAULT_HOST}).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP port (default {DEFAULT_PORT}).")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR, help="Folder for CSV files (default ./data).")
    parser.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: until Ctrl+C).")
    parser.add_argument("--timeout", type=float, default=1.0, help="Socket timeout in seconds (default 1.0).")
    parser.add_argument(
        "--generate-mock",
        action="store_true",
        help="Write synthetic stock_laps.csv and tuned_laps.csv to the output folder instead of listening.",
    )
    parser.add_argument("--mock-laps", type=int, default=3, help="Laps per mock file (default 3).")
    parser.add_argument("--seed", type=int, default=5, help="Random seed for mock data (default 5).")
    return parser


def run(args):
    if args.generate_mock:
        from mock_telemetry import generate_mock_logs

        generate_mock_logs(args.output_dir, laps=args.mock_laps, seed=args.seed)
        return 0

    if not (0 < args.port < 65536):
        print(f"error: invalid port {args.port}", file=sys.stderr)
        return 2

    path = record(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        duration_s=args.duration,
        socket_timeout_s=args.timeout,
    )
    return 0 if path is not None else 1


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
