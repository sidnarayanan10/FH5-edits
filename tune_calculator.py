"""Physics-based aero and suspension tune calculator for Forza Horizon 5."""

import argparse
import re
import sys

KG_PER_LB = 0.45359237
AUTO_LBS_THRESHOLD = 1800.0
SPRING_AERO_COMPENSATION = 0.60
RIDE_HEIGHT_AERO_RAISE = 0.10
MIN_AERO_FRACTION = 0.25
MAX_AERO_FRACTION = 0.75
DEFAULT_DAMPING_RANGE = (1.0, 20.0)

DRIVETRAINS = ("RWD", "AWD", "FWD")
PROFILES = ("technical", "high_speed")

PROFILES_DATA = {
    "technical": {
        "description": "technical track - maximum agility / rotation",
        "aero_front_offset": +0.02,
        "arb_front_shift": -0.04,
        "arb_rear_shift": +0.05,
        "spring_front_shift": -0.02,
        "spring_rear_shift": +0.02,
        "ride_height_position": 0.15,
        "rake": 0.10,
        "camber_front": -1.8,
        "camber_rear": -1.2,
        "toe_front": -0.2,
        "toe_rear": +0.1,
        "bump_ratio_front": 0.65,
        "bump_ratio_rear": 0.60,
    },
    "high_speed": {
        "description": "high-speed track - maximum stability",
        "aero_front_offset": -0.03,
        "arb_front_shift": +0.02,
        "arb_rear_shift": -0.02,
        "spring_front_shift": +0.04,
        "spring_rear_shift": +0.04,
        "ride_height_position": 0.05,
        "rake": 0.15,
        "camber_front": -1.4,
        "camber_rear": -0.9,
        "toe_front": 0.0,
        "toe_rear": +0.2,
        "bump_ratio_front": 0.55,
        "bump_ratio_rear": 0.55,
    },
}

DRIVETRAIN_DATA = {
    "RWD": {"aero_front_offset": -0.01, "arb_front_shift": 0.00, "arb_rear_shift": -0.03,
            "camber_front_delta": 0.0, "camber_rear_delta": +0.2, "toe_rear_delta": +0.1},
    "AWD": {"aero_front_offset": 0.00, "arb_front_shift": -0.02, "arb_rear_shift": 0.00,
            "camber_front_delta": 0.0, "camber_rear_delta": 0.0, "toe_rear_delta": 0.0},
    "FWD": {"aero_front_offset": +0.02, "arb_front_shift": -0.03, "arb_rear_shift": +0.06,
            "camber_front_delta": +0.2, "camber_rear_delta": 0.0, "toe_rear_delta": 0.0},
}

DEMO_DEFAULTS = {
    "weight": "3100lbs",
    "front_weight": 52.0,
    "drivetrain": "RWD",
    "front_aero": (80.0, 260.0),
    "rear_aero": (120.0, 380.0),
    "front_arb": (1.0, 65.0),
    "rear_arb": (1.0, 65.0),
    "front_springs": (150.0, 450.0),
    "rear_springs": (140.0, 430.0),
    "ride_height": (3.5, 7.0),
    "profile": "technical",
}

WEIGHT_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*$")
LB_NAMES = {"lb", "lbs", "pound", "pounds"}
KG_NAMES = {"kg", "kgs", "kilo", "kilos", "kilogram", "kilograms"}


def clamp(value, low, high):
    return max(low, min(high, value))


def slider_value(slider_range, position):
    """Pick a value along (min, max) using a 0.0 to 1.0 position."""
    low, high = slider_range
    return low + (high - low) * clamp(position, 0.0, 1.0)


def slider_position(slider_range, value):
    """Find where a value sits between (min, max) as a 0.0 to 1.0 fraction."""
    low, high = slider_range
    if high == low:
        return 0.0
    return (value - low) / (high - low)


def parse_weight(text, unit="auto"):
    """Return (weight_value, unit_str) where unit_str is 'lb' or 'kg'."""
    match = WEIGHT_PATTERN.match(text.replace(",", ""))
    if not match:
        raise ValueError(f"could not understand weight {text!r} (try '3100', '3100lbs' or '1406kg')")

    value = float(match.group(1))
    if value <= 0:
        raise ValueError("weight must be positive")

    suffix = match.group(2).lower()
    if suffix in LB_NAMES:
        return value, "lb"
    if suffix in KG_NAMES:
        return value, "kg"
    if suffix:
        raise ValueError(f"unknown weight unit {suffix!r} (use lb/lbs or kg)")

    unit = unit.lower()
    if unit in LB_NAMES:
        return value, "lb"
    if unit in KG_NAMES:
        return value, "kg"
    return value, ("lb" if value >= AUTO_LBS_THRESHOLD else "kg")


def parse_front_weight(value):
    pct = value * 100.0 if 0.0 < value <= 1.0 else value
    if not 25.0 <= pct <= 75.0:
        raise ValueError(f"front weight distribution {value} looks wrong (expected roughly 35-65 %)")
    return pct


def solve_aero(front_range, rear_range, front_fraction):
    """Maximize downforce while keeping the target front/rear balance."""
    phi = clamp(front_fraction, MIN_AERO_FRACTION, MAX_AERO_FRACTION)
    total = min(front_range[1] / phi, rear_range[1] / (1.0 - phi))
    front = clamp(phi * total, front_range[0], front_range[1])
    rear = clamp((1.0 - phi) * total, rear_range[0], rear_range[1])
    return front, rear


def calculate_tune(spec):
    """Compute all recommended slider values from a car spec dictionary."""
    profile = PROFILES_DATA[spec["profile"]]
    drive = DRIVETRAIN_DATA[spec["drivetrain"]]
    notes = []

    weight_val, weight_unit = spec["weight"]
    front_frac = spec["front_weight_pct"] / 100.0
    rear_frac = 1.0 - front_frac

    # 1. Aerodynamics
    target_front_frac = clamp(
        front_frac + profile["aero_front_offset"] + drive["aero_front_offset"],
        MIN_AERO_FRACTION,
        MAX_AERO_FRACTION,
    )
    front_aero, rear_aero = solve_aero(spec["front_aero"], spec["rear_aero"], target_front_frac)
    total_aero = front_aero + rear_aero
    actual_front_frac = front_aero / total_aero if total_aero > 0 else target_front_frac

    if abs(actual_front_frac - target_front_frac) > 0.01:
        notes.append(
            f"Aero slider limits prevent the ideal {target_front_frac:.0%} front split; "
            f"the closest achievable is {actual_front_frac:.0%}."
        )

    # Ratio of aero downforce to static car weight on each axle
    rho_front = front_aero / (weight_val * front_frac)
    rho_rear = rear_aero / (weight_val * rear_frac)
    rho_total = total_aero / weight_val

    # 2. Anti-roll bars
    front_arb = slider_value(spec["front_arb"], front_frac + profile["arb_front_shift"] + drive["arb_front_shift"])
    rear_arb = slider_value(spec["rear_arb"], rear_frac + profile["arb_rear_shift"] + drive["arb_rear_shift"])

    # 3. Springs (stiffened so downforce doesn't bottom out the car)
    base_front_spring = slider_value(spec["front_springs"], front_frac + profile["spring_front_shift"])
    base_rear_spring = slider_value(spec["rear_springs"], rear_frac + profile["spring_rear_shift"])

    front_spring = clamp(
        base_front_spring * (1.0 + SPRING_AERO_COMPENSATION * rho_front),
        spec["front_springs"][0],
        spec["front_springs"][1],
    )
    rear_spring = clamp(
        base_rear_spring * (1.0 + SPRING_AERO_COMPENSATION * rho_rear),
        spec["rear_springs"][0],
        spec["rear_springs"][1],
    )

    for label, val, s_range in (
        ("front", front_spring, spec["front_springs"]),
        ("rear", rear_spring, spec["rear_springs"]),
    ):
        if s_range[1] > s_range[0] and val >= s_range[1] - 1e-9:
            notes.append(
                f"{label.capitalize()} springs hit the slider maximum while compensating for downforce; "
                "consider raising ride height or lowering aero if you still bottom out."
            )

    # 4. Ride height
    aero_raise = RIDE_HEIGHT_AERO_RAISE * min(rho_total, 1.0)
    ride_front = slider_value(spec["ride_height"], profile["ride_height_position"] + aero_raise)
    ride_rear = slider_value(spec["ride_height"], profile["ride_height_position"] + aero_raise + profile["rake"])

    # 5. Alignment (camber & toe)
    camber_extra = -0.3 * min(rho_total, 1.0)
    camber_front = profile["camber_front"] + camber_extra + drive["camber_front_delta"]
    camber_rear = profile["camber_rear"] + camber_extra + drive["camber_rear_delta"]
    toe_front = profile["toe_front"]
    toe_rear = profile["toe_rear"] + drive["toe_rear_delta"]

    # 6. Damping (rebound & bump)
    damping_range = spec["damping"]
    front_spring_pos = slider_position(spec["front_springs"], front_spring)
    rebound_front = slider_value(damping_range, 0.30 + 0.40 * front_spring_pos)
    bump_front = clamp(
        rebound_front * (profile["bump_ratio_front"] + 0.20 * min(rho_front, 0.5)),
        damping_range[0],
        damping_range[1],
    )

    rear_spring_pos = slider_position(spec["rear_springs"], rear_spring)
    rebound_rear = slider_value(damping_range, 0.30 + 0.40 * rear_spring_pos)
    bump_rear = clamp(
        rebound_rear * (profile["bump_ratio_rear"] + 0.20 * min(rho_rear, 0.5)),
        damping_range[0],
        damping_range[1],
    )

    if max(rho_front, rho_rear) > 0.35:
        notes.append("Downforce is high relative to axle weight: watch suspension travel on the telemetry.")

    return {
        "front_aero": round(front_aero, 1),
        "rear_aero": round(rear_aero, 1),
        "total_downforce": round(front_aero, 1) + round(rear_aero, 1),
        "actual_front_aero_fraction": actual_front_frac,
        "front_arb": round(front_arb, 1),
        "rear_arb": round(rear_arb, 1),
        "front_spring": round(front_spring, 1),
        "rear_spring": round(rear_spring, 1),
        "rho_front": rho_front,
        "rho_rear": rho_rear,
        "ride_height_front": round(ride_front, 1),
        "ride_height_rear": round(ride_rear, 1),
        "camber_front": round(camber_front, 1),
        "camber_rear": round(camber_rear, 1),
        "toe_front": round(toe_front, 1),
        "toe_rear": round(toe_rear, 1),
        "rebound_front": round(rebound_front, 1),
        "rebound_rear": round(rebound_rear, 1),
        "bump_front": round(bump_front, 1),
        "bump_rear": round(bump_rear, 1),
        "notes": notes,
    }


def format_report(spec, result):
    profile = PROFILES_DATA[spec["profile"]]
    weight_val, unit = spec["weight"]
    lb = weight_val if unit == "lb" else weight_val / KG_PER_LB
    kg = weight_val * KG_PER_LB if unit == "lb" else weight_val
    rho_f = result["rho_front"]
    rho_r = result["rho_rear"]

    def row(label, front, rear, suffix=""):
        return f"  {label:<22}{front:>9.1f}{suffix:<3} {rear:>9.1f}{suffix}"

    def ratio(num, den):
        return num / den if den else 0.0

    lines = [
        "=" * 60,
        "FH5 RECOMMENDED TUNE",
        "=" * 60,
        f"Car      : {lb:,.0f} lb = {kg:,.0f} kg, {spec['front_weight_pct']:.1f}% front, {spec['drivetrain']}",
        f"Profile  : {profile['description']}",
        "",
        f"  {'':<22}{'FRONT':>9}{'':<3} {'REAR':>9}",
        "AERODYNAMICS",
        row("Downforce", result["front_aero"], result["rear_aero"], f" {unit}"[:3]),
        f"  {'Total downforce':<22}{result['total_downforce']:>9.1f} {unit}"
        f"   (front share {result['actual_front_aero_fraction']:.1%} vs "
        f"{spec['front_weight_pct']:.1f}% front weight)",
        "",
        "ANTI-ROLL BARS & SPRINGS",
        row("Anti-roll bars", result["front_arb"], result["rear_arb"]),
        row("Springs", result["front_spring"], result["rear_spring"]),
        f"  Aero load / axle weight  front {rho_f:.2f}, rear {rho_r:.2f}  "
        f"(springs stiffened {SPRING_AERO_COMPENSATION * rho_f:.0%} / {SPRING_AERO_COMPENSATION * rho_r:.0%})",
        row("Ride height", result["ride_height_front"], result["ride_height_rear"]),
        "",
        "ALIGNMENT",
        row("Camber (deg)", result["camber_front"], result["camber_rear"]),
        row("Toe (deg, +in/-out)", result["toe_front"], result["toe_rear"]),
        "",
        "DAMPING",
        row("Rebound", result["rebound_front"], result["rebound_rear"]),
        row("Bump", result["bump_front"], result["bump_rear"]),
        f"  {'Bump/rebound ratio':<22}{ratio(result['bump_front'], result['rebound_front']):>9.2f}{'':<3} "
        f"{ratio(result['bump_rear'], result['rebound_rear']):>9.2f}",
    ]
    if result["notes"]:
        lines += ["", "NOTES"] + [f"  - {note}" for note in result["notes"]]
    lines.append("=" * 60)
    return "\n".join(lines)


def build_parser(add_help=True):
    parser = argparse.ArgumentParser(
        prog="tune_calculator.py",
        description="Calculate a physics-based FH5 aero/suspension tune for cornering downforce and agility.",
        add_help=add_help,
    )
    parser.add_argument("--demo", action="store_true",
                        help="Use a realistic demo car (3,100 lb RWD, 52%% front). Other options override it.")
    parser.add_argument("--weight", type=str, help="Car weight, e.g. 3100, 3100lbs or 1406kg.")
    parser.add_argument("--unit", choices=("auto", "lbs", "kg"), default="auto",
                        help="Unit for --weight when it has no suffix (auto: >=1800 means lbs).")
    parser.add_argument("--front-weight", type=float, help="Front weight distribution in percent, e.g. 53.")
    parser.add_argument("--drivetrain", type=str.upper, choices=DRIVETRAINS, help="RWD, AWD or FWD.")
    parser.add_argument("--profile", choices=PROFILES,
                        help="technical = agility/rotation, high_speed = stability (default technical).")

    slider = dict(nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--front-aero", **slider, help="Front aero slider min/max.")
    parser.add_argument("--rear-aero", **slider, help="Rear aero slider min/max.")
    parser.add_argument("--front-arb", **slider, help="Front anti-roll bar min/max.")
    parser.add_argument("--rear-arb", **slider, help="Rear anti-roll bar min/max.")
    parser.add_argument("--front-springs", **slider, help="Front spring rate min/max.")
    parser.add_argument("--rear-springs", **slider, help="Rear spring rate min/max.")
    parser.add_argument("--ride-height", **slider, help="Ride height min/max.")
    parser.add_argument("--damping-range", **slider, default=None,
                        help="Rebound/bump slider min/max (default 1 20).")
    return parser


def spec_from_args(args):
    required = ["weight", "front_weight", "drivetrain", "front_aero", "rear_aero", "front_arb",
                "rear_arb", "front_springs", "rear_springs", "ride_height"]
    values = {}
    missing = []
    for name in required + ["profile"]:
        given = getattr(args, name, None)
        if given is None and args.demo:
            given = DEMO_DEFAULTS[name]
        if given is None and name != "profile":
            missing.append("--" + name.replace("_", "-"))
        values[name] = given

    if missing:
        raise ValueError(f"missing required option(s): {', '.join(missing)} (or use --demo)")

    def make_range(key):
        low, high = float(values[key][0]), float(values[key][1])
        if low > high:
            raise ValueError(f"{key}: min ({low}) is greater than max ({high})")
        return (low, high)

    damping = getattr(args, "damping_range", None) or DEFAULT_DAMPING_RANGE
    if float(damping[0]) > float(damping[1]):
        raise ValueError(f"Damping: min ({damping[0]}) is greater than max ({damping[1]})")

    return {
        "weight": parse_weight(str(values["weight"]), args.unit),
        "front_weight_pct": parse_front_weight(float(values["front_weight"])),
        "drivetrain": str(values["drivetrain"]).upper(),
        "profile": str(values["profile"] or "technical"),
        "front_aero": make_range("front_aero"),
        "rear_aero": make_range("rear_aero"),
        "front_arb": make_range("front_arb"),
        "rear_arb": make_range("rear_arb"),
        "front_springs": make_range("front_springs"),
        "rear_springs": make_range("rear_springs"),
        "ride_height": make_range("ride_height"),
        "damping": (float(damping[0]), float(damping[1])),
    }


def run(args):
    try:
        spec = spec_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(spec, calculate_tune(spec)))
    return 0


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
