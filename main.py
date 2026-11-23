"""Main command-line menu for the FH5 aero telemetry optimizer."""

import argparse
import sys
from pathlib import Path

import analyze_run
import telemetry_logger
import tune_calculator
from telemetry_logger import DATA_DIR


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FH5 aero & suspension tuning toolkit: calculate, record, analyze.",
    )
    sub = parser.add_subparsers(dest="command", metavar="{calculate,record,analyze,demo}")
    sub.required = True

    sub.add_parser("calculate", parents=[tune_calculator.build_parser(add_help=False)],
                   help="Calculate a recommended aero/suspension tune.",
                   description="Calculate a recommended aero/suspension tune.")
    sub.add_parser("record", parents=[telemetry_logger.build_parser(add_help=False)],
                   help="Record live FH5 UDP telemetry to CSV (or --generate-mock).",
                   description="Record live FH5 UDP telemetry to CSV (or --generate-mock).")
    sub.add_parser("analyze", parents=[analyze_run.build_parser(add_help=False)],
                   help="Compare baseline vs tuned telemetry CSVs.",
                   description="Compare baseline vs tuned telemetry CSVs.")

    demo = sub.add_parser("demo", help="Run everything end to end with mock data.",
                          description="Generate mock telemetry, calculate a sample tune, then analyze and plot.")
    demo.add_argument("--laps", type=int, default=3, help="Laps per mock log (default 3).")
    demo.add_argument("--seed", type=int, default=5, help="Random seed for mock data (default 5).")
    demo.add_argument("--output", type=Path, default=analyze_run.DEFAULT_OUTPUT,
                      help="Where to save the comparison PNG.")
    return parser


def run_demo(args):
    from mock_telemetry import generate_mock_logs

    print("\n[1/3] Generating mock telemetry (stock vs high-downforce setup)")
    stock_path, tuned_path = generate_mock_logs(DATA_DIR, laps=args.laps, seed=args.seed)

    print("\n[2/3] Sample tune calculation (3,100 lb RWD, 52% front, technical track)")
    status = tune_calculator.main(["--demo"])
    if status != 0:
        return status

    print("\n[3/3] Analyzing baseline vs tuned telemetry")
    return analyze_run.main(["--baseline", str(stock_path), "--tuned", str(tuned_path),
                             "--output", str(args.output)])


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "calculate":
        return tune_calculator.run(args)
    if args.command == "record":
        return telemetry_logger.run(args)
    if args.command == "analyze":
        return analyze_run.run(args)
    return run_demo(args)


if __name__ == "__main__":
    sys.exit(main())
