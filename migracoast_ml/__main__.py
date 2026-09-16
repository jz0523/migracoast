"""Command line entry point for the Migracoast ML pipeline."""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="migracoast_ml",
        description="Learned coastal flood risk pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Download every external data source.")
    fetch.add_argument("--force", action="store_true", help="Re-download existing files.")

    sub.add_parser("build", help="Build the training table from raw data.")
    sub.add_parser("train", help="Fit the model ladder and save the winner.")
    sub.add_parser("evaluate", help="Print cross-validation metrics.")
    sub.add_parser("baseline", help="Compare the rule engine against the model.")

    serve = sub.add_parser("serve", help="Run the prediction API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5001)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "fetch":
        from migracoast_ml.sources.fetch_all import fetch_all

        for name, path in fetch_all(force=args.force).items():
            print(f"{name}: {path}")
        return 0

    if args.command == "build":
        from migracoast_ml.dataset import build_training_table

        table = build_training_table()
        print(f"Wrote {len(table)} cells")
        return 0

    if args.command == "train":
        from migracoast_ml.models.train import train_and_save

        result = train_and_save()
        print(f"Selected {result['winner']}")
        return 0

    if args.command == "evaluate":
        from migracoast_ml.models.train import print_metrics

        print_metrics()
        return 0

    if args.command == "baseline":
        from migracoast_ml.baseline import run_comparison

        print(run_comparison())
        return 0

    if args.command == "serve":
        from migracoast_ml.serve import run

        run(host=args.host, port=args.port)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
