"""Command line interface entry point for msrd4-hypergraph-rgvq-distill."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=[
        "train_teacher",
        "export_distill",
        "train_student",
        "eval",
    ])
    return parser


def main() -> None:
    """Entry point for the CLI. Currently only parses arguments."""

    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
