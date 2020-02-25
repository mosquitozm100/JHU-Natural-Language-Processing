#!/usr/bin/env python3
"""
findsim.py
JHU NLP HW3
Name: ______________
Email: ________@jhu.edu
Term: Fall 2019
"""
import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional # Type annotations.

try:
    # Numpy is your friend. Not *using* it will make your program so slow.
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import numpy as np
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.


def read_into_data_structure(vector_file: Path) -> Any:  # 'Any' matches any type.
    """Parse the file into memory in some usable, efficient format."""
    log.info("Reading data into usable format...")
    log.info("- done.")


def do_work(
    data_structure: Any, word1: str, word2: Optional[str], word3: Optional[str]
) -> List[str]:
    """Perform your actual computation here."""
    log.info("Doing real work...")
    # Make sure to sort by *decreasing* similarity.
    log.info("- done.")
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if len([x for x in (args.minus, args.plus) if x is not None]) == 1:
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args


def main():
    args = parse_args()
    data = read_into_data_structure(args.embeddings)
    words = do_work(data, args.word, args.minus, args.plus)
    print(" ".join(words))


if __name__ == "__main__":
    main()

