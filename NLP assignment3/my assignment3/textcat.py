#!/usr/bin/env python3
"""
Computes the log probability of the sequence of tokens in file,
according to a trigram model.  The training source is specified by
the currently open corpus, and the smoothing method used by
prob() is polymorphic.
"""
import argparse
import logging
from pathlib import Path
import math

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


from Probs import LanguageModel

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, train_file: Path) -> Path:
    return Path(f"{smoother}_{lexicon.name}_{train_file.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "smoother",
        type=str,
        help="""Possible values: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the "1" in add1/backoff_add1 can be replaced with any real λ >= 0
   the "1" in loglinear1 can be replaced with any C >= 0 )
""",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word vector file; only used in the loglinear model",
    )
    parser.add_argument("train_file1", type=Path, help="location of the first training corpus")
    parser.add_argument("train_file2", type=Path, help="location of the second training corpus")
    parser.add_argument("prior_prob", type=float, nargs="?", help="prior probability of the first training corpus")
    parser.add_argument("test_files", type=Path, nargs="*")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()


    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and not args.test_files:
        parser.error("No test files specified.")

    return args

def print_name_cnt_CE(dict_log_prob):
    for key, value in dict_log_prob.items():
        print("{}\t{}\t{}".format(key,str(key).split('.')[1],value))

def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path1 = get_model_filename(args.smoother, args.lexicon, args.train_file1)
    model_path2 = get_model_filename(args.smoother, args.lexicon, args.train_file2)
    #print("test_file", args.test_files)
    #print("Current smoother = ", args.smoother)

    if args.mode == TRAIN:
        log.info("Training...")
        lm1 = LanguageModel.make(args.smoother, args.lexicon)
        lm2 = LanguageModel.make(args.smoother, args.lexicon)
        if args.smoother[:3] == "add":
            lm1.set_vocab_size(args.train_file1, args.train_file2)
            lm2.set_vocab_size(args.train_file1, args.train_file2)
        
        

        lm1.train(args.train_file1)
        lm2.train(args.train_file2)
        lm1.save(destination=model_path1)
        lm2.save(destination=model_path2)
    elif args.mode == TEST:
        log.info("Testing...")
        lm1 = LanguageModel.load(model_path1)
        lm2 = LanguageModel.load(model_path2)
        # We use natural log for our internal computations and that's
        # the kind of log-probability that fileLogProb returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        log.info("Printing file log-likelihoods.")
        total_log_prob1 = 0.0
        dict_log_prob1 = {}
        for test_file in args.test_files:
            log_prob1 = lm1.file_log_prob(test_file) / np.log(2)
            #print(f"{log_prob1:g}\t{test_file}")
            total_log_prob1 += log_prob1
            dict_log_prob1[test_file] = log_prob1
            

        total_tokens1 = sum(lm1.num_tokens(test_file) for test_file in args.test_files)
        #print(f"{args.train_file1.name} Overall cross-entropy:\t{-total_log_prob1 / total_tokens1:.5f}")

        total_log_prob2 = 0.0
        dict_log_prob2 = {}
        for test_file in args.test_files:
            log_prob2 = lm2.file_log_prob(test_file) / np.log(2)
            #print(f"{log_prob2:g}\t{test_file}")
            total_log_prob2 += log_prob2
            dict_log_prob2[test_file] = log_prob2

        total_tokens2 = sum(lm2.num_tokens(test_file) for test_file in args.test_files)
        #print(f"{args.train_file2.name} Overall cross-entropy:\t{-total_log_prob2 / total_tokens2:.5f}")
        #print(f"{args.train_file1.name} & {args.train_file2.name} Overall cross-entropy:\t{-(total_log_prob1 + total_log_prob2) / (total_tokens1 + total_tokens2):.5f}")

        cnt_on_len_right = {}       #3(e)
        cnt_on_len_wrong = {}       #3(e)
        cnt_right = cnt_wrong = 0
        cnt_file1 = cnt_file2 = 0
        for test_file in args.test_files:
            if dict_log_prob1[test_file] + (math.log(args.prior_prob,2)) > dict_log_prob2[test_file] + (math.log(1 - args.prior_prob,2)):
                print(args.train_file1.name.split('.')[0], end = '\t')
                print(test_file)
                cnt_file1 += 1
                if test_file.name.split('.')[0] == args.train_file1.name.split('.')[0]:
                    cnt_right += 1
                else:
                    cnt_wrong += 1            

            else:
                print(args.train_file2.name.split('.')[0], end = '\t')
                print(test_file)
                cnt_file2 += 1
                if test_file.name.split('.')[0] == args.train_file2.name.split('.')[0]:
                    cnt_right += 1
                else:
                    cnt_wrong += 1    
    
        print("{} files were more probably {} ({:.2%})".format(cnt_file1,args.train_file1.name,cnt_file1 / (cnt_file1 + cnt_file2)))
        print("{} files were more probably {} ({:.2%})".format(cnt_file2,args.train_file2.name,cnt_file2 / (cnt_file1 + cnt_file2)))

        #print("Accuracy is {:.2%}({}/{})".format((cnt_right / (cnt_right + cnt_wrong)), cnt_right, cnt_right + cnt_wrong))

        #print_name_cnt_CE(dict_log_prob1)
        #print_name_cnt_CE(dict_log_prob2)
    
    else:
        raise ValueError("Inappropriate mode of operation.")

if __name__ == "__main__":
    main()

