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
from pathlib import WindowsPath
import os
import numpy as np
from Probs import LanguageModel
import math
import matplotlib.pyplot as plt

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
    parser.add_argument("train_file1", type=Path, help="location of the training corpus 1")
    parser.add_argument("train_file2", type=Path, help="location of the training corpus 2")

    parser.add_argument("prior_gen", type=float, nargs='?', help = "Prior probability for genuine email")

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




    my_test = 2

    if my_test == 1:
        args.train_file1 = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\gen_spam\\train\\gen")
        args.train_file2 = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\gen_spam\\train\\spam")
        gen_Path = "C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\gen_spam\\dev\\gen\\"
        spam_Path = "C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\gen_spam\\dev\\spam\\"
        all_gen = os.listdir(gen_Path)
        all_spam = os.listdir(spam_Path)
        args.test_files = []
        for item in all_gen:
            args.test_files.append(WindowsPath(gen_Path + item))
        for item in all_spam:
            args.test_files.append(WindowsPath(spam_Path + item))
        args.lexicon = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm/lexicons/words-10.txt")

    elif my_test == 2:
        args.train_file1 = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\english_spanish\\train\\en.1k")
        args.train_file2 = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\english_spanish\\train\\sp.1k")
        gen_Path = "C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\english_spanish\\dev\\english\\length-20\\"
        spam_Path = "C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm\\english_spanish\\dev\\spanish\\length-20\\"
        all_gen = os.listdir(gen_Path)
        all_spam = os.listdir(spam_Path)
        args.test_files = []

        #for item in all_gen:
        #    args.test_files.append(WindowsPath(gen_Path + item))
        for item in all_spam:
            args.test_files.append(WindowsPath(spam_Path + item))
        args.lexicon = WindowsPath("C:\\Users\\99721\\OneDrive - Johns Hopkins University\\Fall 2019 course\\NLP\\Homework3\\hw-lm/lexicons/words-10.txt")

    return args

def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    model_path1 = get_model_filename(args.smoother, args.lexicon, args.train_file1)
    model_path2 = get_model_filename(args.smoother, args.lexicon, args.train_file2)
    if args.mode == TRAIN:
        #log.info("Training...")
        lm1 = LanguageModel.make(args.smoother, args.lexicon)
        lm2 = LanguageModel.make(args.smoother, args.lexicon)


        lm1.set_vocab_size(args.train_file1,args.train_file2)
        lm2.set_vocab_size(args.train_file1,args.train_file2)

        lm1.train(args.train_file1)
        lm2.train(args.train_file2)


        lm1.save(destination=model_path1)
        lm2.save(destination=model_path2)
    elif args.mode == TEST:

        #log.info("Testing...")
        lm1 = LanguageModel.load(model_path1)
        lm2 = LanguageModel.load(model_path2)

        # We use natural log for our internal computations and that's
        # the kind of log-probability that fileLogProb returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        #log.info("Printing file log-likelihoods.")
        num_gen = 0
        num_spam = 0

        total_log_prob1 = 0
        total_log_prob2 = 0

        accuracy = []
        length = []
        info = []
        for test_file in args.test_files:
            '''
            log_prob1 = lm1.file_log_prob(test_file) / np.log(2)
            log_prob2 = lm2.file_log_prob(test_file) / np.log(2)
            posteriori1 = 2**log_prob1 * args.prior_gen
            posteriori2 = 2**log_prob2 * (1-args.prior_gen)
            '''
            log_prob1 = lm1.file_log_prob(test_file)
            log_prob2 = lm2.file_log_prob(test_file)
            posteriori1 = np.exp(log_prob1) * args.prior_gen
            posteriori2 = np.exp(log_prob2) * (1-args.prior_gen)

            log_posteriori1 = log_prob1 + math.log(args.prior_gen)
            log_posteriori2 = log_prob2 + math.log(1-args.prior_gen)

            test_names = test_file.name.split('.')

            if log_posteriori1 > log_posteriori2:
                name = 'gen'
                print(f"gen\t{test_file.name}")
                num_gen += 1
            else:
                name = 'spam'
                print(f"spam\t{test_file.name}")
                num_spam += 1

            accuracy.append(name == test_names[0])
            length.append(int(test_names[1]))
            info.append({'Result':name == test_names[0], 'Length':int(test_names[1])})
            total_log_prob1 += log_prob1
            total_log_prob2 += log_prob2


        info.sort(key=lambda x:x ['Length'])
        M = 20
        sep = [0]
        rate = []
        sep[0] = info[0]["Length"]
        k = 0
        right = 0
        for item in info:
            right += int(item["Result"])
            k += 1
            if k == M:
                rate.append(right/M)
                sep.append(item["Length"])
                k = 0
                right = 0
        mid = []
        for i in range(len(rate)):
            mid.append((sep[i]+sep[i+1])/2)
        print(mid)
        print(rate)


        plt.plot(mid,rate)
        plt.show()







        '''
        N = 20
        all_num = [0]*N
        right_num = [0]*N
        plt.hist(length, bins=N, range=(0,700))
        sep = np.linspace(0, 700, num=N+1, endpoint=True, retstep=False)
        for j, item in enumerate(length):
            for i in range(N):
                if item > sep[i] and item < sep[i + 1]:

                    all_num[i] += 1
                    if accuracy[j] == True:
                        right_num[i] += 1
        rate = [0]*N
        for i in range(N):
            rate[i] = right_num[i]/all_num[i]
        '''





        total_tokens1 = sum(lm1.num_tokens(test_file) for test_file in args.test_files)
        #print(f"Overall cross-entropy for gen model:\t{-total_log_prob1 / total_tokens1:.5f}")

        total_tokens2 = sum(lm2.num_tokens(test_file) for test_file in args.test_files)
        #print(f"Overall cross-entropy for spam model:\t{-total_log_prob2 / total_tokens2:.5f}")

        print(f"{num_gen} files were more probably gen (%.2f%%)"% (num_gen/len(args.test_files) * 100))
        print(f"{num_spam} files were more probably spam (%.2f%%)"% (num_spam/len(args.test_files) * 100))


    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()
