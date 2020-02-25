import argparse
from collections import defaultdict
from collections import namedtuple
from scipy.special import logsumexp
import numpy as np
import math


def lse(x):
    # log sum exp
    m = np.max(x)
    x -= m
    return m + np.log(np.sum(np.exp(x)))


# add OOV to tag_dict
def prob_generation(file):
    tag_dict = defaultdict(list)
    count_t = defaultdict(int)
    count_tt = defaultdict(int)
    count_tw = defaultdict(int)
    trainWord = []  # words in train set in order
    tag_L = []  # tags in train set in order
    VocWords = []  # vocabulary of words
    VocTags = []  # vocabulary of tags
    em = namedtuple('em', ['word', 'tag'])
    tr = namedtuple('tr', ['given', 'get'])
    with open(file, 'r') as f:
        s = f.readlines()
    for i in range(len(s)):
        sp = s[i].strip().split('/')
        word = sp[0]
        tag = sp[1]
        trainWord.append(word)
        tag_L.append(tag)
        if word not in VocWords:
            VocWords.append(word)
        # OOV can be all the tags except for ###
        if tag not in VocTags and tag != '###':
            VocTags.append(tag)
        if tag not in tag_dict[word]:
            tag_dict[word].append(tag)
        if i != 0:
            tt = tr(given=tag_L[i - 1], get=tag)
            count_tt[tt] += 1
            tw = em(word=word, tag=tag)
            count_tw[tw] += 1
            # c(t) is same for [0, n-1] and [1, n], just use 1 to n
            count_t[tag] += 1
    tag_dict['OOV'] = VocTags

    return VocWords, VocTags, trainWord, tag_dict, count_t, count_tt, count_tw


# transfer unknown words to OOV
def test_list(file, VocWords):
    word_L = []
    tag_L = []
    with open(file, 'r') as f:
        s = f.readlines()
    for pair in s:
        sp = pair.strip().split('/')
        word = sp[0]
        tag = sp[1]
        if word not in VocWords:
            word_L.append('OOV')
        else:
            word_L.append(word)
        tag_L.append(tag)
    return word_L, tag_L


# use lambda to smoothing
def forward_backward(smooth, Vw, Vt, word_list, tag_dict, count_t, count_tt, count_tw):
    length = len(word_list)
    forward = defaultdict(float)
    backward = defaultdict(float)
    predict = ['' for i in range(0, length)]
    predict[length - 1] = '###'
    if smooth == 0:
        Vw = 0
        Vt = 0
    a_start = (0, '###')
    forward[a_start] = math.log(1)
    # 0 time step is excluded, n = length-1
    for i in range(1, length):
        for now_tag in tag_dict[word_list[i]]:
            for previous_tag in tag_dict[word_list[i - 1]]:
                p_tt = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (count_t[previous_tag] + smooth * Vt))
                p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + smooth * Vw))
                p = p_tt + p_tw
                if (i, now_tag) not in forward.keys():
                    forward[(i, now_tag)] = forward[(i - 1, previous_tag)] + p
                else:
                    forward[(i, now_tag)] = lse([forward[(i, now_tag)], forward[(i - 1, previous_tag)] + p])
    Z = forward[(length - 1, '###')]
    b_start = (length - 1, '###')
    backward[b_start] = math.log(1)
    for i in range(length - 1, 0, -1):
        p_max = 1
        for now_tag in tag_dict[word_list[i]]:
            tw_bi = forward[(i, now_tag)] + backward[(i, now_tag)] - Z
            if p_max == 1 or p_max < tw_bi:
                p_max = tw_bi
                tag = now_tag
            for next_tag in tag_dict[word_list[i - 1]]:
                p_tt = math.log((count_tt[(next_tag, now_tag)] + smooth) / (count_t[next_tag] + Vt))
                p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + Vw))
                p = p_tt + p_tw
                if (i-1, next_tag) not in backward.keys():
                    backward[(i - 1, next_tag)] = backward[(i, now_tag)] + p
                else:
                    backward[(i - 1, next_tag)] = lse([backward[(i - 1, next_tag)], backward[(i, now_tag)] + p])
        predict[i] = tag

    return predict


def viterbi(smooth, Vw, Vt, word_list, tag_dict, count_t, count_tt, count_tw):
    length = len(word_list)
    miu_table = defaultdict(float)
    backpointer = defaultdict(str)
    if smooth == 0:
        Vw = 0
        Vt = 0
    a_start = (0, '###')
    miu_table[a_start] = math.log(1)
    # 0 time step is excluded, n = length-1
    for i in range(1, length):
        for now_tag in tag_dict[word_list[i]]:
            miu_table[(i, now_tag)] = float("-inf")
            for previous_tag in tag_dict[word_list[i - 1]]:
                # smooth: (c_tt + lambda)/(count_t + Voftag)
                p_tt = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (count_t[previous_tag] + smooth * Vt))
                # smooth: c_tw + lambda /(count_t + Vofwords)
                p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + smooth * Vw))
                # logp = logptt + logptw = log(ptt*ptw)
                p = p_tt + p_tw
                # log ti-1_ti = log ti-1 + log ti
                miu = miu_table[(i - 1, previous_tag)] + p
                # print(miu_table[(i - 1, previous_tag)])
                if miu > miu_table[(i, now_tag)]:
                    miu_table[(i, now_tag)] = miu
                    backpointer[(i, now_tag)] = previous_tag
                    # print(previous_tag)
                    # print(backpointer[(i, now_tag)])
    predict = [' ' for i in range(0, length)]
    predict[length - 1] = '###'
    for i in range(length - 1, 0, -1):
        predict[i - 1] = backpointer[(i, predict[i])]
    return predict


def cal_perplexity(smooth, Vw, Vt, word_list, test_tag, count_t, count_tt, count_tw):
    p = 0
    if smooth == 0:
        Vw = 0
        Vt = 0
    length = len(test_tag)
    for i in range(1, length):
        p_tt = math.log((count_tt[(test_tag[i - 1], test_tag[i])] + smooth) / (count_t[test_tag[i - 1]] + Vt))
        p_tw = math.log((count_tw[(word_list[i], test_tag[i])] + smooth) / (count_t[test_tag[i]] + Vw))
        p += p_tt + p_tw
    perplexity = round(math.exp(-(p / (length - 1))), 3)
    print(f"Model perplexity per tagged test word: {perplexity}")


def cal_acc(predict, test_tag, test_word, vocwords):
    cRight = 0
    cKnownRight = 0
    cNovelRight = 0
    cExclude = 0
    cKnownExclude = 0
    cNovelExclude = 0
    for i in range(len(test_tag)):
        if test_tag[i] != '###':
            cExclude += 1
            if predict[i] == test_tag[i]:
                cRight += 1
            if test_word[i] in vocwords:
                cKnownExclude += 1
                if predict[i] == test_tag[i]:
                    cKnownRight += 1
            else:
                cNovelExclude += 1
                if predict[i] == test_tag[i]:
                    cNovelRight += 1

    acc = round((cRight / cExclude) * 100, 2)
    knownAcc = round((cKnownRight / cKnownExclude) * 100, 2)
    if cNovelExclude == 0:
        novelAcc = '0.00'
    else:
        novelAcc = round((cNovelRight / cNovelExclude) * 100, 2)
    return acc, knownAcc, novelAcc


def main():
    parser = argparse.ArgumentParser(description='Process argument')
    parser.add_argument('train_file', type=str, help='a string of filepath')
    parser.add_argument('test_file', type=str, help='a string of filepath')
    ARGS = parser.parse_args()
    VocWords, VocTags, trainWord, tag_dict, count_t, count_tt, count_tw = prob_generation(ARGS.train_file)
    testWord, testTag = test_list(ARGS.test_file, VocWords)
    # predict = viterbi(testWord, tag_dict, count_t, count_tt, count_tw)
    # add lambda
    smooth = 1
    Vw = len(VocWords) + 1
    Vt = len(VocTags) + 1
    predictV = viterbi(smooth, Vw, Vt, testWord, tag_dict, count_t, count_tt, count_tw)
    predictP = forward_backward(smooth, Vw, Vt, testWord, tag_dict, count_t, count_tt, count_tw)
    cal_perplexity(smooth, Vw, Vt, testWord, testTag, count_t, count_tt, count_tw)
    acc_V, knownAcc_V, novelAcc_V = cal_acc(predictV, testTag, testWord, VocWords)
    print(f"Tagging accuracy (Viterbi decoding): {acc_V}%\t(known: {knownAcc_V}% novel: {novelAcc_V}%)")
    acc_P, knownAcc_P, novelAcc_P = cal_acc(predictP, testTag, testWord, VocWords)
    print(f"Tagging accuracy (posterior decoding): {acc_P}%\t(known: {knownAcc_P}% novel: {novelAcc_P}%)")
    f = open('test-output', 'w')
    for i in range(len(predictP)):
        f.write(testWord[i] + '/' + predictP[i] + '\n')
    f.close()


if __name__ == "__main__":
    main()
