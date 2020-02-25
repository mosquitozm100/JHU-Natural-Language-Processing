import argparse
from collections import defaultdict
from collections import namedtuple
import math


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
        if tag not in VocTags:
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
def forward_backward(word_list, tag_dict, count_t, count_tt, count_tw):
    length = len(word_list)
    forward = defaultdict(int)
    backward = defaultdict(int)

    a_start = (0, '###')
    forward[a_start] = 1
    # 0 time step is excluded, n = length-1
    for i in range(1, length):
        for now_tag in tag_dict[word_list[i]]:
            for previous_tag in tag_dict[word_list[i - 1]]:
                # smooth: (c_tt + lambda)/(count_t + Voftag)
                p_tt = count_tt[(previous_tag, now_tag)] / count_t[previous_tag]
                # smooth: c_tw + lambda /(count_t + Vofwords)
                p_tw = count_tw[(word_list[i], now_tag)] / count_t[now_tag]
                p = p_tt * p_tw
                forward[(i, now_tag)] += forward[(i - 1, previous_tag)] * p

    b_start = (length - 1, '###')
    backward[b_start] = 1
    for i in range(length - 1, 0, -1):
        for now_tag in tag_dict[word_list[i]]:
            for next_tag in tag_dict[word_list[i - 1]]:
                p_tt = count_tt[(next_tag, now_tag)] / count_t[next_tag]
                p_tw = count_tw[(word_list[i], now_tag)] / count_t[now_tag]
                p = p_tt * p_tw
                backward[(i - 1, next_tag)] += backward[(i, now_tag)] * p

    print(forward[(length - 1, '###')])


def viterbi(smooth, Vw, Vt, word_list, tag_dict, count_t, count_tt, count_tw):
    length = len(word_list)
    miu_table = defaultdict(float)
    backpointer = defaultdict(str)

    a_start = (0, '###')
    miu_table[a_start] = math.log(1)
    # 0 time step is excluded, n = length-1
    for i in range(1, length):
        for now_tag in tag_dict[word_list[i]]:
            miu_table[(i, now_tag)] = -1000000
            for previous_tag in tag_dict[word_list[i - 1]]:
                # smooth: (c_tt + lambda)/(count_t + Voftag)
                p_tt = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (count_t[previous_tag] + Vt))
                # smooth: c_tw + lambda /(count_t + Vofwords)
                p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + Vw))
                # logp = logptt + logptw = log(ptt*ptw)
                p = p_tt + p_tw
                # log ti-1_ti = log ti-1 + log ti
                miu = miu_table[(i - 1, previous_tag)] + p
                # print(miu_table[(i - 1, previous_tag)])
                if miu > miu_table[(i, now_tag)]:
                    print(miu)
                    miu_table[(i, now_tag)] = miu
                    backpointer[(i, now_tag)] = previous_tag
                    print(previous_tag)
                    # print(backpointer[(i, now_tag)])
    predict = [' ' for i in range(0, length)]
    predict[length - 1] = '###'
    for i in range(length - 1, 0, -1):
        predict[i-1] = backpointer[(i, predict[i])]
    #print(predict)
    return predict


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
    print(f"Tagging accuracy (Viterbi decoding): {acc}%\t(known: {knownAcc}% novel: {novelAcc}%)")


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
    Vw = len(VocWords)
    Vt = len(VocTags)
    predict = viterbi(smooth, Vw, Vt, testWord, tag_dict, count_t, count_tt, count_tw)
    cal_acc(predict, testTag, testWord, VocWords)


if __name__ == "__main__":
    main()
