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
        if tag not in VocTags:
            VocTags.append(tag)
        if tag not in tag_dict[word]:
            tag_dict[word].append(tag)
        if i != 0:
            tt = tr(given=tag_L[i - 1], get=tag)
            count_tt[tt] += 1
            count_t[tag] += 1
        tw = em(word=word, tag=tag)
        count_tw[tw] += 1
            # c(t) is same for [0, n-1] and [1, n], just use 1 to n
            
    #count_t["###"] -= 1   #for the purpose to get same value with xhr
    VocTags.append("OOT")
    VocWords.append("OOV")
    tag_dict['OOV'] = VocTags

    return VocWords, VocTags, trainWord, tag_dict, count_t, count_tt, count_tw


# transfer unknown words to OOV
def test_list(file, VocWords, VocTags):
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
        if tag not in VocTags:
            tag_L.append("OOT")
        else:
            tag_L.append(tag)

    return word_L, tag_L


# handle raw data same way as test data without tag
def raw_list(file, VocWords):
    word_L = []
    with open(file, 'r') as f:
        s = f.readlines()
    for pair in s:
        sp = pair.strip().split('/')
        word = sp[0]
        if word not in VocWords:
            word_L.append('OOV')
        else:
            word_L.append(word)
    return word_L


def compute_init_A_B(smooth, Vw, Vt, VocWords, VocTags, count_t, count_tt, count_tw):
    em = namedtuple('em', ['word', 'tag'])
    tr = namedtuple('tr', ['given', 'get'])
    A = defaultdict(float)
    B = defaultdict(float)
    new_count_t = defaultdict(int)
    new_count_w = {}
    #new_count_w = defaultdict(int)
    #print("count_t", count_t)
    #print("count_tt", count_tt)
    #print("count_tw", count_tw)
    new_count_t['OOT'] = 0
    for tag in count_tt:
        new_count_t[tag[0]] += count_tt[tag]
    for previous_tag in VocTags:
        for now_tag in VocTags:
            A[tr(previous_tag, now_tag)] = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (new_count_t[previous_tag] + smooth * Vt))
    #count_tw[("###","###")] += 1
    new_count_w['OOT'] = 0
    for tag in count_tw:
        new_count_w[tag[1]] = new_count_w.get(tag[1], 0) + count_tw[tag]
    #print("count_t", new_count_t, " count_w", new_count_w)
    for tag in VocTags:
        for word in VocWords:
            B[em(word, tag)] = math.log((count_tw[(word, tag)] + smooth) / (new_count_w[tag] + smooth * Vw))
    return A, B


# use lambda to smoothing
def forward_backward(smooth, Vw, Vt, word_list, tag_dict, A, B, count_t, count_tt, count_tw, VocTags, VocWords):
    length = len(word_list)
    new_count_t = defaultdict(int)
    new_count_tt = defaultdict(int)
    new_count_tw = defaultdict(int)
    forward = defaultdict(float)
    backward = defaultdict(float)
    prob_for_perp = 0
    if smooth == 0:
        Vw = 0
        Vt = 0
    a_start = (0, '###')
    forward[a_start] = math.log(1)
    # 0 time step is excluded, n = length-1
    for i in range(1, length):
        for now_tag in tag_dict[word_list[i]]:
            for previous_tag in tag_dict[word_list[i - 1]]:
                #p_tt = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (count_t[previous_tag] + smooth * Vt))
                #p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + smooth * Vw))
                #p = p_tt + p_tw
                p = A[(previous_tag, now_tag)] + B[(word_list[i], now_tag)]
                if (i, now_tag) not in forward.keys():
                    forward[(i, now_tag)] = lse([-100, forward[(i - 1, previous_tag)] + p])
                else:
                    forward[(i, now_tag)] = lse([forward[(i, now_tag)], forward[(i - 1, previous_tag)] + p])
            if i == len(word_list) - 1:
                prob_for_perp += forward[(i, now_tag)]
    Z = forward[(length - 1, '###')]
    #print("Z:",Z)
    b_start = (length - 1, '###')
    backward[b_start] = math.log(1)
    for i in range(length - 1, 0, -1):
        p_max = 1
        for now_tag in tag_dict[word_list[i]]:
            tw_bi = forward[(i, now_tag)] + backward[(i, now_tag)] - Z
            new_count_tw[(word_list[i],now_tag)] += np.exp(tw_bi)
            if p_max == 1 or p_max < tw_bi:
                p_max = tw_bi
                tag = now_tag
            for prev_tag in tag_dict[word_list[i - 1]]:
                #p_tt = math.log((count_tt[(next_tag, now_tag)] + smooth) / (count_t[next_tag] + Vt))
                #p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + Vw))
                #p = p_tt + p_tw
                p = A[(prev_tag, now_tag)] + B[(word_list[i], now_tag)]
                if (i-1, prev_tag) not in backward.keys():
                    backward[(i - 1, prev_tag)] = lse([-100, backward[(i, now_tag)] + p])
                else:
                    backward[(i - 1, prev_tag)] = lse([backward[(i - 1, prev_tag)], backward[(i, now_tag)] + p])
                #print("alpha:", forward[(prev_tag, i - 1)], "beta:", backward[(now_tag, i)])
                #print("now_tag:", now_tag, "prev_tag", prev_tag)
                Xi = forward[(i - 1, prev_tag)] + p + backward[(i, now_tag)] - Z
                #print("new_A", new_A)
                new_count_tt[(prev_tag, now_tag)] = count_tt[(prev_tag, now_tag)] + np.exp(Xi)
    
    new_count_tw[("###","###")] = count_tw[("###", "###")] + 1
    A, B = compute_init_A_B(smooth, Vw, Vt, VocWords, VocTags, count_t, new_count_tt, new_count_tw)

    perplexity = np.exp(-prob_for_perp/(len(word_list)-1))
    print('Model perplexity per untagged raw word: {:.3f}'.format(perplexity)) 
    
    return A, B, new_count_tt, new_count_tw



def viterbi(smooth, Vw, Vt, word_list, tag_dict, A, B):
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
                #p_tt = math.log((count_tt[(previous_tag, now_tag)] + smooth) / (count_t[previous_tag] + smooth * Vt))
                # smooth: c_tw + lambda /(count_t + Vofwords)
                #p_tw = math.log((count_tw[(word_list[i], now_tag)] + smooth) / (count_t[now_tag] + smooth * Vw))
                # logp = logptt + logptw = log(ptt*ptw)
                #p = p_tt + p_tw
                p = A[(previous_tag, now_tag)] + B[(word_list[i], now_tag)]
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
    parser.add_argument('raw_train_file', type=str, help='a string of filepath')
    ARGS = parser.parse_args()
    VocWords, VocTags, trainWord, tag_dict, count_t, count_tt, count_tw = prob_generation(ARGS.train_file)
    testWord, testTag = test_list(ARGS.test_file, VocWords, VocTags)
    rawWord = raw_list(ARGS.raw_train_file, VocWords)
    # predict = viterbi(testWord, tag_dict, count_t, count_tt, count_tw)
    # add lambda
    smooth = 1
    Vw = len(VocWords)
    Vt = len(VocTags)
    

    #trying to do EM
    A, B = compute_init_A_B(smooth, Vw, Vt, VocWords, VocTags, count_t, count_tt, count_tw)
    #print("A:", A)
    #print("B:", B)
    #predictV = viterbi(smooth, Vw, Vt, testWord, tag_dict, A, B)
    #acc_V, knownAcc_V, novelAcc_V = cal_acc(predictV, testTag, testWord, VocWords)
    #print(f"Tagging accuracy (Viterbi decoding): {acc_V}%\t(known: {knownAcc_V}% novel: {novelAcc_V}%)")
    for iter in range(10):
        predictV = viterbi(smooth, Vw, Vt, testWord, tag_dict, A, B)
        acc_V, knownAcc_V, novelAcc_V = cal_acc(predictV, testTag, testWord, VocWords)  
        seenAcc_V = 0
        cal_perplexity(smooth, Vw, Vt, testWord, testTag, count_t, count_tt, count_tw)
        print(f"Tagging accuracy (Viterbi decoding): {acc_V}%\t(known: {knownAcc_V}% seen: {seenAcc_V}% novel: {novelAcc_V}%)")
        print('Iteration', iter, ":", end = ' ')
        A, B, count_tt, count_tw= forward_backward(smooth, Vw, Vt, rawWord, tag_dict, A, B, count_t, count_tt, count_tw, VocTags, VocWords)



    #predictV = viterbi(smooth, Vw, Vt, testWord, tag_dict, count_t, count_tt, count_tw)
    #predictP = forward_backward(smooth, Vw, Vt, testWord, tag_dict, count_t, count_tt, count_tw)
    #cal_perplexity(smooth, Vw, Vt, testWord, testTag, count_t, count_tt, count_tw)
    #acc_V, knownAcc_V, novelAcc_V = cal_acc(predictV, testTag, testWord, VocWords)
    #print(f"Tagging accuracy (Viterbi decoding): {acc_V}%\t(known: {knownAcc_V}% novel: {novelAcc_V}%)")
    #acc_P, knownAcc_P, novelAcc_P = cal_acc(predictP, testTag, testWord, VocWords)
    #print(f"Tagging accuracy (posterior decoding): {acc_P}%\t(known: {knownAcc_P}% novel: {novelAcc_P}%)")
    predictV = viterbi(smooth, Vw, Vt, testWord, tag_dict, A, B)
    f = open('test-output', 'w')
    for i in range(len(predictV)):
        f.write(testWord[i] + '/' + predictV[i] + '\n')
    f.close()


if __name__ == "__main__":
    main()
