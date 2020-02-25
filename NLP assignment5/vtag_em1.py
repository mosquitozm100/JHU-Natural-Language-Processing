import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train", help="train file")
parser.add_argument("test", help="test file")
parser.add_argument("raw", help="raw file")
args = parser.parse_args()

class data(object):

    def __init__(self, file, raw = None):
        f = open(file, 'r')
        self.tags, self.words = [], []
        self.tag_to_tag = {}
        self.tag_to_word = {}
        self.tag_dict = {}
        prev_tag = None
        for line in f:
            word, tag = line.strip().split('/')
            self.words.append(word)
            self.tags.append(tag)
            if prev_tag:
                self.tag_to_tag[(prev_tag, tag)] = self.tag_to_tag.get((prev_tag, tag), 0) + 1
            self.tag_to_word[(tag, word)] = self.tag_to_word.get((tag, word), 0) + 1
            if tag not in self.tag_dict.get(word, []):
                self.tag_dict[word] = self.tag_dict.get(word, []) + [tag]
            prev_tag = tag
        self.tag_vocab = list(set(self.tags)) + ['OOT']
        self.word_vocab = list(set(self.words)) + ['OOV']
        # self.tag_vocab = list(set(self.tags))
        self.tag_dict['OOV'] = self.tag_vocab.copy()
        self.tag_dict['OOV'].pop(self.tag_dict['OOV'].index('###'))
        f.close()
        if raw:
            self.raw_words = []
            with open(raw, 'r') as f:
                for line in f:
                    word = line.strip()
                    self.raw_words.append(word)
                    if word not in self.word_vocab:
                        self.word_vocab.append(word)
                        self.tag_dict[word] = self.tag_vocab.copy()
                        self.tag_dict[word].pop(self.tag_dict[word].index('###'))
        # print(self.tag_to_word, self.tag_to_tag, self.tag_dict)

    def get_p(self):
        lam = 1
        ptt = {}
        count_tag = {}
        count_tag['OOT'] = 0
        for tag in self.tag_to_tag:
            count_tag[tag[0]] = count_tag.get(tag[0], 0) + self.tag_to_tag[tag]
        for tag1 in self.tag_vocab:
            for tag2 in self.tag_vocab:
                ptt[(tag1, tag2)] = np.log((self.tag_to_tag.get((tag1, tag2), 0) + lam) / (count_tag[tag1] + lam * len(self.tag_vocab)))
        ptw = {}
        count_word = {}
        count_word['OOT'] = 0
        for tag in self.tag_to_word:
            count_word[tag[0]] = count_word.get(tag[0], 0) + self.tag_to_word[tag]
        for tag in self.tag_vocab:
            for word in self.word_vocab:
                ptw[(tag, word)] = np.log((self.tag_to_word.get((tag, word), 0) + lam) / (count_word[tag] + lam * len(self.word_vocab)))
        return ptt, ptw

def lse(x):
    # log sum exp
    m = np.max(x)
    x -= m
    return m + np.log(np.sum(np.exp(x)))

class EM(object):

    def __init__(self, train_file, test_file, raw_file):
        self.train = data(train_file, raw_file)
        self.ptt, self.ptw = self.train.get_p()
        self.test = data(test_file)
        self.original_tt, self.original_tw = self.train.tag_to_tag.copy(), self.train.tag_to_word.copy()

        for i in range(10):
            self.predict()
            self.accuracy()
            print('Iteration', i, end = ' ')
            self.em_step()
            # self.acc_posterior()
        self.save('test-output')

    def predict(self):
        self.mu = {}
        self.mu[('###', 0)] = 0
        self.total_prob = 0
        self.backpointer = {}
        self.test.tags = [w if w in self.train.tag_vocab else 'OOT' for w in self.test.tags]
        self.test.words = [w if w in self.train.word_vocab else 'OOV' for w in self.test.words]
        for i in range(1, len(self.test.words)):
            self.total_prob += self.ptt[(self.test.tags[i-1], self.test.tags[i])] + self.ptw[(self.test.tags[i], self.test.words[i])]
            for t_i in self.train.tag_dict[self.test.words[i]]:
                for t_iminus1 in self.train.tag_dict[self.test.words[i-1]]:
                    p = self.ptt[(t_iminus1, t_i)] + self.ptw[(t_i, self.test.words[i])]
                    u = self.mu[(t_iminus1, i-1)] + p
                    if (t_i, i) not in self.mu or u > self.mu[(t_i, i)]:
                        self.mu[(t_i, i)] = u
                        self.backpointer[(t_i, i)] = t_iminus1

    def accuracy(self):
        total, correct = 0, 0
        total_known, correct_known = 0, 0
        total_novel, correct_novel = 0, 0
        total_seen, correct_seen = 0, 0
        t = '###'
        for i in range(len(self.test.words)-1, 0, -1):
            t = self.backpointer[(t, i)]
            if self.test.tags[i-1] != '###':
                total += 1
                if self.test.words[i-1] != '###':
                    if self.test.words[i-1] in self.train.words:
                        total_known += 1
                    elif self.test.words[i-1] != 'OOV':
                        total_seen += 1
                    else:
                        total_novel += 1
                if t == self.test.tags[i-1]:
                    correct += 1
                    if self.test.words[i-1] != '###':
                        if self.test.words[i-1] in self.train.words:
                            correct_known += 1
                        elif self.test.words[i-1] != 'OOV':
                            correct_seen += 1
                        else:
                            correct_novel += 1
        acc = correct/total * 100
        if total_known == 0:
            acc_k = 0
        else:
            acc_k = correct_known / total_known * 100
        if total_seen == 0:
            acc_s = 0
        else:
            acc_s = correct_seen / total_seen * 100
        if total_novel == 0:
            acc_n = 0
        else:
            acc_n = correct_novel / total_novel * 100
        perplexity = np.exp(-self.total_prob/(len(self.test.tags)-1))
        print('Model perplexity per tagged test word: {:.3f}'.format(perplexity))
        print('Tagging accuracy (Viterbi decoding): {:.2f}% (known: {:.2f}% seen: {:.2f}% novel: {:.2f}%)'.format(acc, acc_k, acc_s, acc_n))

    def em_step(self):
        # self.count_tt, self.count_tw = self.original_tt.copy(), self.original_tw.copy()
        self.count_tt, self.count_tw = {}, {}
        self.alpha, self.beta, self.posterior_tags = {}, {}, []
        self.alpha[('###', 0)] = 0
        self.raw_words = [w if w in self.train.word_vocab else 'OOV' for w in self.train.raw_words]
        self.untagged_prob = 0
        for i in range(1, len(self.raw_words)):
            for t_i in self.train.tag_dict[self.raw_words[i]]:
                for t_iminus1 in self.train.tag_dict[self.raw_words[i-1]]:
                    p = self.ptt[(t_iminus1, t_i)] + self.ptw[(t_i, self.raw_words[i])]
                    if (t_i, i) not in self.alpha:
                        self.alpha[(t_i, i)] = self.alpha[(t_iminus1, i-1)] + p
                    else:
                        self.alpha[(t_i, i)] = lse([self.alpha[(t_i, i)], self.alpha[(t_iminus1, i-1)] + p])
                if i == len(self.raw_words) - 1:
                    self.untagged_prob += self.alpha[(t_i, i)]
        Z = self.alpha[('###', len(self.raw_words)-1)]
        self.beta[('###', len(self.raw_words)-1)] = 0
        for i in range(len(self.raw_words)-1, 0, -1):
            pmax = 1
            for t_i in self.train.tag_dict[self.raw_words[i]]:
                pi = self.alpha[(t_i, i)] + self.beta[(t_i, i)] - Z
                # if i == 16:
                #     print(t_i, np.exp(pi), np.exp(self.alpha[(t_i, i)]), np.exp(self.beta[(t_i, i)]))
                self.count_tw[(t_i, self.raw_words[i])] = self.count_tw.get((t_i, self.raw_words[i]), 0) + np.exp(pi)
                if pmax == 1 or pi > pmax:
                    pmax = pi
                    tag = t_i
                for t_iminus1 in self.train.tag_dict[self.raw_words[i-1]]:
                    p = self.ptt[(t_iminus1, t_i)] + self.ptw[(t_i, self.raw_words[i])]
                    if (t_iminus1, i-1) not in self.beta:
                        self.beta[(t_iminus1, i-1)] = self.beta[(t_i, i)] + p
                    else:
                        self.beta[(t_iminus1, i-1)] = lse([self.beta[(t_iminus1, i-1)], self.beta[(t_i, i)] + p])
                    ptt = self.alpha[(t_iminus1, i-1)] + p + self.beta[(t_i, i)] - Z
                    self.count_tt[(t_iminus1, t_i)] = self.count_tt.get((t_iminus1, t_i), 0) + np.exp(ptt)
            self.posterior_tags = [tag] + self.posterior_tags
        self.count_tw[('###', '###')] += 1
        self.train.tag_to_tag, self.train.tag_to_word = self.count_tt, self.count_tw
        self.ptt, self.ptw = self.train.get_p()
        perplexity = np.exp(-self.untagged_prob/(len(self.raw_words)-1))
        print('Model perplexity per untagged raw word: {:.3f}'.format(perplexity))
        # print(self.count_tt, self.count_tw)
        print("Z:", Z)
        #print(self.p_tt)
        #print(self.count_tw)
    def acc_posterior(self):
        total, correct = 0, 0
        total_known, correct_known = 0, 0
        total_novel, correct_novel = 0, 0
        t = '###'
        for i in range(1, len(self.test.tags)):
            if self.test.tags[i] != '###':
                total += 1
                if self.test.words[i] != '###':
                    if self.test.words[i] != 'OOV':
                        total_known += 1
                    else:
                        total_novel += 1
                if self.posterior_tags[i-1] == self.test.tags[i]:
                    correct += 1
                    if self.test.words[i] != '###':
                        if self.test.words[i] != 'OOV':
                            correct_known += 1
                        else:
                            correct_novel += 1
        acc = correct/total * 100
        if total_known == 0:
            acc_k = 0
        else:
            acc_k = correct_known / total_known * 100
        if total_novel == 0:
            acc_n = 0
        else:
            acc_n = correct_novel / total_novel * 100
        print('Tagging accuracy (posterior decoding): {:.2f}% (known: {:.2f}% novel: {:.2f}%)'.format(acc, acc_k, acc_n))

    def save(self, file):
        f = open(file, 'w')
        for i in range(len(self.test.words)):
            if i == 0:
                f.write(self.test.words[i] + '/' + '###' + '\n')
            else:
                f.write(self.test.words[i] + '/' + self.test.tags[i-1] + '\n')
        f.close()

if __name__ == '__main__':
    result = EM(args.train, args.test, args.raw)
