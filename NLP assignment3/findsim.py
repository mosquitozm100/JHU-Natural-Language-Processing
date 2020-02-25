import io
import sys
import argparse

word_vecs = {}
numOfWords = 0
numOfDimentions = 0

def readfile(inputFileName):
    in_file = open(inputFileName,"r")
    first_line = in_file.readline()
    numOfWords= first_line.split()[0]
    numOfDimentions = first_line.split()[1]

#    print("first_line" + first_line + "\n")
    lines = in_file.readlines()
#    print(lines)
    for line in lines:
        word_vecs[line.split()[0]] = list(map(lambda x: float(x), line.split()[1:]))


import math
def calcCos(A_list, B_list):            #Calculate the Cosine value of two vector(in terms of list)
    sum_A_square = 0
    sum_B_square = 0
    sum_dot = 0
    len_A = len(A_list)
    for i in range(len_A):
        sum_A_square += A_list[i] ** 2
        sum_B_square += B_list[i] ** 2
        sum_dot += A_list[i] * B_list[i]
    return (sum_dot) / (math.sqrt(sum_A_square) * math.sqrt(sum_B_square))

def find_simiest_word(word_to_be_search, answers_list):
    target_vec = word_vecs[word_to_be_search]
    highest_similarity = 0
    similariest_word = ""
    for word, vec in word_vecs.items():
        if word == word_to_be_search:
            continue
        if word in answers_list:
            continue
        tmp_simi = calcCos(vec, target_vec)
        if tmp_simi > highest_similarity:
            highest_similarity = tmp_simi
            similariest_word = word
    return similariest_word

def find_simiest_word_minus_and_plus(word_to_be_search, answers_list, word_minus, word_plus):
    target_vec = list(map(lambda x, y, z: x - y + z, word_vecs[word_to_be_search], word_vecs[word_minus], word_vecs[word_plus]))
    highest_similarity = 0
    similariest_word = ""
    for word, vec in word_vecs.items():
        if word == word_to_be_search or word == word_minus or word == word_plus:
            continue
        if word in answers_list:
            continue
        tmp_simi = calcCos(vec, target_vec)
        if tmp_simi > highest_similarity:
            highest_similarity = tmp_simi
            similariest_word = word
    return similariest_word

if __name__=="__main__":
    #deal with the argument -t
    parser = argparse.ArgumentParser()
#    parser.add_argument("-t", "--tree", help= "whether generate a tree or not",
#                        action="store_true")
#    parser.add_argument("gram_file_name", type = str, help = "Name of the input grammer file")
#    parser.add_argument("repeat_times",type = int, help = "How many sentences do you want to generate?")
    parser.add_argument("input_word2vec_name", type = str, help = "input file in form of word2vec")
    parser.add_argument("word_to_be_search", type = str, help = "The word waiting to search for similiar words")
    parser.add_argument("--minus", type = str, help= "the minus word")
    parser.add_argument("--plus", type = str, help= "the plus word ")
    args = parser.parse_args()
    
    inpurFileName = args.input_word2vec_name
    readfile(inpurFileName)

    word_to_be_search = args.word_to_be_search
    answers_list = []
    for i in range(10):
        if(args.minus and args.plus):
            simiest_word = find_simiest_word_minus_and_plus(word_to_be_search, answers_list, args.minus, args.plus)
        else:
            simiest_word = find_simiest_word(word_to_be_search, answers_list)
            
        answers_list.append(simiest_word)
    #print(answers_list)
    for ans in answers_list:
        print(ans, end = ' ')
    print("")

        

