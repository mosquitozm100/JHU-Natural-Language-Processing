import io
import sys
import argparse


def read_grammar():             # Analyse the grammar
    lines = gram_file.readlines()
    gram_lines = []
    non_term_dict = {}
    for line in lines:
        if(line.split() == []):
            continue
        if line.split()[0][0] != '#':
            gram_lines.append(line)        #find all lines which start by 1

    for line in gram_lines:
        split_line_list = line.split()
        #print(split_line_list)
        split_line_list.append('#')
        if not split_line_list[1] in non_term_dict.keys():
            non_term_dict[split_line_list[1]] = []
        for end in range(len(split_line_list)):
            if(split_line_list[end][0] == '#'):
                break
        
        non_term_list = split_line_list[2 : end]
        non_term_list.insert(0, split_line_list[0])
        #print(non_term_list)

        non_term_dict[split_line_list[1]].append(non_term_list)
    #print(non_term_dict)
    return non_term_dict

def random_with_weight(weight_list):
    #print(weight_list)
    sum_prob_from_start = 0
    total = sum(weight_list)
    rand_num = random.uniform(0, total)
    for index in range(0, len(weight_list)):
        sum_prob_from_start += weight_list[index]
        if(sum_prob_from_start >= rand_num):
            return index
    return len(weight_list) - 1

# generate the snetence recursively using dfs in list
import random
def dfs_S(gram_list):         #gram_list is the gram you need to use in a list form, the initail gram_list shoule be ['ROOT']. iter_steps stands for the steps the program has already iterated 
    global M
    if M <= 0:
        return ['...']
    M = M - 1
    gene_S_list = []
    for index in range(1, len(gram_list)):
        symbol = gram_list[index]
        if not symbol in non_term_dict.keys():
            gene_S_list.append(symbol)
            continue
        #choice = random.randint(0, len(non_term_dict[symbol]) - 1)
        weight_list = [float(non_term_dict[symbol][x][0]) for x in range(0, len(non_term_dict[symbol]))]
        choice = random_with_weight(weight_list)
        #print('choice:', choice)
        new_list = dfs_S(non_term_dict[symbol][choice])
        gene_S_list = gene_S_list + new_list
    return gene_S_list

def dfs_T(gram_list):         #gram_list is the gram you need to use in a list form, the initail gram_list shoule be ['ROOT']. iter_steps stands for the steps the program has already iterated 
    global M
    if M <= 0:
        return ' ... '
    M = M - 1
    str = ""
    for index in range(1, len(gram_list)):
        symbol = gram_list[index]
        if not symbol in non_term_dict.keys():      #terminal
            str = str + ' ' + symbol
            continue
        str = str + '( ' + symbol        #non-terminal
        #choice = random.randint(0, len(non_term_dict[symbol]) - 1)
        weight_list = [float(non_term_dict[symbol][x][0]) for x in range(0, len(non_term_dict[symbol]))]
        choice = random_with_weight(weight_list)
        #print('choice:', choice)
        #str = str + '(' 
        new_str = dfs_T(non_term_dict[symbol][choice])
        str = str + new_str
        str = str + ')'
    return str

def print_S(S_list):            #print the sentences 
    pri_str = ""
    for word in S_list:
        pri_str = pri_str + word + ' '
    #out_file.write(pri_str + '\n')
    print(pri_str)

def gene_S():
    global M 
    M = 450
    S_list = dfs_S(['1', 'ROOT'])
    print_S(S_list)

def gene_T():
    global M
    M = 450
    T_str = dfs_T(['1', 'ROOT'])
    print(T_str)
    #out_file.write(T_str + '\n')


if __name__=="__main__":
    #parser = argparse.ArgumentParser()
    #parser.parse_args()
    M = 450     #the limit number of non-external expanion
    InputFileName = sys.argv[1]
    OutputFileName = "sentences.txt"
    NumOfSen = sys.argv[2]

    gram_file = open(InputFileName,"r")
    #out_file = open('sentences.txt','w')
    
    non_term_dict = read_grammar()
    # geenerate specific sentences
    for i in range(0, int(NumOfSen)):
        #gene_S()
        gene_T()
       