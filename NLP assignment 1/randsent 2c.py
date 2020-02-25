import io

# Analyse the grammer
gram_file = open("grammar.gr","r")
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
    print(split_line_list)
    split_line_list.append('#')
    if not split_line_list[1] in non_term_dict.keys():
        non_term_dict[split_line_list[1]] = []
    for end in range(len(split_line_list)):
        if(split_line_list[end] == '#'):
            break
    
    non_term_list = split_line_list[2 : end]
    non_term_list.insert(0, split_line_list[0])

    non_term_dict[split_line_list[1]].append(non_term_list)

print(non_term_dict)

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
def dfs(gram_list, iter_steps):         #gram_list is the gram you need to use in a list form, the initail gram_list shoule be ['ROOT']. iter_steps stands for the steps the program has already iterated 
    gene_S = []
    if(iter_steps >= 10):
        return gene_S
    for index in range(1, len(gram_list)):
        symbol = gram_list[index]
        if not symbol in non_term_dict.keys():
            gene_S.append(symbol)
            continue
        #choice = random.randint(0, len(non_term_dict[symbol]) - 1)
        weight_list = [float(non_term_dict[symbol][x][0]) for x in range(0, len(non_term_dict[symbol]))]
        choice = random_with_weight(weight_list)
        #print('choice:', choice)
        new_list = dfs(non_term_dict[symbol][choice], iter_steps + 1)
        gene_S = gene_S + new_list
    return gene_S

def print_S(S_list):            #print the sentences 
    out_file = open('sentences.txt','a')
    pri_str = ""
    for word in S_list:
        pri_str = pri_str + word + ' '
    out_file.write(pri_str + '\n')
    #print(pri_str)

# geenerate specific sentences
for i in range(0, 10):
    Sample = dfs(['1', 'ROOT'], 0)
    print_S(Sample)

    
