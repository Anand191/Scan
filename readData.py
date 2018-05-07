import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

path = './Data/CLEANED-SCAN/length_split'

data_arr = pd.read_csv(os.path.join(path,"tasks_train_length.txt"),sep='\t', header=None).values
data_arr = np.c_[data_arr, np.zeros(data_arr.shape[0], dtype=object)]

rev = False
s_counters = {'twice':2, 'thrice':3,}
c_counters = {'opposite':2, 'around':4}
directions = {'right': 90, 'left': -90}
connectives = {'and':rev, 'after': not rev}
actions = {'turn':0, 'jump':1, 'run':2, 'walk':3, 'look':4}

pos = {'sc':s_counters, 'cc':c_counters, 'dir':directions, 'conj':connectives, 'act':actions}

def search_class(sentence):
    word_tag = []
    for word in sentence.split(' '):
        for k,v in pos.items():
            if word in list(v.keys()):
                word_tag.append(k)
    return word_tag


def gen_parts(words, pos_tags):
    word_idx = np.arange(0,len(words)).tolist()
    indices = np.where(np.asarray(pos_tags)=='conj')[0]
    sub_sents = []
    word_keys = []
    i = 0
    j = 0
    while i<len(words):
        if(j==len(indices)):
            sub_sents.append(words[i:])
            word_keys.append(word_idx[i:])
            break
        else:
            sub_sents.append(words[i:indices[j]])
            word_keys.append(word_idx[i:indices[j]])
        i = indices[j]+1
        j +=1
    sub_idx = np.arange(0,len(sub_sents)).tolist()
    sub_idx = sub_idx[::-1]
    ord_idx = sub_idx
    position = max(sub_idx)
    for idx in indices[::-1]:
        if (words[idx]=='and'):
           id1 = ord_idx.index(position)
           temp_list = ord_idx[id1:]
           id2 = temp_list.index(position-1)
           ord_idx[id1] = temp_list[id2]
           temp_list.pop(id2)
           ord_idx[id1+1:] = temp_list
        else:
            ord_idx = ord_idx
        position -= 1
    ord_subs = []
    ord_keys = []
    for k in ord_idx:
        ord_subs.append(sub_sents[k])
        ord_keys.append(word_keys[k])
    ord_tags = [search_class(' '.join(map(str, sub))) for sub in ord_subs]
    # print(ord_idx)
    # print(sub_sents)
    # print(ord_subs)
    # print(ord_tags)
    # print(ord_keys)
    return (ord_subs, ord_tags, ord_keys)

def attention (idx, length):
    attn_vector = np.zeros(length).tolist()
    attn_vector[idx] = 1
    return attn_vector


def execute_step(sent, tag, index, length):
    attn_list = []
    if 'cc' in tag:
        idx1 = tag.index('cc')
        idx2 = tag.index('dir')
        idx3 = tag.index('act')
        if(sent[idx1]=='opposite'):
            for i in range (2):
                attn_list.append(attention(index[idx2], length))
                #print(attention(index[idx2], length))
            if (sent[idx3] != "turn"):
                attn_list.append(attention(index[idx3], length))
                #print(attention(index[idx3], length))
        else:
            for i in range(4):
                attn_list.append(attention(index[idx2], length))
                #print(attention(index[idx2], length))
                if (sent[idx3] != "turn"):
                    attn_list.append(attention(index[idx3], length))
                    #print(attention(index[idx3], length))
    elif 'dir' in tag:
        attn_list.append(attention(index[tag.index('dir')], length))
        #print(attention(index[tag.index('dir')], length))
        if (sent[tag.index('act')] != "turn"):
            attn_list.append(attention(index[tag.index('act')], length))
            #print(attention(index[tag.index('act')], length))

    else:
        if (sent[tag.index('act')] != "turn"):
            attn_list.append(attention(index[tag.index('act')], length))
            #print(attention(index[tag.index('act')], length))
    return attn_list



def gen_attn(sentence,sub_sentences, tags, idxs):
    words = sentence.split(' ')
    length = len(words)
    attn = []
    for i, sub in enumerate(sub_sentences):
        if 'sc' in tags[i]:
            id = tags[i].index('sc')
            count = sub[id]
            sub.pop(id)
            idxs[i].pop(id)
            tags[i].pop(id)
            for j in range(pos['sc'][count]):
                temp_attn = execute_step(sub, tags[i],idxs[i], length)
                for temp in temp_attn:
                    attn.append(temp)
        else:
            temp_attn = execute_step(sub, tags[i],idxs[i], length)
            for temp in temp_attn:
                attn.append(temp)
    return (np.asarray(attn))

def plot_attention(input_sentence, attentions):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone', vmin=0, vmax=1, aspect='auto')

    # fig.colorbar(cax)
    cb = plt.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' '), rotation=0)

    # X and Y labels
    ax.set_xlabel("INPUT")
    ax.xaxis.set_label_position('top')

    plt.show()


while True:
    cmd = raw_input("Enter command:")
    if (cmd=='exit'):
        break
    tags = search_class(cmd)
    sents, sent_tags, sent_idx = gen_parts(cmd.split(' '), tags)
    final_attn = gen_attn(cmd, sents, sent_tags, sent_idx)
    plot_attention(cmd, final_attn)



#=======================================================================================================================
# class Lang:
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {0: "SOS", 1: "EOS"}
#         self.n_words = 2  # Count SOS and PAD #remove EOS for now
#
#     def addSentence(self, sentence):
#         for word in sentence.split(' '):
#             self.addWord(word)
#
#     def addWord(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.n_words
#             self.word2count[word] = 1
#             self.index2word[self.n_words] = word
#             self.n_words += 1
#         else:
#             self.word2count[word] += 1
#
# ipt = Lang('input_vocab')
# tgt = Lang('target_vocab')
#
# for i in range(data_arr.shape[0]):
#     ipt.addSentence(data_arr[i,0])
#     tgt.addSentence(data_arr[i,1])
#     data_arr[i,2] = search_class(data_arr[i,0], pos)


