import pandas as pd
import numpy as np
import os
from readData import search_class, gen_parts, gen_attn

path = './Data/CLEANED-SCAN/length_split'

data_arr = pd.read_csv(os.path.join(path,"tasks_train_length.txt"),sep='\t', header=None).values

data_arr = np.c_[data_arr, np.zeros(data_arr.shape[0], dtype=object)]

for i in range (data_arr.shape[0]):
    tags = search_class(data_arr[i,0])
    sents, sent_tags, sent_idx = gen_parts(data_arr[i,0].strip().split(' '), tags)
    src = gen_attn(data_arr[i,0], sents, sent_tags, sent_idx)
    data_arr[i,2] = ' '.join(map(str, np.nonzero(src)[1]))

df = pd.DataFrame(data_arr)
df.to_csv(os.path.join(path,'tasks_train_length_attn.txt'),sep='\t',header=False,index=False)
