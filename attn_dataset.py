import pandas as pd
import numpy as np
import os
from readData import search_class, gen_parts, gen_attn

path = os.path.join('Data', 'CLEANED-SCAN')
split = 'add_prim_split'
input_filename  = "tasks_test_addprim_turn_left.txt"
output_filename = "test_turn_left.tsv"



in_file = os.path.join(path, split, input_filename)
out_file = os.path.join(split, output_filename)

data_arr = pd.read_csv(in_file,sep='\t', header=None).values

data_arr = np.c_[data_arr, np.zeros(data_arr.shape[0], dtype=object)]

for i in range (data_arr.shape[0]):
    tags = search_class(data_arr[i,0])
    sents, sent_tags, sent_idx = gen_parts(data_arr[i,0].strip().split(' '), tags)
    src = gen_attn(data_arr[i,0].strip(), sents, sent_tags, sent_idx)
    data_arr[i,2] = ' '.join(map(str, np.nonzero(src)[1]))

df = pd.DataFrame(data_arr)
df.to_csv(out_file,sep='\t',header=False,index=False)
