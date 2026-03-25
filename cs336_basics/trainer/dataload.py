import torch
import numpy as np

def data_loading(dataset,batch_size,context_length,device):
    N=len(dataset)
    max_i=N-context_length
    rand_sample=np.random.randint(0,max_i,size=batch_size)
    input_list=[]
    target_list=[]
    for i in rand_sample:
        input1=dataset[i:i+context_length]
        input_list.append(input1)
        target1=dataset[i+1:i+1+context_length]
        target_list.append(target1)
    input_list2=torch.tensor(input_list,dtype=torch.long).to(device)
    target_list2=torch.tensor(target_list,dtype=torch.long).to(device)
    return (input_list2,target_list2)