import torch
import torch.nn.functional as F
import torch.autograd as autograd
# import torch
import random
import numpy
# import pandas as pd
# data=pd.read_csv(r'./trainset/category/0a0a780a-7ec9-4fb8-91d2-135348649e4b_channelVELO_TOP.csv')
# # print(data)
# data=numpy.array(data)
# data=torch.from_numpy(data)
# data = data.squeeze()
# print(type(data))
# print(data.size())
# print(data)
input= torch.randn(2,5,7)
print(input.data.max(1)[1])
input=input.view(10,7)
# print(input)
target = torch.tensor([[0,1,0,2,1],[0,0,1,1,1]])
target = target.view(2*5)
print(target.size())
loss = F.nll_loss(input,target)
print(loss)
print(target.data)
# data=autograd.Variable(torch.FloatTensor([1.0,2.0,3.0]))
# log_softmax=F.log_softmax(data,dim=0)
# # print(log_softmax)
# cat={'2241':'o','2247':'p','6553':'q'}
# classes = dict(zip(sorted(cat), range(len(cat))))
# print(classes)