# %%
import dgl
import dgl.function as fn
from dgl.nn import SAGEConv
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import *
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import itertools
import time 
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import *

# %%
dem=64
hidden_size=16
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    

# %%
rootPath='.\\EthereumK2\\'
file=[]
for dirpath, dirnames, filenames in os.walk(rootPath):
    # print(dirpath, " *******",dirnames, " *******",filenames)
    for i in filenames:
        if i[-4:]=='.csv':
            file.append(os.path.join(dirpath, i))
            # print(file)
print(len(file),'个csv文件')


# %%
Labeldir='./EthereumK2/Phishing/'
Labelfiles = os.listdir(Labeldir)
LabelAddresses = [ str.lower(Labelfile.replace('.csv','')) for Labelfile in Labelfiles ]
LabelAddresses.sort()
print(len(LabelAddresses),'个Label地址')
number_Label=len(LabelAddresses)


# %%
Normaldir='./EthereumK2/Normal/'
Normalfiles = os.listdir(Normaldir)
NormalAddresses = [ str.lower(Normalfile.replace('.csv','')) for Normalfile in Normalfiles ]
NormalAddresses.sort()
print(len(NormalAddresses),'个NormalLabel地址')
number_normal=len(NormalAddresses)

# %%
# %%time
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
# i=0
# graph = nx.MultiDiGraph()

# for csv1 in tqdm(file):#把二阶交易网络的交易记录转化为 图的节点和边
#     csv = pd.read_csv(csv1)
#     if len(csv) == 0:
#         print('Empty!')
    
#     for row in csv.iterrows():

#         nodeFrom = str(row[1]['From'])
#         nodeTo = str(row[1]['To'])
#         txValue = float(row[1]['Value'])    
#         txTimeStamp =float( int(row[1]['TimeStamp'])-1438920447+45+832)
#         # txTimeStamp = int(row[1]['TimeStamp'])


        
#         if nodeFrom not in graph: #为了把节点加入到新图中
#             if nodeFrom in LabelAddresses:
#                 graph.add_node(nodeFrom, isp=1)#1表示节点属于钓鱼节点（is phishing） 0表示一般节点  0表示normal节点
#             elif nodeFrom in NormalAddresses:
#                 graph.add_node(nodeFrom, isp=0)
#             else:
#                 graph.add_node(nodeFrom, isp=2)
#         if nodeTo not in graph:
#             if nodeTo in LabelAddresses:
#                 graph.add_node(nodeTo, isp=1)
#             elif nodeTo in NormalAddresses:
#                 graph.add_node(nodeTo, isp=0)
#             else:
#                 graph.add_node(nodeTo, isp=2)

#         graph.add_edge(nodeFrom, nodeTo, timestamp=txTimeStamp, amount=txValue)



# %%
# nx.write_gpickle(graph, "K2Graph.gpickle")
graph = nx.read_gpickle("K2Graph.gpickle")   

# %%


# %%
import dgl as d
dgl_graph = d.from_networkx(graph, node_attrs=['isp'], edge_attrs=['timestamp','amount'])

# %%

# dgl_graph.ndata['out_deg']=dgl_graph.out_degrees(dgl_graph.nodes())
# dgl_graph.ndata['in_deg']=dgl_graph.in_degrees(dgl_graph.nodes())

# timestamp=dgl_graph.edata['timestamp']
# print(timestamp)
# amount=dgl_graph.edata['amount']

# w=timestamp*amount
# dgl_graph.edata['w']=w

# dgl_graph.edata['w']=dgl_graph.edata['timestamp']*dgl_graph.edata(['amount'])
# dgl_graph.ndata['in_amount']=dgl_graph.nodes()
# dgl_graph.ndata['out_amount']=dgl_graph.

# %%
# print(dgl_graph.edata['timestamp'])
# print("---------------------------------------------")
# print(dgl_graph.edata['amount'])
# print("---------------------------------------------")
# print(dgl_graph.ndata['out_deg'])
# print("---------------------------------------------")
# print(dgl_graph.ndata['in_deg'])
# print("---------------------------------------------")
# print(int(max(dgl_graph.edata['timestamp'])))
# print(int(min(dgl_graph.edata['timestamp'])))
# print(float(min(dgl_graph.edata['amount'])))
# print("---------------------------------------------")
# print(dgl_graph.edata['w'])

# %%
print("dgl IS successful create! ")
print('We have %d nodes.' % dgl_graph.number_of_nodes())
print('We have %d edges.' % dgl_graph.number_of_edges())
# We have 100804 nodes.
# We have 222091 edges.

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
embed = nn.Embedding(dgl_graph.number_of_nodes(), dem)
dgl_graph.ndata['feat'] = embed.weight
inputs = embed.weight

# %%
%%time
import torch.nn as nn
import torch.nn.functional as F
# 定义GCNLayer模块
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        # self.softmax = nn.Softmax(out_feats)
    def forward(self, g, inputs):
        # g 是图（graph） 并且 inputs 是输入的节点特征向量
        # 首先设置图中 节点的特征向量
        g.ndata['feat'] = inputs
        # # 触发在所有边上传递信息
        # g.update_all(fn.u_mul_e('feat','timestamp','msg'),fn.max('msg','feat'))
        g.update_all(fn.u_mul_e('feat','amount','msg'),fn.max('msg','feat'))
        # g.update_all(gcn_message,gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('feat')
        # 进行线性变换
        return self.linear(h)

# class GCN(nn.Module):
#     def __init__(self, in_feats, num_classes):
#         super(GCN, self).__int__()
#         self.gcn1 = GCNLayer(in_feats, num_classes)
        
#     def forward(self, g, inputs):
#         h = self.gcn1(g, inputs)
#         h = torch.relu(h)
#         # h = self.gcn2(g, h)
#         # return h



class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__int__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)
        
    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h= torch.dropout(h,p=0.5,train=self.training)
        h = self.gcn2(g, h)
        return h

# net = GCNLayer(dem, 2)
net=GCN(dem,hidden_size,2 )


# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
node_features = dgl_graph.ndata['feat']
node_labels = dgl_graph.ndata['isp']
n_features = node_features.shape[1] # 特征数量
n_labels = int(node_labels.max().item() + 1)# 标签种类




# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
print("图的节点数和边数: ", dgl_graph.num_nodes(), dgl_graph.num_edges())
# print("训练集节点数：", train_mask.sum().item())
# print("测试集节点数：", test_mask.sum().item())
print("节点特征维数：", n_features)
print("标签类目数：", n_labels)


# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
labeled_nodes=[]
labels=[]
all_mask=[]
train_mask=[]
test_mask=[]
number_Label=1265
number_normal=1590
# 划分训练集和测试集
for i,j in enumerate(dgl_graph.ndata["isp"]):
    
    if j==1 or j==0:
        labeled_nodes.append(i)#第几个节点
        labels.append(j)#节点label
        all_mask.append(True) #节点是否要进行训练
    else:
        all_mask.append(False)
labels=torch.tensor(labels)

count=0

for i,j in enumerate(all_mask):
    if j==False:
        # train_mask.append(False)
        test_mask.append(False)
    if j==True:
        count+=1
        if count<=2200:
            test_mask.append(False)
        else:
             test_mask.append(True)
        
        



# %%
def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        
        return correct.item() * 1.0 / len(labels)

def evaluate_acc(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        accuracy=accuracy_score(labels,indices)
        return accuracy

def evaluate_precision(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        precision=precision_score(labels,indices)
        return precision

def evaluate_recall(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        recall=recall_score(labels,indices)
        return recall

def evaluate_f1_score(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        _f1=f1_score(labels,indices,average='micro')
        return _f1



def test(net,dgl_graph,inputs,labels,mask):
    net.eval()
    with torch.no_grad():
        logits = net(dgl_graph, inputs)
        test_mask_logits = logits[mask]
        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, labels[mask]).float().mean()
        return accuarcy

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
dgl_graph = dgl.add_self_loop(dgl_graph)

# ----------- 4. training -------------------------------- #


epochess=[]
accm=[]
prem=[]
recallm=[]
f1_scorem=[]
for epochs in range(3000):
    # net.train()
    epoch=epochs+1
    # 使用所有节点(全图)进行前向传播计算
    logits = net(dgl_graph, inputs)
    # 计算损失值
    logp = F.log_softmax(logits, 1)
    loss=F.binary_cross_entropy

    # loss = F.nll_loss(logp, node_labels)
    loss = F.nll_loss(logp[labeled_nodes[:2200]], labels[:2200])
    
    # 进行反向传播计算
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    if epoch%10==0:
        acc=evaluate_acc(net, dgl_graph, inputs, node_labels, test_mask)
        precision=evaluate_precision(net, dgl_graph, inputs, node_labels, test_mask)
        recall=evaluate_recall(net, dgl_graph, inputs, node_labels, test_mask)
        # f1_score=evaluate_f1_score(net, dgl_graph, inputs, node_labels, test_mask)

        # acc = evaluate(net, dgl_graph, inputs, node_labels, test_mask)
        print('Epoch %4d | Loss: %.6f| Test acc :%.6f| precision :%.6f| recall :%.6f' %(epoch,loss.item(),acc,precision,recall))
        epochess.append(epoch)
        accm.append(acc)
        prem.append(precision)
        recallm.append(recall)
        # f1_scorem.append(f1_score)






        # # print('Epoch %d | Loss: %.8f' % (epoch, loss.item()))
        # # acc=test(net, dgl_graph, inputs, node_labels, test_mask)
        # acc = evaluate(net, dgl_graph, inputs, node_labels, test_mask)
        # print('Epoch %d | Loss: %.8f| Test Acc :%.8f' % (epoch, loss.item(),acc))
        # epochess.append(epoch)
        # accm.append(acc)



# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
print("max acc=",max(accm))
print("max pre=",max(prem))
print("max recall=",max(recallm))
# print("max f1_score=",max(f1_scorem))

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
# ----------- 5. check results ------------------------ #
pred = torch.argmax(logits[labeled_nodes], axis=1)
print('accuracy_score', accuracy_score(pred,labels))
print('precision_score', precision_score(pred,labels))
print('recall_score', recall_score(pred,labels))
# print('f1_score', f1_score(pred,labels))
