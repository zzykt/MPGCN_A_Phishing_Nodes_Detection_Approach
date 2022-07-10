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

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import itertools
import time 
from sklearn.metrics import *
import numpy as np
#21s

# %%
dem=16
hidden_size=8
# torch.manual_seed(1)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    

# %%

csvdir = './EthereumK1/set/'
csvfiles = os.listdir(csvdir)

# %%
csvfiles.sort()
csvfiles = csvfiles

# %%
Labeldir='./EthereumK1/k1_phishing/'
Labelfiles = os.listdir(Labeldir)
LabelAddresses = [ str.lower(Labelfile.replace('.csv','')) for Labelfile in Labelfiles ]
LabelAddresses.sort()
len(LabelAddresses)

# %%
Normaldir='./EthereumK1/k1_Normal/'
Normalfiles = os.listdir(Normaldir)
NormalAddresses = [ str.lower(Normalfile.replace('.csv','')) for Normalfile in Normalfiles ]
NormalAddresses.sort()
len(NormalAddresses)

# %%
# %%time
# print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
# i=0

# graph = nx.MultiDiGraph()

# for csv in tqdm(csvfiles):#把一阶交易网络的交易记录转化为 图的节点和边
#     csvpath = os.path.join(csvdir, csv)
#     csv = pd.read_csv(csvpath)
#     if len(csv) == 0:
#         print('Empty!')
#     timestamp=1
#     for row in csv.iterrows():
   
#         nodeFrom = str(row[1]['From'])
#         nodeTo = str(row[1]['To'])
#         txValue = float(row[1]['Value'])    
#         # txTimeStamp =float( int(row[1]['TimeStamp'])-1438920447+45)
#         txTimeStamp=float(timestamp)
#         timestamp+=1
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

# nx.write_gpickle(graph, "k1Graph_1206.gpickle")


# %%
# nx.write_gpickle(graph, "k1Graph_timestamp_from_1.gpickle")
# graph = nx.read_gpickle("k1Graph_timestamp_from_1.gpickle")   
# graph = nx.read_gpickle("K2Graph.gpickle")   
graph = nx.read_gpickle("k1Graph_1206.gpickle")   

# nx.write_gpickle(graph, "k1Graph.gpickle")
# graph = nx.read_gpickle("k1Graph.gpickle")   

# %%
import dgl as d
dgl_graph = d.from_networkx(graph, node_attrs=['isp'], edge_attrs=['timestamp','amount'])
# k1 7s
#k2 2min

# %%

# dgl_graph.ndata['out_deg']=dgl_graph.out_degrees(dgl_graph.nodes())
# dgl_graph.ndata['in_deg']=dgl_graph.in_degrees(dgl_graph.nodes())

# timestamp=dgl_graph.edata['timestamp']
# print(timestamp)
# print(min(timestamp))
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
nn
embed = nn.Embedding(dgl_graph.number_of_nodes(), dem)
dgl_graph.ndata['feat'] = embed.weight
inputs = embed.weight


# %%
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATConv
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
        # g.update_all(fn.u_mul_e('feat','amount','msg'),fn.max('msg','feat'))
        g.update_all(fn.u_mul_e('feat','timestamp','msg'),fn.max('msg','feat'))
        # g.update_all(gcn_message,gcn_reduce)
        # 取得节点向量
        h = g.ndata.pop('feat')
        
  
        # 进行线性变换
        return self.linear(h)

# class GCN(nn.Module):
#     def __init__(self, in_feats, num_classes):
#         super(GCN, self).__init__()
#         self.gcn1 = GCNLayer(in_feats, num_classes)
        
#     def forward(self, g, inputs):
#         h = self.gcn1(g, inputs)
#         h = torch.relu(h)
#         h= torch.dropout(h,p=0.5,train=self.training)  
#         # h = self.gcn2(g, h)
#         return h

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

        
    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h= torch.dropout(h,p=0.5,train=self.training)  
        h = self.gcn2(g, h)

        return h

# class GAT(torch.nn.Module):
#     def __init__(self, d,hidden_channels):
#         super(GAT, self).__init__()

#         self.conv1 = GATConv(d, hidden_channels)
#         self.conv2 = GATConv(hidden_channels, 2)
        
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x
# model = GAT(hidden_channels=16)
net=GCN(dem,hidden_size,2)
# net=GCN(dem,2)

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
import random
# 划分训练集和测试集
for i,j in enumerate(dgl_graph.ndata["isp"]):
    
    if j==1 or j==0:
        labeled_nodes.append(i)
        labels.append(j)
        all_mask.append(True)
    else:
        all_mask.append(False)
labels=torch.tensor(labels)
count=0


# count_0=0
# count_1=0
# for i,j in enumerate(all_mask):
#     if j==False:
#         test_mask.append(False)
#     if j==True:
#         count+=1
#         if count<=2000:
#             if dgl_graph.ndata["isp"][i]==0:
#                 count_0+=1
#             if dgl_graph.ndata["isp"][i]==1:    
#                 count_1+=1
# print(count_0)
# print(count_1)

            


for i,j in enumerate(all_mask):
    if j==False:
        # train_mask.append(False)
        test_mask.append(False)
    if j==True:
        count+=1
        if count<=2000:
            test_mask.append(False)
        else:
            test_mask.append(True)
        
# for i,j in enumerate(all_mask):
#     if j==False:
#         # train_mask.append(False)
#         test_mask.append(False)
#     if j==True:
#         count+=1
#         if count<=1400:
#             test_mask.append(False)
#         else:
#             test_mask.append(True)
        

# %%

import sklearn
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
        mf1=sklearn.metrics.f1_score(labels,indices,average='micro')
        return mf1
# net = GCNLayer(dem,2)
# net = GAT(hidden_channels=16)

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
# optimizer = torch.optim.BGD(itertools.chain(net.parameters(), embed.parameters()), lr=0.1)
dgl_graph = dgl.add_self_loop(dgl_graph)

# ----------- 4. training -------------------------------- #


epochess=[]
accm=[]
prem=[]
recallm=[]
f1_scorem=[]
for epochs in range(1500):
    # net.train()
    epoch=epochs+1
    # 使用所有节点(全图)进行前向传播计算
    logits = net(dgl_graph, inputs)
    # 计算损失值
    logp = F.log_softmax(logits, 1)

    # loss = F.nll_loss(logp, node_labels)
    # loss = F.nll_loss(logp[labeled_nodes[:2500]], labels[:2500])
    # loss = F.nll_loss(logp[labeled_nodes[:3024]], labels[:3024])
    loss = F.nll_loss(logp[labeled_nodes[:2000]], labels[:2000])
    
    # loss = F.nll_loss(logp[labeled_nodes[:3100]], labels[:3100])
    # 进行反向传播计算
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    if epoch%10==0:
        acc=evaluate_acc(net, dgl_graph, inputs, node_labels, test_mask)
        precision=evaluate_precision(net, dgl_graph, inputs, node_labels, test_mask)
        recall=evaluate_recall(net, dgl_graph, inputs, node_labels, test_mask)
        f1_score=evaluate_f1_score(net, dgl_graph, inputs, node_labels, test_mask)
        # acc = evaluate(net, dgl_graph, inputs, node_labels, test_mask)
        print('Epoch %4d | Loss: %.6f| Test acc :%.6f| precision :%.6f| recall :%.6f|f1_score :%.6f' %(epoch,loss.item(),acc,precision,recall,f1_score))
        epochess.append(epoch)
        accm.append(acc)
        prem.append(precision)
        recallm.append(recall)
        f1_scorem.append(f1_score)
  






print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
# ----------- 5. check results ------------------------ #

pred = torch.argmax(logits[labeled_nodes], axis=1)
print('accuracy_score', accuracy_score(pred,labels))
print('precision_score', precision_score(pred,labels))
print('recall_score', recall_score(pred,labels))
print('f1_score', sklearn.metrics.f1_score(pred,labels))


# print('Accuracy', (pred == labels).sum().item() / len(pred))

# %%
from sklearn.metrics import roc_curve
fpr, tpr, thersholds = roc_curve(pred, labels, pos_label=1)
print(pred[:100])
print(labels[:100])
from sklearn.metrics import auc
roc_auc = auc(fpr, tpr)


# %%
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
 
plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)
 
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# %%
print(net)


