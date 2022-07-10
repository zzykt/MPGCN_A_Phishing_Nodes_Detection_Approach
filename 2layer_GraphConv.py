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

# %%
dem=64
middle_dem=32
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )


# %%
csvdir = './EthereumK1/set/'
csvfiles = os.listdir(csvdir)

# %%
csvfiles.sort()
csvfiles = csvfiles

# %%
Labeldir='./EthereumK1/Phishing/'
Labelfiles = os.listdir(Labeldir)
LabelAddresses = [ str.lower(Labelfile.replace('.csv','')) for Labelfile in Labelfiles ]
LabelAddresses.sort()
len(LabelAddresses)

# %%
Normaldir='./EthereumK1/Normal/'
Normalfiles = os.listdir(Normaldir)
NormalAddresses = [ str.lower(Normalfile.replace('.csv','')) for Normalfile in Normalfiles ]
NormalAddresses.sort()
len(NormalAddresses)

# %%
# i=0
# graph = nx.MultiDiGraph()

# for csv in tqdm(csvfiles):#把一阶交易网络的交易记录转化为 图的节点和边
#     csvpath = os.path.join(csvdir, csv)
#     csv = pd.read_csv(csvpath)
#     if len(csv) == 0:
#         print('Empty!')
    
#     for row in csv.iterrows():
#         nodeFrom = str(row[1]['From'])
#         nodeTo = str(row[1]['To'])
#         txValue = float(row[1]['Value'])    
#         txTimeStamp = int(row[1]['TimeStamp'])


        
#         if nodeFrom not in graph: #为了把节点加入到新图中
#             if nodeFrom in LabelAddresses:
#                 graph.add_node(nodeFrom, isp=1)#1表示节点属于钓鱼节点（is phishing） 0表示一般节点  0表示normal节点
#             elif nodeFrom in NormalAddresses:
#                 graph.add_node(nodeFrom, isp=0)
#             else:
#                 graph.add_node(nodeFrom, isp=-1)
#         if nodeTo not in graph:
#             if nodeTo in LabelAddresses:
#                 graph.add_node(nodeTo, isp=1)
#             elif nodeTo in NormalAddresses:
#                 graph.add_node(nodeTo, isp=0)
#             else:
#                 graph.add_node(nodeTo, isp=-1)

#         graph.add_edge(nodeFrom, nodeTo, timestamp=txTimeStamp, amount=txValue)


# i=0
# graph = nx.MultiDiGraph()

# for csv in tqdm(csvfiles):#把一阶交易网络的交易记录转化为 图的节点和边
#     csvpath = os.path.join(csvdir, csv)
#     csv = pd.read_csv(csvpath)
#     if len(csv) == 0:
#         print('Empty!')
    
#     for row in csv.iterrows():

#         nodeFrom = str(row[1]['From'])
#         nodeTo = str(row[1]['To'])
#         txValue = float(row[1]['Value'])    
#         txTimeStamp =float( int(row[1]['TimeStamp'])-1438920447+45)
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
# nx.write_gpickle(graph, "k1Graph_timestamp_from_1.gpickle")
graph = nx.read_gpickle("k1Graph_timestamp_from_1.gpickle")   


# nx.write_gpickle(graph, "k1Graph.gpickle")
# graph = nx.read_gpickle("k1Graph.gpickle")   

# %%
import  dgl as d
dgl_graph = d.from_networkx(graph, node_attrs=['isp'], edge_attrs=['timestamp','amount'])

# %%
print(dgl_graph.edata['timestamp'])
print("------------")
print(dgl_graph.edata['amount'])

# %%
print("dgl IS successful create! ")
print('We have %d nodes.' % dgl_graph.number_of_nodes())
print('We have %d edges.' % dgl_graph.number_of_edges())
# We have 100804 nodes.
# We have 222091 edges.

# %%
embed = nn.Embedding(dgl_graph.number_of_nodes(), dem)
dgl_graph.ndata['feat'] = embed.weight
inputs=embed.weight

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
node_features = dgl_graph.ndata['feat']
node_labels = dgl_graph.ndata['isp']
n_features = node_features.shape[1] # 特征数量
n_labels = int(node_labels.max().item() + 1)# 标签种类


# %%
from dgl.nn.pytorch import GraphConv
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)
    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h
    

#第一层将大小为5的输入特征转换为隐藏大小为5的输入特征。
#第二层对隐含层进行变换，生成的输出特征大小2，代表 是（1） 不是（0）钓鱼节点。
# # 定义GCNLayer模块
# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)
#         # self.softmax = nn.Softmax(out_feats)
#     def forward(self, g, inputs):
#         # g 是图（graph） 并且 inputs 是输入的节点特征向量
#         # 首先设置图中 节点的特征向量
#         g.ndata['feat'] = inputs
#         # # 触发在所有边上传递信息
#         g.update_all(fn.u_mul_e('feat','amount','msg'),fn.max('msg','feat'))
#         # g.update_all(gcn_message,gcn_reduce)
#         # 取得节点向量
#         h = g.ndata.pop('feat')
#         # 进行线性变换
#         return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCN(in_feats, hidden_size)
        self.gcn2 = GCN(hidden_size, num_classes)
        
    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h



net = GCN(dem, middle_dem, 2)


# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
labeled_nodes=[]
labels=[]
all_mask=[]
train_mask=[]
test_mask=[]

for i,j in enumerate(dgl_graph.ndata["isp"]):
    
    if j==1 or j==0:
        labeled_nodes.append(i)
        labels.append(j)
        all_mask.append(True)
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
        if count<=3024:
            test_mask.append(False)
        else:
             test_mask.append(True)
# number_label=1660
# number_normal=1700
# # 划分训练集和测试集
# count_Label=0
# count_Normal=0
# rate=0.8
# for i,j in enumerate(dgl_graph.ndata["isp"]):
#     if j==1:
#         if count_Label<=int(number_label*rate):
#             count_Label+=1
#             labeled_nodes.append(i)
#             labels.append(j)
#             train_mask.append(True)
#             test_mask.append(False)
#         else:
#             labeled_nodes.append(i)
#             labels.append(j)
#             train_mask.append(False)
#             test_mask.append(True)
#     if j==0:
#         if count_Normal<=int(number_normal*rate):
#             count_Normal+=1
#             labeled_nodes.append(i)
#             labels.append(j)
#             train_mask.append(True)
#             test_mask.append(False)
#         else:
#             labeled_nodes.append(i)
#             labels.append(j)
#             train_mask.append(False)
#             test_mask.append(True)
#     else:
#         labeled_nodes.append(i)
#         labels.append(j)
#         train_mask.append(False)
#         test_mask.append(False)
# train_node=[]
# for i,j in enumerate(train_mask):
#     if j==True:
#         train_node.append(i)
        
        


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

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
    
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
dgl_graph = dgl.add_self_loop(dgl_graph)

# ----------- 4. training -------------------------------- #


epochess=[]
accm=[]
import itertools
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
dgl_graph = dgl.add_self_loop(dgl_graph)
for epochs in range(3000):
    epoch=epochs+1
    logits = net(dgl_graph, inputs)	# net() 是一个GCN模型 G是一个图数据 inputs 是节点 embedding 
    logp = F.log_softmax(logits, 1)
    
    # 我们仅仅为标记过的节点计算loss
    loss = F.nll_loss(logp[labeled_nodes[:3024]], labels[:3024])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        acc = evaluate(net, dgl_graph, inputs, node_labels, test_mask)
        print('Epoch %d | Loss: %.8f| Test Acc :%.8f' % (epoch, loss.item(),acc))
        epochess.append(epoch)
        accm.append(acc)

    

# %%
plt.plot(epochess,accm)
max(accm)

# %%
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
# ----------- 5. check results ------------------------ #

pred = torch.argmax(logits[labeled_nodes[3024:]], axis=1)

print('Accuracy', (pred == labels[3024:]).sum().item() / len(pred))

# %%



