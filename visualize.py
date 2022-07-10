# %%
# 绘图，特征图
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
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

# %%

csvdir = './EthereumK1/Phishing/'
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

%%time
graph = nx.MultiDiGraph()


# %%
i=0
for csv in tqdm(csvfiles):#把一阶交易网络的交易记录转化为 图的节点和边
    csvpath = os.path.join(csvdir, csv)
    csv = pd.read_csv(csvpath)
    if len(csv) == 0:
        print('Empty!')
    graph.__init__()
    for row in csv.iterrows():
        nodeFrom = str(row[1]['From'])
        nodeTo = str(row[1]['To'])
        txValue = float(row[1]['Value'])    
        txTimeStamp = int(row[1]['TimeStamp'])
        if nodeFrom not in graph:
            if nodeFrom in LabelAddresses or nodeTo in LabelAddresses:

                graph.add_node(nodeFrom, isp=1)
            else:
                graph.add_node(nodeFrom, isp=0)
        if nodeTo not in graph:
            if nodeFrom in LabelAddresses or nodeTo in LabelAddresses:
                graph.add_node(nodeTo, isp=1)
            else:
                graph.add_node(nodeTo, isp=0)    
        graph.add_edge(nodeFrom, nodeTo, timestamp=txTimeStamp, amount=txValue)

    pos=nx.spring_layout(graph) 
    # nodecolor=graph.degree() #度数越大，节点越大，连接边颜色越深
    # nodecolor2=pd.DataFrame(nodecolor) #转化称矩阵形式
    # nodecolor3=nodecolor2.iloc[:,1] #索引第二列
    # edgecolor=range(graph.number_of_edges()) #设置边权颜色
    # nx.draw(graph, pos, with_labels=False,node_size=nodecolor3*5) 
    nx.draw(graph, pos, with_labels=False) 


    # pos = nx.kamada_kawai_layout(graph) 
    # nx.draw_spring(graph,node_size=10,width=0.5)
    # nx.draw_networkx(graph,pos,with_labels=False,node_size=10,width=0.5)
    
    
    # plt.savefig("./picture/%i.png",dpi=200, bbox_inches='tight')
    # nx.draw(graph,edge_color='lightseagreen',alpha=0.5,connectionstyle='arc3, rad = 0.2')
    # nx.draw(graph,pos=None,edge_color='lightseagreen',alpha=0.5,connectionstyle='arc3, rad = 0.2',width=[float(v['Value']) for (r,c,v) in graph.edges(data=True)])

    
    # plt.savefig("D:/Phishing/temp1.jpg".format(i),format='eps',dpi=1000, bbox_inches='tight')
    plt.savefig("e:/Phishing/temp{}.jpg".format(i),dpi=10000, bbox_inches='tight')
    # plt.show()
    plt.clf() 
    plt.close()
    i+=1
    # if i>20:
    #     break
            


            # if nodeFrom not in graph: #为了把节点加入到新图中
            #     if nodeFrom in LabelAddresses:
            #         graph.add_node(nodeFrom, isp=1)#1表示节点属于钓鱼节点（is phishing） 0表示不属于
            #     else:
            #         graph.add_node(nodeFrom, isp=0)
            # if nodeTo not in graph:
            #     if nodeTo in LabelAddresses:
            #         graph.add_node(nodeTo, isp=1)
            #     else:
            #         graph.add_node(nodeTo, isp=0)
            #把该项交易记录作为边记录在图中。
            # graph.add_edge(nodeFrom, nodeTo, timestamp=txTimeStamp, amount=txValue)


# %%



