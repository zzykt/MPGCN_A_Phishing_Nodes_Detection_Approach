# %%
# 绘图，特征图
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import *
import os
import networkx as nx
import pandas as pd

# %%
csvdir = './EthereumK1/phishing/'
csvfiles = os.listdir(csvdir)


# %%

plt_y=[]
for csv in tqdm(csvfiles):#把一阶交易网络的交易记录转化为 图的节点和边
    count_fishing=0
    csvpath = os.path.join(csvdir, csv)
    csv = pd.read_csv(csvpath)
    if len(csv) == 0:
        print('Empty!')
    for row in csv.iterrows():
        count_fishing+=1
    plt_y.append(count_fishing)


# %%

plt_y.sort()
plt_x=[]
for i in range(len(plt_y)):
    plt_x.append(i+1)

f, ax = plt.subplots(1)
ax.plot(plt_x,plt_y)

plt.xlabel('range')
plt.ylabel("count") 
plt.show()





