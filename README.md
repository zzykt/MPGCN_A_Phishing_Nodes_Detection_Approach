# MPGCN_A_Phishing_Nodes_Detection_Approach
A Phishing Nodes Detection Approach GCN for Ethereum 
py文件是ipynb转的，可能会有一点点问题，复制进juypter就好了。

实验用的是message_passing.py 运行python message_passing.py     实验结果根据seed值浮动，seed值不理想的话，结果也很一般。

xblock.pro给的数据集有K1（一阶邻域）的，还有K2的，k1中钓鱼节点有1300左右，k2中钓鱼节点有1660个，我也不清楚怎么多的那么多额外的
所以，大部分我都是用K2的1660做实验，最终写论文用1300个节点进行最终测试和验证。


ps：K2中的1600个节点中有400+节点他们的一阶交易数量少于5个，k1中钓鱼节点1300个节点至少为5笔交易。
