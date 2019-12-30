# NERs
A simple review about Name Entity Recognition
https://www.linuxidc.com/Linux/2019-05/158461.htm 用来解决git clone 速度慢的问题
git remote rm origin删除远程保持的链接
git clone git://github.com:username/repositoryname 比http协议要快一些
epochs==50 and use_CRF==True and glove_path==True f1_score==0.648420, correct/total=0.304929
epochs==50 and use_CRF==False and no golve_path f1_score==0.63255478, correct/total=0.8984388
考虑char时
epoches==50 and use_CRF==True and glove_path==True f1_score==0.6819606,correct/total=0.9239794

观察实验中loss值的变化可以看出在随机初始化的词嵌入时，每一个epoch的loss值都高于经过预训练的glove词向量的情况下的loss值
