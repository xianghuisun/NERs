# 关于命名实体识别的一些典型方法的学习

data文件夹中的自然就是数据。

log文件夹是保存训练后的模型的检查点，当然每次模型架构不同，里面的检查点都会重新生成的。

HMM.py就是隐马尔科夫模型的代码，直接运行即可出现论文中给出的结果。

process_data.py是处理数据的代码，目的是为了得到词嵌入矩阵，并且将每一个句子的每一个单词转换成对应的整数值，这样句子就可以送进神经网络了
char_data_process.py是在考虑字符层面的表示下处理字符的代码，目的是得到字符的嵌入矩阵，并且将每一个句子的每一个单词的每一个字符都转成对应的整数值。然后就可以利用CNN或者BiLSTM来学习这些特征。

BiLSTM-CRF.py和BiLSTM-CNN-CRF.py可以直接运行，要想调整模型的架构比如说不用CRF或者不用CNN只需要修改里面的参数即可。各个模型的结果在论文中已经给出这里不再给出图片了。

paper文件夹包含写的论文，以及各种模型运行后的结果。论文是用latex排版，源码在presentation.tex文件中。需要有LaTeX环境才能编译，presentation.pdf就是论文。
下面就是各个模型的结果，在报告中已经给出

![s](https://github.com/xianghuisun/NERs/tree/master/paper/result.png)

