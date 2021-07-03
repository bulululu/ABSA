#  一 任务介绍

基于属性的情感分析（Aspect Based Sentiment Analysis, ABSA）是一种细粒度的情感分析任务，旨在识别一条句子中一个指定属性（Aspect）的情感极性。ABSA的两个任务为情感对象识别（Aspect Term Extraction）与情感对象倾向分析（Aspect Term Polarity Analysis）。从本源上看，前者是一个NLP的标注任务（如命名实体识别等），后者则是一个分类任务（针对性地判断Aspect的情感类别）。

通常有4个子任务：

- ATSA（Aspect Term Sentiment Analysis）

  一个Aspect term是句子中的一个词或词组，该任务可以建模为一个分类问题。

- ACSA（Aspect Category Sentiment Analysis）

  一个Aspect category是句子中隐式表达的描述事物的一个预先定义的角度，Aspect category来自一个预先定义好的集合，其不必显式地出现在句子中。该任务可以建模为一个分类问题。

- Aspect Term Extraction

  该任务可以建模为一个序列标注问题。

- Aspect Category Extraction

  该任务可以建模为一个多标签分类问题。

# 二 数据集介绍

- [SemEval 14 Restaurant Review数据集](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

  数据集中包含ATSA和ACSA两种版本，共有四千多条数据，分为训练集和测试集。ATSA的数据集也可以用来做Aspect Term Extraction，ACSA的数据集也可以用来做Aspect Category Extraction。

- [SemEval 14 Laptop Review数据集](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools)

  数据集中只有ATSA版本，共有三千多条数据，也分为训练集和测试集。Laptop数据集相比Restaurant数据集有更多隐式表达情感的样本，数据量更少，难度更大。

- [Twitter数据集](https://github.com/songyouwei/ABSA-PyTorch/tree/master/datasets/acl-14-short-data)

  数据集中只有ATSA版本，共有六千多条数据，分为训练集和测试集。Twitter数据集相比Restaurant数据集和Laptop数据集质量较低。

- [MAMS数据集](https://github.com/siat-nlp/MAMS-for-ABSA)

  数据集中包含ATSA和ACSA两种版本，共有一万多条数据。MAMS的特点是，一个句子中一定包含至少两个Aspect，并且同一个句子中至少有两个Aspect情感极性是不同的。而Restaurant，Laptop和Twitter这三个数据集中，大多数句子只包含一个Aspect或者包含多个相同情感的Aspect，这样会造成基于方面的情感分析任务退化成句子级别的情感分析任务。

# 三 经典模型概述

## [1 TD-LSTM](https://www.aclweb.org/anthology/C16-1311.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/td_lstm.py)
- [LSTM讲解](https://blog.csdn.net/weixin_44857688/article/details/113972364?spm=1001.2014.3001.5502)
- TD-LSTM使用两个LSTM分别去建模target和左边的上下文以及target和右边的上下文，将两个LSTM最后的隐含向量拼接送入到一个softmax分类器中进行分类。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8aw36uj31j80fmwpc.jpg" alt="image-20210630155852143" style="zoom: 33%;" />



## [2 TC-LSTM](https://www.aclweb.org/anthology/C16-1311.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/tc_lstm.py)

- TC-LSTM将Target词向量的平均值与句子中每个词的词向量进行拼接然后进行和TD-LSTM一样的操作。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8gvfyxj31ji0eqqdg.jpg" alt="image-20210630155914545" style="zoom:33%;" />



## [3 ATAE-LSTM](https://www.aclweb.org/anthology/D16-1058.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/atae_lstm.py)

- [Attention讲解](https://blog.csdn.net/weixin_44857688/article/details/113972364?spm=1001.2014.3001.5502)

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8lk4zgj30zu0mcwm6.jpg" alt="image-20210701141420643" style="zoom: 50%;" />

- 理解Q、K、V
  - [Transformer讲解](https://blog.csdn.net/weixin_44857688/article/details/114002905?spm=1001.2014.3001.5502)
  - [[李宏毅]NLP领域BERT大火却不懂Transformer？](https://www.bilibili.com/video/BV1GE411o7XE)
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - <img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8rzihwj313q06oq61.jpg" alt="image-20210701111922321" style="zoom:25%;" />
  - <img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8vcuoqj31j407sgrd.jpg" alt="image-20210701111944054" style="zoom: 25%;" />

- 句子中的每个词首先被转换成词向量表示，使用LSTM编码得到隐层状态。使用一个可学习的参数向量作为query，aspect的embedding经过线性变换后的表示和隐层状态的拼接作为key，隐层状态作为value计算attention。输出的上下文向量和最后一个隐层状态拼接后做一次非线性变换得到最终的分类特征，再经过一个softmax分类器预测每个类别的概率分布。为了更好地利用aspect的信息，在embedding层将每个词的embedding和aspect的embedding进行拼接，得到aspect相关的词表示。后续的操作和AT-LSTM一样。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j8yogcqj31fm0sedzl.jpg" alt="image-20210630160944192" style="zoom:33%;" />

## [4 MemNet](https://www.aclweb.org/anthology/D16-1021.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/memnet.py)

- 句子中的每个词首先被转换成词向量表示，整个句子中所有词的词向量就组成了一个矩阵，称为memory。aspect中所有词的词向量取平均作为初始的aspect表示。每一个hop包含一个attention层和一个线性层。attention层使用上一步得出的aspect表示作query，计算query和句子中每个词的词向量的相似度，对词向量进行加权求和得到上下文向量。上下文向量再与经过线性变换的aspect进行求和得到此步骤中的aspect表示。相同结构的hop堆叠多次，使用不同层次的aspect表示作为query提取上下文特征，最终得到的aspect表示经过softmax分类器进行分类。由于 attention机制不具有建模位置信息的能力，这篇文章中首次提出了位置编码以在模型中编码位置信息。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j92euvxj30zi0sugxt.jpg" alt="image-20210630160550897" style="zoom: 33%;" />

## [5 RAM](https://www.aclweb.org/anthology/D17-1047.pdf)

- [代码](https://github.com/lpq29743/RAM)
- 提出的RAM从两方面改进了MemNet。MemNet直接使用句子中每个词的embedding作为memory，而RAM使用BiLSTM计算得到的隐层状态作为memory。与MemNet相似，RAM也有一个基于位置对memory进行加权的机制，即Location Weighted Memory。另外，RAM使用GRU来更新aspect的表示。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j95orpaj31ea0qw4ew.jpg" alt="image-20210630161327225" style="zoom:33%;" />

## [6 IAN](https://arxiv.org/pdf/1709.00893.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/ian.py)
- 提出一种交互式注意力网络模型IAN，计算context和target之间的交互式注意力，不仅使用target信息选择context中重要的词，也使用context信息选择target中重要的词。具体的，target和context分别使用一个LSTM计算隐层状态，将target和context的隐层状态进行平均池化得到固定长度的target和context的向量表示。使用target的池化表示作为query，context的隐层状态作为key和value计算attention，得到target相关的context表示。同样的，使用context的池化表示作为query，target的隐层状态作为key和value计算attention，得到context相关的target表示。将两者拼接起来作为分类特征经过一个线性层和tanh激活得到对数几率。再经过softmax归一化得到每个类别的概率分布。简单鲁棒，效果很好。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j98w5sxj30rw0oswrv.jpg" alt="image-20210630161218969" style="zoom: 50%;" />

## [7 GCAE](https://www.aclweb.org/anthology/P18-1234.pdf)

- [代码](https://github.com/wxue004cs/GCAE)

- [TextCNN讲解](https://blog.csdn.net/weixin_44857688/article/details/113951156?spm=1001.2014.3001.5502)

- 提出了一种基于CNN的方法GCAE。针对ACSA和ATSA两个子任务有两种不同的变体。总体思路是使用一组成对的卷积核提取局部n-gram特征，每对卷积核中包含一个aspect无关的卷积核和一个aspect相关的卷积核。aspect无关的卷积核像TextCNN里的卷积一样提取句子中的n-gram特征，并使用tanh激活。aspect相关的卷积核作为门来控制n-gram特征输出与否。值得注意的是，这里使用relu而不是常见的sigmoid作为门。作者做了对比实验，relu比sigmoid更好。tanh函数是用于捕捉情感特征，relu用于捕捉aspect信息。对于ACSA任务，aspect为固定种类的aspect category，使用可学习的embedding作为其表示。对于ATSA任务，aspect是句子中的一个子序列，使用CNN提取局部特征再在时间维度上做最大池化得到。模型简单有效，速度快效果好鲁棒性好。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j9c4cd6j31dg0naakr.jpg" alt="image-20210630161557413" style="zoom:33%;" />

## [8 AOA-LSTM](https://arxiv.org/pdf/1804.06536.pdf)

- [代码](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aoa.py)

- 提出的AOA-LSTM借鉴了问答系统里的attention-over-attention机制。首先使用两个BiLSTM分别提取句子和aspect的特征。然后attention-over-attention模块使用点积计算句子的隐层状态和aspect的隐层状态的交互矩阵。对交互矩阵按列归一化得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) ，按行归一化得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 。对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta) 矩阵求列平均值得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7B%5Cbeta%7D+) ，以 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbar%7B%5Cbeta%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 相乘，得到最终的句子中每个词的权重 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 。使用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 作为权重对句子中每个词做加权平均得到最终的分类特征向量。再经过softmax分类器得到最终的分类向量。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3j9fz01sj31c40n6dtt.jpg" alt="image-20210630174906707" style="zoom: 50%;" />

# 四 BERT模型

- [Exploiting BERT for End-to-End Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/D19-5505.pdf)
  - [代码](https://github.com/lixin4ever/BERT-E2E-ABSA)
  - <img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3m2ilaztj61fq0bkjz902.jpg" alt="image-20210703114214768" style="zoom: 25%;" />
  - [CRF讲解](https://www.bilibili.com/video/BV1K54y117yD)
  - 给定输入标记序列X，我们首先使用L个Transformer的BERT分量来计算相应的上下文表示H。上下文化的表示H被提供给特定于任务的层E2E-ABSA(线性层、RNN、Self-Attention Network、条件随机场层)，以预测标签序列。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3ja9qfxyj30yg0scgoz.jpg" alt="image-20210703095251175" style="zoom: 40%;" />

- [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/N19-1242.pdf)
  - [代码](https://github.com/howardhsu/BERT-for-RRC-ABSA)
  - 作者认为BERT在wiki等语料集上进行训练造成了一些“领域的偏差”，因此模型需要获取一些领域相关知识和任务相关知识。使用BERT典型的对无标记数据的两个训练任务进行了训练。这两个任务分别是MLM ( masked language model)，和NSP ( next sentence prediction)。简单描述一下这两个任务，MLM指将文本中一些词随机用[MASK]替代，并让模型预测它们到底是什么词。NSP将两句话拼接，让模型预测它们是否是来自同一个文本。针对会出现的显存容量占用过多的问题，作者提出可以将一个batch分为小batch，并将这些小batch对参数的更新值计算出来先不更新，而是将一个batch的小batch更新值加起来平均，然后更新。
  - RRC：模型将问题和评论拼接起来[CLS], ![[公式]](https://www.zhihu.com/equation?tex=q_{1}%2C...%2Cq_{m}) , [SEP], ![[公式]](https://www.zhihu.com/equation?tex=d_{1}%2C...%2Cd_{n}), [SEP] 作为输入，预测了评论中每个单词作为答案的开始和结尾的概率，然后与真实答案的位置计算损失。
  - AE：在监督学习中，这一问题可以看作是一个序列标注问题。对于一个m长度的输入句子，整理为[CLS], ![[公式]](https://www.zhihu.com/equation?tex=x_{1}%2C...%2Cx_{m}), [SEP]作为模型输入，经过BERT提取特征后，把特征向量输入到全连接层，接softmax激活，最终得到序列中每个位置的结果表示为{Begin, Inside，Outside}，损失函数就是在所有位置上预测label与标注label的交叉熵。
  - ASC：这一任务的输入有两部分组成：即属性和提及该属性的评论文本。因此这一任务和RRC是类似的，可以把属性看作question，而评论文本看作是document，但是与RRC不同的是情感分析的输出只有类别label而不是一个文本区间。具体细节不再赘述。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3jd7t68pj30l40ug42e.jpg" alt="image-20210703100843847" style="zoom:50%;" />

- [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf)
  - [代码](https://github.com/HSLCY/ABSA-BERT-pair)
  
  - 这篇论文主要是通过用aspect构造辅助句子，将ABSA问题转成sentence-pair分类任务。
  
    以句子“LOCATION1 is central London so extremely expensive”为例。
  
    - QA-M：辅助句子"what do you think of the price of LOCATION1"。
  - NLI-M： 辅助句子"LOCATION1- price"。
    - QA-B： 辅助句子"the polarity of the aspect price of LOCATION1 is positive/none/negative"，标记为yes/no。构造三个句子（positive, none, negative），选择yes得分最高的作为预测类别。
  - NLI-B：辅助句子"LOCATION1- price - positive/none/negative"，预测方法同QA-B。

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gs3snaa67nj30mi0by0v5.jpg" alt="image-20210703152945572" style="zoom: 50%;" />

# 五 拓展资料

- [数据、论文与代码](https://github.com/jiangqn/Aspect-Based-Sentiment-Analysis)

