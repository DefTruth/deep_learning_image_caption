Introduction:
 最近一直在做Image-Caption的任务，在Fine-tuning和CNN-RNN的基础结构上尝试了加入了attention机制，这是目前处理Image-Caption任务的流行架构。然后发现今年8月份有一篇非常有意思的paper《Pervasive Attention: 2D Convolutional Neural Networks for Sequence-to-Sequence Prediction》在sep2seq任务上完全抛弃了Ecoder-Decoder结构，采用CNN结果完成生成任务。看完后，我尝试将这种pervasive-attention model迁移到手上正在做的Image-caption任务上来，发现效果也不错。关于pervasive-attention是如何应用在seq2seq任务上的，可以参考这篇博客：

    原文链接：https://blog.csdn.net/linchuhai/article/details/82585803
默认数据的存放方式：

images：png文件，从0开始以数值作为文件名，存放在/datas/img_data

captions: txt文件，从0开始以数值作为文件名，与images中的图片一一对应, 存放在/datas/caption_text

读者需要根据自己的情况更改数据路径才能运行代码



folder:
some empty folders are not upload in this project, you have to create yourself.

1.adaptive-attention-lstm:Adaptive-Attention-LSTM模型的复现，（该项目中基本不会用到）

	(1).adaptive-attention-lstm.py: Knowing-When-To-Look(paper)

2.models:训练完的模型保存路径，10个epochs保存一次

3.temp_data:在train时会生成以下临时数据，在测试时可以直接调用

```
(1).input_token_index.pkl:词表索引

(2).input_words_count.pkl:词频统计信息

(3).reverse_input_word_index.pkl:词表逆词典

(4).sen_len_ocunt.pkl:文本长度统计

(5).test_lst.pkl:测试样本的index列表

(6).train_lst.pkl:训练样本的index列表

(7).vaild_lst.pkl:验证集的index列表
```

4.test:测试样本的测试结果, 其中test_50用于保存50epochs模型的测试结果，其他类似

5.densenet:用于fine-tuning的densenet模型的保存路径

6.img_person:一些样本数据



py:

`1.data_augment_outline.py:离线数据增强`

`2.generator.py:生成器脚本（数据预处理， 不包括处理多标签数据）; 用于使用生成器方式训练模型`

`3.params.py:初始化参数脚本`

`4.finetune_densenet.py:编写用于fine-tuning的densenet模型`

`5.show_image.py:用于展示预测后的样本`

`6.model.py:编写模型`

`7.train.py:训练模型`

`8.test.py:测试模型`









