# -*- coding: utf-8 -*-
import os


class Params:
    """
    Parameters of our model
    """
    # 配置样本参数
    img_total_num = 46795  # 总样本量
    img_train_prob = 0.95  # 训练样本比例
    img_valid_num = 200  # 验证集样本量
    is_count = True  # 是否重新统计词频
    return_one_hot = True  # 是否采用one-hot形式计算损失

    # 配置公共路径
    root_data_path = '/datas/'  # 基础数据公共路径
    img_data_path = os.path.join(root_data_path, 'img_data/')  # 图片数据公共路径
    text_data_path = os.path.join(root_data_path, 'caption_text/')  # 文本数据公共路径
    text_remove_path = os.path.join(root_data_path, 'caption_text_remove.txt')  # 文本数据公共路径 这个路径不一定有用 看实际情况

    # 配置增强后的数据路径
    augment_img_path = os.path.join(root_data_path, 'augment_img_data/')  # 增强后的图像数据
    augment_text_path = os.path.join(root_data_path, 'augment_caption_text/')  # 增强后的文本数据

    # 配置项目路径
    temp_data_path = './temp_data/'  # 临时数据项目路径
    log_path = './log/'  # 日志文件路径

    # 配置模型参数
    trainable_imagenet = False  # 是否对imagenet的参数进行训练
    blocks = [2, 2, 2, 4, 4, 4]  # pervasive attention model的结构
    dense_layers = int(sum(blocks) * 2 + 5)
    null_token_value = 0  # <PAD>对应的索引
    image_size = 224  # 图片大小
    max_dec_len = 140  # caption的最大长度
    dec_word_num = 7000  # 词表大小
    embedding_dim = 256  # Decoder 的hidden-state的维度

    batch_size = 32  # batch 32/64...
    growth_rate = 32  # densenet 的 growth rate/每一次conv之后的通道数
    reduction = 0.5  # transition layer中的reduction rate，即保留的feature-maps数目
    dropout = 0.5  # dropout rate in conv block

    # 配置生成器参数
    steps_per_epoch = int(img_total_num / batch_size) - 10  # 生成器每生成多少次数据记一个epochs结束
    workers = 4  # 线程并行数

    # 配置训练参数
    epochs = 200  # 迭代次数
    learning_rate = 0.0005  # 学习率
    decay = 1 / steps_per_epoch * 10  # 10个epochs之后降到1/2
    clip = 5.  # 防止梯度爆炸

    # 配置图像预处理参数
    preprocess_method = 'center'


