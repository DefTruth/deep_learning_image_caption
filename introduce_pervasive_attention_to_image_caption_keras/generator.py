# -*- coding: utf-8 -*-
import os
import re
import tqdm
from PIL import ImageFile
from keras.preprocessing import image
from keras.utils import to_categorical
import pickle
import codecs
import numpy as np
import random
from keras.applications.imagenet_utils import \
    preprocess_input as imagenet_preprocess_input
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 图片预处理函数
def preprocess_input(x, preprocess_method='center'):
    """
    :param x: 输入的图像像素矩阵[array]
    :param preprocess_method: 'center'表示将每张图片压缩到[-1, 1];
        'imagenet'表示减imagenet采用的数据预处理[str]
    :return: 处理后的图像像素矩阵
    """
    if preprocess_method == 'center':
        x /= 255.
        x -= 0.5
        x *= 2.
        return x
    elif preprocess_method == 'imagenet':
        return imagenet_preprocess_input(x)


# 生成训练/验证/测试的train_filename_list
# /valid_filename_list/test_filename_list
def generate_filename_list(total_num=10000,
                           train_prob=0.95,
                           valid_num=200,
                           data_path='./datas/img_data/'):
    """
    生成训练和测试对应的文件名列表，图片的文件名必须以数值命名，格式为png
    :param total_num: 样本量总数[int]
    :param train_prob: 训练数据占比[float]
    :param valid_num: 验证集样本量[int]
    :param data_path: 数据路径[str]
    :return:
    """
    file_paths = os.listdir(data_path)
    filename_lst = [x[:-4] for x in file_paths]
    if total_num != len(filename_lst):
        total_num = len(filename_lst)
        print('total num is update as {0}.'.format(len(filename_lst)))
    cal_train_num = int(total_num * train_prob)
    train_num = cal_train_num - valid_num
    train_lst = random.sample(filename_lst, train_num)
    rest_lst = [x for x in filename_lst if x not in train_lst]
    valid_lst = random.sample(rest_lst, valid_num)
    test_lst = [x for x in rest_lst if x not in valid_lst]

    return train_lst, valid_lst, test_lst


# 统计词频
def word_count(is_count=True,
               input_data_path='./datas/caption_text_remove.txt',
               output_data_path='./datas/input_words_count.pkl'):
    """
    统计词频， 注意， 这里统计所用样本的词频而不仅仅是训练样本
    逐行统计，不将文本数据加载到内存
    :param is_count: 是否重新统计词频[boolean]
    :param input_data_path: 加载文本数据的路径, caption_text是分词后的数据 [str]
    :param output_data_path: 加载或输出词频统计dict的路径[str]
    :return:
    """
    input_words_count = Counter()
    if is_count:
        # 统计词频
        with codecs.open(input_data_path, 'r', encoding='utf-8') as fr:
            for line in tqdm.tqdm(fr):
                # 分隔符因人而异，我用的是'_SPLIT_'
                line = re.split(pattern='_SPLIT_', string=line)
                sentence = line[-1]
                # 统计词频时需要保留\t \n
                sentence = sentence.split(' ')
                for word in sentence:
                    input_words_count[word] += 1

        with codecs.open(output_data_path, 'wb') as fw:
            pickle.dump(input_words_count, fw)
    else:
        # 直接加载词频数据
        if os.path.exists(output_data_path):
            with codecs.open(output_data_path, 'rb') as f:
                input_words_count = pickle.load(f)
        else:
            raise Exception('{0} is not exist.'.format(output_data_path))

    return input_words_count


# 统计句子长度
def sentence_len(input_data_path='./datas/caption_text_remove.txt',
                 output_data_path='./datas/sen_len_count.pkl'):
    """

    :param input_data_path: 加载文本数据的路径, caption_text是分词后的数据
    :param output_data_path:  加载或输出句长统计dict的路径[str]
    :return:
    """
    sen_len_count = Counter()
    len_val = []
    with codecs.open(input_data_path, 'r', encoding='utf-8') as fr:
        for line in tqdm.tqdm(fr):
            line = re.split(pattern='_SPLIT_', string=line)
            sentence = line[-1]
            # 统计句子长度不需要保留\t \n
            sentence = sentence.strip().split(' ')
            sen_len = len(sentence)
            len_val.append(sen_len)
            sen_len_count[str(sen_len)] += 1

    with codecs.open(output_data_path, 'wb') as fw:
        pickle.dump(sen_len_count, fw)

    # 统计分位数
    percents = [25, 50, 75, 85, 90, 95, 100]
    for percent in percents:
        percentile = np.percentile(len_val, percent)
        print('percent {0} value is: {1}'.format(percent, percentile))

    return sen_len_count


# 建立词典get_token
def get_token(input_words_count=None,
              input_word_num=4000):
    """
    按照指定的词汇量大小，对词汇进行截取，并生成词汇字典和逆词汇字典
    :param input_words_count: 输入词汇词频[dict]
    :param input_word_num: 输入词汇大小[int]
    :return:
    """
    # 对词汇出现的次数进行排序
    input_words_count = sorted(input_words_count.items(), key=lambda x: x[1], reverse=True)
    # 选取前word_num个词汇，其他字符集统一用未知词汇符号，即UNK表示
    input_words_count_select = input_words_count[:input_word_num]
    # 对词汇进行排序
    input_words = sorted([i[0] for i in input_words_count_select])

    # 增加未知字符
    input_words.append('<UNK>')
    # 增加pad字符
    input_words = ['<PAD>'] + input_words
    # 构建词汇集字典
    input_token_index = dict([(word, i) for i, word in enumerate(input_words)])
    # 词汇逆字典
    reverse_input_word_index = dict((i, word) for word, i in input_token_index.items())

    return input_token_index, reverse_input_word_index


# 转换成one-hot或索引sequence
# one-hot时采用categorical_crossentropy
# sequence时采用sparse_loss
def text_to_seq(batch_texts=None,
                token_index=None,
                max_seq_length=200,
                return_one_hot=True,
                word_num=4000):
    """
    将文本转化为整数序列的形式
    batch_texts中文本的数目可以是任意的, 当为全部caption_text.txt对应的文本时会将所有文本数据读入内存
    一次性处理；当batch_texts只是某一部分文本时，则返回部分的数值化sequence,采用生成器训练时用

    :param batch_texts: 文本数据 [list], 每个元素就是用空格符分隔的文本
    :param token_index: 文本词汇字典 [dict], e.g input_token_index
    :param max_seq_length: 文本的最大长度 [int], default 200
    :param return_one_hot: 是否返回one-hot形式 [boolean] default False
    :param word_num: 最大词汇数量/词表大小 [int] e.g input_word_num 4000
    :return:
    """

    # 将输出数据都转化为整数序列格式
    batch_input_data = np.zeros((len(batch_texts), max_seq_length), dtype='int')
    batch_target_data = np.zeros((len(batch_texts), max_seq_length), dtype='int')

    for i, target_text in enumerate(batch_texts):
        target_text = target_text.split(' ')
        if len(target_text) > max_seq_length:
            target_text = target_text[:(max_seq_length - 1)]
            target_text.append('\n')
        for t, word in enumerate(target_text[:max_seq_length]):
            # decoder_target_data要比decoder_input_data早一个时间步
            try:
                batch_input_data[i, t] = token_index[word]
            except:
                batch_input_data[i, t] = token_index['<UNK>']
            if t > 0:
                try:
                    batch_target_data[i, t - 1] = token_index[word]
                except:
                    batch_target_data[i, t - 1] = token_index['<UNK>']

    # 将decoder_target_data转化为one-hot形式
    if return_one_hot:
        batch_target_data = to_categorical(batch_target_data, num_classes=(word_num + 2))
        return batch_input_data, batch_target_data
    else:
        return batch_input_data, batch_target_data


# 获取数据集，train=True获得训练集，train=False获得测试集合
def load_img_data(filename_lst=None,
                  image_size=(224, 224),
                  preprocess_method='center',
                  data_path='./datas/img_data/'):
    """
    加载图片数据/生成对应文件件名（图片序号) 图片的格式为png
    :param image_size: 图片的target_size.[tuple]
    :param filename_lst: 图片对应的图片序号 [list], 可以为一个batch对应的数量
    :param preprocess_method: 预处理所用的方法, center/mean.[str]
    :param data_path: 图片的存储路径.[str]
    :return:
    """

    if isinstance(filename_lst, list):
        file_paths = [os.path.join(data_path, str(x) + '.png') for x in filename_lst]
    else:
        return None

    data = []
    try:
        for path in file_paths:
            img = image.load_img(path, target_size=image_size)
            img = image.img_to_array(img)
            img = preprocess_input(img, preprocess_method=preprocess_method)
            data.append(img)
    except:
        return None

    # 将数据集转化为数组
    if not bool(data):
        return None
    if isinstance(data, list):
        try:
            return np.asarray(data)
        except:
            return None


def load_one_img(file_path=None,
                 image_size=(224, 224),
                 preprocess_method='center'
                 ):
    """
    只处理一张图片，可用于单独测试某张图片
    :param file_path: 图片完整的绝对路径[str]
    :param image_size: 图片的像素大小[tuple]
    :param preprocess_method: 图片预处理方法 center/mean[str]
    :return:
    """

    img = image.load_img(file_path, target_size=image_size)
    img = image.img_to_array(img)
    img = preprocess_input(img, preprocess_method=preprocess_method)

    return img


# 加载文本数据并序列化，直接根据输入的filename_lst从correct_text.txt加载
def load_text_data(filename_lst=None,
                   token_index=None,
                   max_seq_length=200,
                   return_one_hot=True,
                   word_num=4000,
                   data_path='./datas/caption_text/'):
    """
    加载filename_lst中指定的index对应的文本数据并进行序列化
    :param filename_lst: 文本对应的index [list], 可以为一个batch对应的数量
    :param token_index: 文本词汇字典 [dict], e.g input_token_index
    :param max_seq_length: 文本的最大长度 [int], default 200
    :param return_one_hot: 是否返回one-hot形式 [boolean] default False
    :param word_num: 最大词汇数量/词表大小 [int] e.g input_word_num 4000
    :param data_path: 原始caption_text.txt文件所在的路径 [str]
    :return:
    """
    filename_lst = [str(x) for x in filename_lst]
    filename_lst = [os.path.join(data_path, x + '.txt') for x in filename_lst]

    texts_data = []
    for file_path in filename_lst:
        with codecs.open(file_path, 'r', encoding='utf-8') as fr:
            texts_data.append(fr.read())

    if bool(texts_data) and len(texts_data) == len(filename_lst):
        try:
            batch_input_data, batch_target_data = text_to_seq(batch_texts=texts_data,
                                                              token_index=token_index,
                                                              max_seq_length=max_seq_length,
                                                              return_one_hot=return_one_hot,
                                                              word_num=word_num)
            return batch_input_data, batch_target_data
        except:
            return None, None
    else:
        return None, None


# 编写生成训练数据的生成器generator
def train_fit_generator(train_lst=None,
                        token_index=None,
                        return_one_hot=True,
                        batch_size=32,
                        image_size=(224, 224),
                        max_seq_length=200,
                        word_num=4000,
                        preprocess_method='center',
                        img_data_path='./datas/img_data/',
                        text_data_path='./datas/caption_text/'):
    """
    训练数据生成器
    :param train_lst: 训练集对应的filename_list [list]
    :param token_index: 词典，input_token_index [dict]
    :param return_one_hot: 是否返回one-hot
    :param batch_size: batch大小 [int]
    :param image_size: 图像的像素大小 [tuple]
    :param max_seq_length: 文本句子的最大长度 [int]
    :param word_num: 词表大小, input_word_num [int]
    :param preprocess_method: image的预处理方法 [str]
    :param img_data_path: 图像数据的绝对路径 [str]
    :param text_data_path: 文本数据的绝对路径 [str]
    :return:
    """
    # 随机打乱
    random.shuffle(train_lst)
    batches = int(len(train_lst) / batch_size)
    # 生成一个batch的数据
    while 1:
        for count in range(batches):

            start_batch_count = count * batch_size
            end_batch_count = start_batch_count + batch_size
            batch_index_lst = train_lst[start_batch_count: end_batch_count]
            batch_train_img = load_img_data(filename_lst=batch_index_lst,
                                            image_size=image_size,
                                            preprocess_method=preprocess_method,
                                            data_path=img_data_path)
            batch_input_data, batch_target_data = load_text_data(filename_lst=batch_index_lst,
                                                                 token_index=token_index,
                                                                 max_seq_length=max_seq_length,
                                                                 return_one_hot=return_one_hot,
                                                                 word_num=word_num,
                                                                 data_path=text_data_path)
            if batch_train_img is not None \
                    and batch_input_data is not None \
                    and batch_target_data is not None:

                yield ([batch_train_img,
                        batch_input_data],
                       [batch_target_data])
            else:
                print('\n continue')
                continue


# 简单生成指定batch_lst的index对应的数据
def get_batch_data(batch_lst=None,
                   token_index=None,
                   return_one_hot=False,
                   image_size=(224, 224),
                   max_seq_length=200,
                   word_num=4000,
                   preprocess_method='center',
                   img_data_path='./datas/img_data/',
                   text_data_path='./datas/caption_text/'
                   ):
    """
    :param batch_lst: 训练集对应的filename_list[list]
    :param token_index: 词典，input_token_index[dict]
    :param return_one_hot: 是否返回one-hot[boolean]
    :param image_size: 图像的像素大小[tuple]
    :param max_seq_length: 文本句子的最大长度[int]
    :param word_num: 词表大小, input_word_num[int]
    :param preprocess_method: image的预处理方法[str]
    :param img_data_path: 图像数据的绝对路径[str]
    :param text_data_path: 文本数据的绝对路径[str]
    :return:
    """
    batch_train_img = load_img_data(filename_lst=batch_lst,
                                    image_size=image_size,
                                    preprocess_method=preprocess_method,
                                    data_path=img_data_path)
    batch_input_data, batch_target_data = load_text_data(filename_lst=batch_lst,
                                                         token_index=token_index,
                                                         max_seq_length=max_seq_length,
                                                         return_one_hot=return_one_hot,
                                                         word_num=word_num,
                                                         data_path=text_data_path)
    if batch_train_img is not None \
            and batch_input_data is not None \
            and batch_target_data is not None:

        return ([batch_train_img,
                 batch_input_data],
                [batch_target_data])
    else:
        return None

