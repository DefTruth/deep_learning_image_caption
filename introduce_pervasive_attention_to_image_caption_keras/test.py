# -*- coding: utf-8 -*-
import model as md
import os
import show_image as sh
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import codecs
import pickle
import generator
from params import Params

# filename
filename = 'emb{0}_batch{1}_layer{2}_dropout{3}_g{4}_reduc{5}'. \
    format(Params.embedding_dim,
           Params.batch_size,
           Params.dense_layers,
           str(Params.dropout),
           Params.growth_rate,
           str(Params.reduction))

# 加载词典/逆词典/词频/测试文件名等
model_temp_data_path = os.path.join(Params.temp_data_path, filename)
if not os.path.exists(model_temp_data_path):
    raise OSError('There is no such path named: {0}.'.format(model_temp_data_path))
# 路径存在则直接加载
with codecs.open(os.path.join(model_temp_data_path, 'input_token_index.pkl'), 'rb') as fi:
    input_token_index = pickle.load(fi)
with codecs.open(os.path.join(model_temp_data_path, 'reverse_input_word_index.pkl'), 'rb') as fir:
    reverse_input_word_index = pickle.load(fir)
with codecs.open(os.path.join(model_temp_data_path, 'train_lst.pkl'), 'rb') as ftr:
    train_lst = pickle.load(ftr)
with codecs.open(os.path.join(model_temp_data_path, 'valid_lst.pkl'), 'rb') as fva:
    valid_lst = pickle.load(fva)
with codecs.open(os.path.join(model_temp_data_path, 'test_lst.pkl'), 'rb') as fte:
    test_lst = pickle.load(fte)
with codecs.open(os.path.join(model_temp_data_path, 'input_words_count.pkl'), 'rb') as ftw:
    input_words_count = pickle.load(ftw)

# 生成测试数据
if os.path.exists(os.path.join(model_temp_data_path, 'test_texts.pkl')) \
        and os.path.exists(os.path.join(model_temp_data_path, 'test_img.pkl')):
    with codecs.open(os.path.join(model_temp_data_path, 'test_texts.pkl'), 'rb') as ftt:
        test_texts = pickle.load(ftt)
    with codecs.open(os.path.join(model_temp_data_path, 'test_img.pkl'), 'rb') as fti:
        test_img = pickle.load(fti)
else:
    test_file = [str(x) for x in test_lst]
    test_file = [os.path.join(model_temp_data_path, x + '.txt') for x in test_file]

    test_texts = []
    for path in test_file:
        # 需要还原分词前的状态
        with codecs.open(path, 'r', encoding='utf-8') as ft:
            line = ft.read()
            line = line.strip().split(' ')
            line = ''.join(line)
            test_texts.append(line)

    test_img = generator.load_img_data(filename_lst=test_lst,
                                       image_size=Params.image_size,
                                       preprocess_method=Params.preprocess_method,
                                       data_path=Params.img_data_path)

    with codecs.open(os.path.join(model_temp_data_path, 'test_texts.pkl'), 'wb') as f:
        pickle.dump(test_texts, f)
    with codecs.open(os.path.join(model_temp_data_path, 'test_img.pkl'), 'wb') as f:
        pickle.dump(test_img, f)


# 编写测试函数
def rebuild_decoder(model_path=None):
    """
    重建模型
    :param model_path: 模型权重的加载路径 [str]
    :return:
    """
    # 重建model
    base_model = md.pervasive_attention(blocks=Params.blocks,
                                        trainable=Params.trainable_imagenet,
                                        image_size=Params.image_size,
                                        growth_rate=Params.growth_rate,
                                        reduction=Params.reduction,
                                        dropout=Params.dropout,
                                        max_dec_len=Params.max_dec_len,
                                        embedding_dim=Params.embedding_dim,
                                        dec_word_num=Params.dec_word_num,
                                        learning_rate=Params.learning_rate,
                                        decay=Params.decay,
                                        clip=Params.clip)

    base_model.load_weights(model_path)
    # 只关注前两个输入

    return base_model


def predict(predict_model=None,
            image_input=None,
            ):
    """
    得到预测序列, step by step.
    :param predict_model: 预测模型
    :param image_input: 需要预测的图片 [array]
    """
    start_word = ["\t"]
    index = 0

    while True:
        decoder_input = [input_token_index[i] for i in start_word]
        decoder_input = pad_sequences([decoder_input], maxlen=Params.max_dec_len,
                                      padding='post', value=Params.null_token_value)
        decoder_output = predict_model.predict([image_input, np.array(decoder_input)])
        word = reverse_input_word_index[np.argmax(decoder_output[0][index])]

        start_word.append(word)
        if word == '\n' or len(start_word) > Params.max_dec_len:
            break

        index += 1

    return "机器描述：" + ''.join(start_word[1:-1])


def test_model(test_info=None,
               show_org=True,
               save_root_path='./test/test_50/',
               model_path=None):
    """
    :param test_info: 测试数据, e.g (test_lst, test_texts, test_img) [tuple]
    :param show_org: 是否展示人工评语[boolean]
    :param save_root_path: 保存测试结果的根路径[str]
    :param model_path: 测试所需要的模型[str]
    :return:
    """
    names, texts, images = test_info
    predict_model = rebuild_decoder(model_path=model_path)

    for name, text, image in zip(names, texts, images):
        file_path = os.path.join(Params.img_data_path, str(name) + '.png')
        save_path = os.path.join(save_root_path, str(name) + '.png')
        predict_text = predict(image_input=image,
                               predict_model=predict_model)
        print("图片标号：" + str(name))
        original_text = "人工描述：" + text
        print(original_text)

        sh.show_image(file_path=file_path,
                      original_text=original_text,
                      predict_text=predict_text,
                      save_path=save_path,
                      show_org=show_org)

        print(predict_text)


def test_one_picture(file_path='./datas/img_person/0.png',
                     save_path='./test/test_100/0.png',
                     show_org=False,
                     model_path=None):
    """
    只测试一张图片
    :param file_path: 输入单张图片完成的绝对路径[str]
    :param save_path: 保存单张图片的绝对路径[str]
    :param show_org: 是否展示原始人工评语[boolean]
    :param model_path: 测试模型路径[str]
    :return:
    """
    st = time.time()
    img_data = generator.load_one_img(file_path=file_path,
                                      image_size=Params.image_size,
                                      preprocess_method=Params.preprocess_method)
    predict_model = rebuild_decoder(model_path=model_path)
    predict_text = predict(image_input=img_data,
                           predict_model=predict_model)

    sh.show_image(file_path=file_path,
                  predict_text=predict_text,
                  save_path=save_path,
                  show_org=show_org)

    en = time.time()
    print(predict_text)
    print(en - st)


# test the trained model using test-dataset.
test_model(test_info=(test_lst, test_texts, test_img),
           show_org=True,
           save_root_path='./test/test_50/',
           model_path='./models/gru_before/gru_before_epochs100.h5')

# just test one picture.
test_one_picture(file_path='./datas/img_resize_person/0.png',
                 save_path='./test_person/test_100/0.png',
                 show_org=False,
                 model_path='./models/gru_before/gru_before_epochs100.h5')

# test some personal pictures.
TEST_PATH = os.path.join(Params.root_data_path, 'img_resize_person/')
MODEL_ROOT_PATH = './models/gru_before/'
TEST_IMAGE_NUM = [i.split('.')[0] for i in os.listdir(TEST_PATH)]
TEST_EPOCHS_LST = [200, 230, 250, 280, 300, 320, 350]
for TEST_EPOCHS in TEST_EPOCHS_LST:
    SAVE_PTAH = './test_person/test_' + str(TEST_EPOCHS) + '/'
    MODEL_PATH = os.path.join(MODEL_ROOT_PATH, 'model_gru_before_epochs'
                              + str(TEST_EPOCHS) + '.h5')
    t1 = time.time()
    for IMAGE_NUM in TEST_IMAGE_NUM:
        IMAGE_PATH = os.path.join(TEST_PATH, str(IMAGE_NUM) + '.png')
        SAVE_IMAGE = os.path.join(SAVE_PTAH, str(IMAGE_NUM) + '.png')
        test_one_picture(file_path=IMAGE_PATH,
                         save_path=SAVE_IMAGE,
                         show_org=False,
                         model_path=MODEL_PATH)

    t2 = time.time()
    print(t2 - t1)
