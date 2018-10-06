# -*- coding: utf-8 -*-
"""
    manners of data augment for image using keras:
    we choose to handle these methods outline the training process
    for the sake of limited GPU memory.
    1. random_shift            平移变换
    2. random_shear            空间剪切
    3. random_zoom             缩放变换
    4. random_brightness       亮度变换
    5. random_rotation         旋转变换
"""
import os
import codecs
from params import Params
from keras.preprocessing import image


# image augment
def image_augment(arr):
    """
    :param arr: array of a image, 3D. [array]
    :return:
    """
    # 旋转 微调
    x = image.random_rotation(arr, rg=5, row_axis=0, col_axis=1,
                              channel_axis=2, fill_mode='nearest')
    # 平移 上下平移 左右平移 微调
    x = image.random_shift(x, wrg=0.1, hrg=0.1, row_axis=0, col_axis=1,
                           channel_axis=2, fill_mode='nearest')
    # 随机空间剪切
    x = image.random_shear(x, intensity=10, row_axis=0, col_axis=1,
                           channel_axis=2, fill_mode='nearest')
    # 随机放缩
    x = image.random_zoom(x, zoom_range=(0.75, 1.25), row_axis=0, col_axis=1,
                          channel_axis=2, fill_mode='nearest')
    # 随机亮度调整
    x = image.random_brightness(x, brightness_range=(0.75, 1.25))

    return x


def image_augment_for_directory(img_data_path='./img_data/',
                                text_data_path='./caption_text/',
                                save_img_path='./augment_img_data/',
                                save_text_path='./augment_caption_text/',
                                image_size=(224, 224),
                                augment_count=5):
    """
    对一个文件夹中所有的图像进行随机数据增强，并对其匹配对应的文本数据
    将增强后的数据+原始数据单独存在另一个文件夹，不要和原来的数据混合
    :param img_data_path: 原始图片所在的文件夹路径 [str]
    :param text_data_path: 原始文本数据所在的路径 [str]
    :param save_img_path: 增强后的图片数据所在的路径 [str]
    :param save_text_path: 和增强后的图像数据匹配的文本数据存储路径 [str]
    :param image_size: 图片的像素 [tuple]
    :param augment_count: 每一张图片需要随机增强的倍数 [int]
    :return:
    """
    filename_lst = os.listdir(img_data_path)
    img_lst = [int(x[:-4]) for x in filename_lst]
    filename_lst = [os.path.join(img_data_path, x) for x in filename_lst]

    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)
        max_index = max(img_lst)
    else:
        saved_img = os.listdir(save_img_path)
        saved_img_num = [int(x[:-4]) for x in saved_img]
        max_index = max(saved_img_num)
    if not os.path.exists(save_text_path):
        os.mkdir(save_text_path)

    # 进行数据增强
    fail = []

    index = max_index + 1
    for img_path, img_num in zip(filename_lst, img_lst):
        try:
            img_path_to_aug = os.path.join(save_img_path, str(img_num) + '.png')
            text_path_to_aug = os.path.join(save_text_path, str(img_num) + '.txt')
            # 先判断当前图片是否已经处理过 如果是 则跳过
            if os.path.exists(img_path_to_aug) and\
                    os.path.exists(text_path_to_aug):
                print('{0} is done.'.format(img_num))
                continue

            temp_img = image.load_img(path=img_path, target_size=image_size)
            temp_img = image.img_to_array(temp_img)
            with codecs.open(os.path.join(text_data_path, str(img_num) + '.txt'),
                             'r', encoding='utf-8') as fr:
                temp_text = fr.read()

            # 先将原始数据存到save的文件夹
            image.save_img(path=img_path_to_aug, x=temp_img)
            with codecs.open(text_path_to_aug, 'w', encoding='utf-8') as ft:
                ft.writelines(temp_text)

            # augment
            for i in range(augment_count):
                # 给每一个新的图像匹配对应文本数据 并存入文件夹
                new_img_path = os.path.join(save_img_path, str(index) + '.png')
                new_text_path = os.path.join(save_text_path, str(index) + '.txt')
                new_img = image_augment(arr=temp_img)
                image.save_img(path=new_img_path, x=new_img)
                with codecs.open(new_text_path, 'w', encoding='utf-8') as fw:
                    fw.writelines(temp_text)

                index += 1

            print(img_num)
        except:
            fail.append(img_num)
            print('picture {0} is fail.'.format(img_num))
            continue

    return fail


if __name__ == '__main__':
    image_augment_for_directory(img_data_path=Params.img_data_path,
                                text_data_path=Params.text_data_path,
                                save_img_path=Params.augment_img_path,
                                save_text_path=Params.augment_text_path)













