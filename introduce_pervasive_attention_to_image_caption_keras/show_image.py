# -*- coding:utf-8 -*-
from PIL import Image, ImageDraw, ImageFont
import math


def make_text_image(width=None,
                    white=0,
                    text='',
                    save_path=None,
                    mode="rgb"):
    """
    生成一个文字图形
    :param width: 图片的宽度[int]
    :param white: 1/0, white=1表示白底黑字, 否则为黑底白字[int/boolean]
    :param text: 文本[str]
    :param save_path: 保存图片的路径[str]
    :param mode: 图片的模式,rgb表示彩色[str]
    :return:
    """

    # 字体可能要改
    # linux查看支持的汉字字体 # fc-list :lang=zh
    # 整个text的长度和宽度
    ft = ImageFont.truetype("STXIHEI.TTF", 15)
    w, h = ft.getsize(text)

    # 计算要几行
    lines = math.ceil(w / width) + 1
    height = h * lines

    # 一个汉字的宽度
    one_zh_width, h = ft.getsize("中")

    if len(mode) == 1:  # L, 1
        background = (255,)
        color = (0,)
    elif len(mode) == 3:  # RGB
        background = (255, 255, 255)
        color = (0, 0, 0)
    elif len(mode) == 4:  # RGBA, CMYK
        background = (255, 255, 255, 255)
        color = (0, 0, 0, 0)
    else:
        raise Exception('the len(mode) should 1/3/4.')

    # 生成一张包括文字的图片
    new_image = Image.new(mode, (width, height), background if white else color)
    draw = ImageDraw.Draw(new_image)

    # 分割行
    text = text + " "  # 处理最后少一个字问题
    text_list = []
    start = 0
    end = len(text) - 1
    while start < end:
        try_text = None
        choose_n = 0
        for n in range(end):
            try_text = text[start:start + n]
            w, h = ft.getsize(try_text)
            choose_n = n
            if w + 2 * one_zh_width > width:
                break
        text_list.append(try_text[0:-1])
        start = start + choose_n - 1

    # print(text_list)

    i = 0
    for t in text_list:
        draw.text((one_zh_width, i * h), t, color if white else background, font=ft)
        i = i + 1

    new_image.save(save_path)


def resize_canvas(org_image=None,
                  add_image=None,
                  new_image_path=None,
                  text=''):
    """
    把图片和文字整合到一张图片上
    :param org_image: 原始图片的路径[str]
    :param add_image: 被合并图片的路径[str]
    :param new_image_path: 保存新图片的路径[str]
    :param text: 需要展示在图片上的文本[str]
    :return:
    """
    org_im = Image.open(org_image)
    org_im = org_im.resize((1024, 512), Image.ANTIALIAS)
    org_width, org_height = org_im.size
    mode = org_im.mode

    make_text_image(org_width, 0, text, add_image, mode)

    add_im = Image.open(add_image)
    add_width, add_height = add_im.size
    mode = org_im.mode

    new_image = Image.new(mode, (org_width, org_height + add_height))
    new_image.paste(org_im, (0, 0, org_width, org_height))
    new_image.paste(add_im, (0, org_height, add_width, add_height + org_height))
    new_image.save(new_image_path)


def show_image(file_path=None,
               original_text=None,
               predict_text=None,
               show_org=True,
               save_path=None):
    """
    展示图片
    :param file_path: 加载单张图片完整的绝对路径[str]
    :param original_text: 原始caption的文本[str]
    :param predict_text: 生成的caption[str]
    :param show_org: 是否展示原始caption[boolean]
    :param save_path: 保存测试图片完整的绝对路径[str]
    :return:
    """

    if show_org:
        resize_canvas(file_path, 'tmp1.png', 'tmp1.png', original_text)
        resize_canvas('tmp1.png', 'tmp2.png', save_path, predict_text)
    else:
        resize_canvas(file_path, 'tmp1.png', save_path, predict_text)
