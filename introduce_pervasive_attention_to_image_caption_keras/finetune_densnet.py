# -*- coding: utf-8 -*-
from keras.layers import Concatenate, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Reshape
import keras.backend as K


# densenet
def transition_block(x,
                     reduction,
                     name,
                     trainable=False,
                     pooling=True):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
        trainable: boolean, default False.
        pooling: boolean, default True for standard densenet.

    # Returns
        output tensor for the block.
    """
    x = BatchNormalization(axis=3, epsilon=1.001e-5,
                           name=name + '_bn',
                           trainable=trainable)(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[3] * reduction), 1,
               use_bias=False,
               name=name + '_conv',
               trainable=trainable)(x)
    if pooling:
        x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)

    return x


def conv_block(x,
               growth_rate,
               name,
               trainable=False):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
        trainable: boolean, default False.

    # Returns
        Output tensor for the block.
    """
    x1 = BatchNormalization(axis=3,
                            epsilon=1.001e-5,
                            name=name + '_0_bn',
                            trainable=trainable)(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1,
                use_bias=False,
                name=name + '_1_conv',
                trainable=trainable)(x1)
    x1 = BatchNormalization(axis=3, epsilon=1.001e-5,
                            name=name + '_1_bn',
                            trainable=trainable)(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(growth_rate, 3,
                padding='same',
                use_bias=False,
                name=name + '_2_conv',
                trainable=trainable)(x1)
    x = Concatenate(axis=3, name=name + '_concat')([x, x1])
    return x


def dense_block(x,
                blocks,
                name,
                trainable=False):
    """A dense block.

        # Arguments
            x: input tensor.
            blocks: integer, the number of building blocks.
            name: string, block label.
            trainable: boolean, default False.

        # Returns
            output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, trainable=trainable,
                       name=name + '_block' + str(i + 1))
    return x


def finetune_densenet(image_input,
                      blocks,
                      trainable=False):
    """
    这里只采用DenseNet121/DenseNet169/DenseNet201的dense block3 + transition + no-pool
    输出的tensor来抽取图像的特征, 并reshape为shape=(?, 14*14, channels), 把14*14=196的
    image feature看成是翻译任务中的source sentence的长度s:
        reference:arxiv.org/pdf/1808.03867.pdf

    :param image_input: 输入的tensor, e.g (?, 224, 224, 3) [tensor]
    :param blocks: 长度为3的list, 表示densenet不同dense block中的building block 个数
        e.g [6, 12, 48] for DenseNet201的前3个dense block;[6, 12, 32] for
        DenseNet169 的前3个dense block; [6, 12, 24] DenseNet169 的前3个dense block [list]
    :param trainable: 表示densenet模型是否可以训练, default False [boolean]
    :return: (?, 14*14, channels)
    """
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(image_input)
    x = Conv2D(64, 7, strides=2, use_bias=False,
               name='conv1/conv', trainable=trainable)(x)
    x = BatchNormalization(axis=3, epsilon=1.001e-5,
                           name='conv1/bn', trainable=trainable)(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks=blocks[0], name='conv2', trainable=trainable)
    x = transition_block(x, 0.5, name='pool2', trainable=trainable, pooling=True)
    x = dense_block(x, blocks=blocks[1], name='conv3', trainable=trainable)
    x = transition_block(x, 0.5, name='pool3', trainable=trainable, pooling=True)
    x = dense_block(x, blocks=blocks[2], name='conv4', trainable=trainable)
    x = transition_block(x, 0.5, name='pool4', trainable=trainable, pooling=False)

    # reshape
    x = Reshape((int(K.int_shape(x)[1]) * int(K.int_shape(x)[2]),
                 int(K.int_shape(x)[3])))(x)

    return x


def finetune_densenet121(image_input,
                         trainable=False):
    """

    :param image_input: 输入的tensor, e.g (?, 224, 224, 3) [tensor]
    :param trainable: 表示densenet模型是否可以训练, default False [boolean]
    :return:
    """
    return finetune_densenet(image_input=image_input,
                             blocks=[6, 12, 24],
                             trainable=trainable)


def finetune_densenet169(image_input,
                         trainable=False):
    """

    :param image_input: 输入的tensor, e.g (?, 224, 224, 3) [tensor]
    :param trainable: 表示densenet模型是否可以训练, default False [boolean]
    :return:
    """
    return finetune_densenet(image_input=image_input,
                             blocks=[6, 12, 32],
                             trainable=trainable)


def finetune_densenet201(image_input,
                         trainable=False):
    """

    :param image_input: 输入的tensor, e.g (?, 224, 224, 3) [tensor]
    :param trainable: 表示densenet模型是否可以训练, default False [boolean]
    :return:
    """
    return finetune_densenet(image_input=image_input,
                             blocks=[6, 12, 48],
                             trainable=trainable)



