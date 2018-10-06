# -*- coding: utf-8 -*-
from keras.layers import Input, Embedding, \
    Lambda, Concatenate, BatchNormalization, \
    Conv2D, Dropout, Dense, MaxPooling2D, ZeroPadding2D, \
    AveragePooling2D, ZeroPadding1D
from keras.layers import Activation, TimeDistributed, Conv1D
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K
import finetune_densnet as fd


# 编写Lamda层
def cnn_reshape_func(cnn_embedding, repeat):
    """
    这里只采用DenseNet121/DenseNet169/DenseNet201的dense block3 + transition + no-pool
    输出的tensor来抽取图像的特征, 并reshape为shape=(?, 14*14, channels), 把14*14=196的
    image feature看成是翻译任务中的source sentence的长度s:
        reference:arxiv.org/pdf/1808.03867.pdf

    :param cnn_embedding: image 经过 densenet embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    input_shape = cnn_embedding.shape
    cnn_embedding = K.reshape(cnn_embedding, [-1, 1, input_shape[-1]])
    cnn_embedding = K.tile(cnn_embedding, [1, repeat, 1])
    cnn_embedding = K.reshape(cnn_embedding, [-1, input_shape[1], repeat, input_shape[-1]])

    return cnn_embedding


def dec_reshape_func(dec_embedding, repeat):
    """
    对embedding之后的target sentence的tensor转换成pervasive-attention model需要的shape
    arxiv.org/pdf/1808.03867.pdf
    :param dec_embedding: target sentence embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, source sentence s的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    input_shape = dec_embedding.shape
    dec_embedding = K.reshape(dec_embedding, [-1, 1, input_shape[-1]])
    dec_embedding = K.tile(dec_embedding, [1, repeat, 1])
    dec_embedding = K.reshape(dec_embedding, [-1, input_shape[1], repeat, input_shape[-1]])
    dec_embedding = K.permute_dimensions(dec_embedding, [0, 2, 1, 3])

    return dec_embedding


def cnn_embedding_layer(cnn_embedding, repeat):
    """
    转换成Lambda层
    :param cnn_embedding: image 经过 densenet201 embedding之后的结果[tensor]
    :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    return Lambda(cnn_reshape_func,
                  arguments={'repeat': repeat})(cnn_embedding)


def dec_embedding_layer(dec_embedding, repeat):
    """
    转换层Lambda层
    :param dec_embedding: target sentence embedding之后的结果[tensor]
     :param repeat: 需要重复的次数, target sentence t的长度[int]
    :return: 2D tensor (?, s, t, embedding_dim)
    """
    return Lambda(dec_reshape_func,
                  arguments={'repeat': repeat})(dec_embedding)


# avg pooling
def h_avg_pooling_layer(h):
    """
    实现论文中提到的均值池化 arxiv.org/pdf/1808.03867.pdf
    :param h: 由densenet结构输出的shape为(?, s, t, fl)的tensor[tensor]
    :return: (?, t, fl)
    """
    h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(h)
    h = AveragePooling2D(data_format='channels_first',
                         pool_size=(h.shape[2], 1))(h)
    h = Lambda(lambda x: K.squeeze(x, axis=2))(h)

    return h


# max pooling
def h_max_pooling_layer(h):
    """
    实现论文中提到的最大池化 arxiv.org/pdf/1808.03867.pdf
    :param h: 由densenet结构输出的shape为(?, s, t, fl)的tensor[tensor]
    :return: (?, t, fl)
    """
    h = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1, 3]))(h)
    h = MaxPooling2D(data_format='channels_first',
                     pool_size=(h.shape[2], 1))(h)
    h = Lambda(lambda x: K.squeeze(x, axis=2))(h)

    return h


# transition layer
def pervasive_transition_block(x,
                               reduction,
                               pooling=True):
    """A transition block.
    该transition block与densenet的标准操作不一样，此处不包括pooling层
    pervasive-attention model中的transition layer需要保持输入tensor
    的shape不变 arxiv.org/pdf/1808.03867.pdf
    # Arguments
        x: input tensor.
        reduction: float, the rate of feature maps need to retain.
        pooling: boolean, if the transition block choose to pooling
                the size of feature map, default True.

    # Returns
        output tensor for the block.
    """
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(K.int_shape(x)[3] * reduction), 1, use_bias=False)(x)
    if pooling:
        x = MaxPooling2D((2, 1), strides=(2, 1))(x)

    return x


# building block
def pervasive_conv_block(x,
                         growth_rate,
                         dropout):
    """A building block for a dense block.
    该conv block与densenet的标准操作不一样，此处通过
    增加Zeropadding2D层实现论文中的mask操作，并将
    Conv2D的kernel size设置为(3, 2)
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        dropout: float, dropout rate at dense layers.
    Remind that, in paper, a dropout tech is using as a regular manner,
    which make a difference from the standard conv block in DenseNet.
    # Returns
        Output tensor for the block.
    """
    x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False)(x1)
    x1 = BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
    x1 = Activation('relu')(x1)
    x1 = ZeroPadding2D(padding=((1, 1), (1, 0)))(x1)  # mask sake
    x1 = Conv2D(growth_rate, (3, 2), padding='valid')(x1)
    x1 = Dropout(rate=dropout)(x1)

    x = Concatenate(axis=3)([x, x1])

    return x


# dense block
def pervasive_dense_block(x,
                          blocks,
                          growth_rate,
                          dropout):
    """A dense block.
    Each dense block with contain one or more conv blocks/building blocks.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        growth_rate:float, growth rate at dense layers.
        dropout: float, dropout rate at dense layers.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = pervasive_conv_block(x, growth_rate=growth_rate, dropout=dropout)

    return x


# pervasive-attention model
def pervasive_attention(blocks,
                        trainable=False,
                        image_size=224,
                        growth_rate=12,
                        reduction=0.5,
                        dropout=0.5,
                        max_dec_len=140,
                        embedding_dim=128,
                        dec_word_num=7000,
                        learning_rate=0.001,
                        decay=0.0,
                        clip=5.):
    """
    build a pervasive-attention model with a densenet-like cnn structure.

    :param blocks: a list variable length, indicates different number of
        building blocks in dense blocks, e.g which [6, 12, 48, 32]
        for DenseNet201 and [6, 12, 32, 32] for DenseNet169. [list]
    :param trainable: default False, if True, train the model from imagenet. [boolean]
    :param image_size: default 224. [int]
    :param growth_rate: float, growth rate at dense layers. [int]
    :param reduction: float, the rate of feature maps which
        need to retain after transition layer. [float]
    :param dropout: dropout rate used in each conv block, default 0.2. [float]
    :param max_dec_len: the max len of target sentences. [int]
    :param embedding_dim: the hidden units of first two embedding layers. [int]
    :param dec_word_num: the vocabulary size of target sentences. [int]
    :param learning_rate: learning rate. [float]
    :param clip: clip the gradient. [float]
    :param decay: decay rate. [float]
    :return:
    """
    # Inputs
    img_input = Input(shape=(image_size, image_size, 3), name='img_input')
    dec_input = Input(shape=(max_dec_len,), name='dec_input')

    # embedding
    cnn_embedding = fd.finetune_densenet121(image_input=img_input,
                                            trainable=trainable)
    dec_embedding = Embedding(dec_word_num + 2,
                              embedding_dim,
                              name='dec_embedding')(dec_input)
    # implement a convEmbedding
    cnn_embedding = Conv1D(int(K.int_shape(cnn_embedding)[-1] * 0.5),
                           kernel_size=3,
                           padding='same',
                           data_format='channels_last')(cnn_embedding)
    cnn_embedding = Conv1D(int(K.int_shape(cnn_embedding)[-1]), 3,
                           padding='same',
                           data_format='channels_last')(cnn_embedding)
    cnn_embedding = Conv1D(int(K.int_shape(cnn_embedding)[-1]), 3,
                           padding='same',
                           data_format='channels_last')(cnn_embedding)
    dec_embedding = ZeroPadding1D(padding=(2, 0))(dec_embedding)
    dec_embedding = Conv1D(embedding_dim, 3, padding='valid',
                           data_format='channels_last')(dec_embedding)
    dec_embedding = ZeroPadding1D(padding=(2, 0))(dec_embedding)
    dec_embedding = Conv1D(embedding_dim, 3, padding='valid',
                           data_format='channels_last')(dec_embedding)
    dec_embedding = ZeroPadding1D(padding=(2, 0))(dec_embedding)
    dec_embedding = Conv1D(embedding_dim, 3, padding='valid',
                           data_format='channels_last')(dec_embedding)

    # concatenate
    cnn_embedding = cnn_embedding_layer(cnn_embedding, repeat=max_dec_len)
    dec_embedding = dec_embedding_layer(dec_embedding,
                                        repeat=int(K.int_shape(cnn_embedding)[-3]))
    cnn_dec_embedding = Concatenate(axis=3)([cnn_embedding, dec_embedding])

    # densenet conv1 1x1
    x = Conv2D(int(K.int_shape(cnn_dec_embedding)[-1] * 0.5),
               kernel_size=1, strides=1)(cnn_dec_embedding)
    x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)

    # densenet dense block
    for i in range(len(blocks)):
        x = pervasive_dense_block(x,
                                  blocks=blocks[i],
                                  growth_rate=growth_rate,
                                  dropout=dropout)
        if i < len(blocks) - 1:
            x = pervasive_transition_block(x, reduction, pooling=True)
        else:
            x = pervasive_transition_block(x, reduction, pooling=False)

    x = BatchNormalization(axis=3, epsilon=1.001e-5)(x)

    # h pooling
    h = h_max_pooling_layer(x)

    # Target sequence prediction
    output = TimeDistributed(Dense(dec_word_num + 2, activation='softmax'))(h)

    # compile
    model = Model([img_input, dec_input], [output])
    opt = Adam(lr=learning_rate, decay=decay, clipvalue=clip, amsgrad=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy')

    return model


# e.g 89
def pervasive_attention89(image_size=224,
                          growth_rate=12,
                          reduction=0.5,
                          dropout=0.5,
                          max_dec_len=140,
                          embedding_dim=256,
                          dec_word_num=7000):
    return pervasive_attention(blocks=[6, 12, 24],
                               image_size=image_size,
                               growth_rate=growth_rate,
                               reduction=reduction,
                               dropout=dropout,
                               max_dec_len=max_dec_len,
                               embedding_dim=embedding_dim,
                               dec_word_num=dec_word_num)


# e.g 105
def pervasive_attention105(image_size=224,
                           growth_rate=12,
                           reduction=0.5,
                           dropout=0.5,
                           max_dec_len=140,
                           embedding_dim=256,
                           dec_word_num=7000):
    return pervasive_attention(blocks=[6, 12, 32],
                               image_size=image_size,
                               growth_rate=growth_rate,
                               reduction=reduction,
                               dropout=dropout,
                               max_dec_len=max_dec_len,
                               embedding_dim=embedding_dim,
                               dec_word_num=dec_word_num)


# e.g  debug only
# model1 = pervasive_attention(blocks=[2, 2, 2, 4, 4, 4],
#                              trainable=False,
#                              image_size=224,
#                              growth_rate=32,
#                              reduction=0.5,
#                              dropout=0.5,
#                              max_dec_len=140,
#                              embedding_dim=256,
#                              dec_word_num=7000,
#                              learning_rate=0.001,
#                              decay=0.0,
#                              clip=5.)
# model1.summary()
