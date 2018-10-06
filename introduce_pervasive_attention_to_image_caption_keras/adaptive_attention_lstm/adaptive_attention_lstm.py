"""
    Reference: https://arxiv.org/abs/1612.01887.pdf
"""

from keras import layers
from keras.layers import activations, initializers, regularizers, constraints
import keras.backend as K


class ShapeError(Exception):
    pass


class AdaptiveAttentionLSTMCell(layers.Layer):
    # AdaptiveAttentionLSTMCell 编写Cell单元 词表大小(dense)不写在里面
    def __init__(self,
                 units,
                 cnn_encoder_k=None,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        :param units: decoder hidden_size, d in WTL 在WTL中lstm的hidden_size和Vi的维度d是相等
        :param cnn_encoder_k: cnn_encoder中导出来的特征向量个数k
        :param kernel_initializer:
        :param recurrent_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param kwargs:
        """
        self.units = units
        self.state_size = (units, units)

        self.cnn_encoder_k = cnn_encoder_k

        if self.cnn_encoder_k is None:
            raise ValueError('''self.cnn_encoder_k can not be NoneType''')

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_contraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AdaptiveAttentionLSTMCell, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.input_dim = input_shape[0]
        self.constant_shape = input_shape[1]
        """
            这里假定传入的constant(V=[v1,v2,...,vk])的shape必须为[None, k, d] 
            Check if k=self.cnn_encoder_k and d=self.units=cnn_encoder_d
        """
        if self.constant_shape[-2] != self.cnn_encoder_k or \
                self.constant_shape[-1] != self.units:
            raise ShapeError('''The shape of constant(V=[v1,v2,...,vk]) must be (None, {0}, {1})},
            but got (None, {2}, {3})'''.format(self.cnn_encoder_k, self.units,
                                               self.constant_shape[-2], self.constant_shape[-1]))

        """
                f-gate (forget gate)
        """
        self.W_f = self.add_weight(shape=(self.units, self.units),
                                   name='W_f',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        # U_f 表示将decoder的input和Vg(global visual context)并在一起输入 kernel_weight
        # 但这个拼接是在AdaptiveAttentionLSTNCell外部进行的，其总体的shape会被识别出来为self.input_dim
        # 即满足 self.input_dim = embed_size(or vocab_size) + cnn_encoder_d(self.units)
        self.U_f = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_f',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.b_f = self.add_weight(shape=(self.units,),
                                   name='b_f',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        """
                i-gate (input gate)
        """
        self.W_i = self.add_weight(shape=(self.units, self.units),
                                   name='W_i',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.U_i = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_i',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.b_i = self.add_weight(shape=(self.units,),
                                   name='b_i',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        """
                o-gate (output gate)
        """
        self.W_o = self.add_weight(shape=(self.units, self.units),
                                   name='W_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.U_o = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.b_o = self.add_weight(shape=(self.units,),
                                   name='b_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        """
                g-gate (sentinel gate)
        """
        self.W_g = self.add_weight(shape=(self.units, self.units),
                                   name='W_g',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.U_g = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_g',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.b_g = self.add_weight(shape=(self.units,),
                                   name='b_g',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        """
                a-renew input
        """
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )

        """
                c-visual context
        """
        # 由于在计算权重时会将st并进去一起算，因此会多出1维
        self.W_h = self.add_weight(shape=(self.cnn_encoder_k + 1, 1),
                                   name='W_h',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint)
        #  # W_a的第一个维度是decoder隐藏层的维度 第二个维度是任意的attention_alpha
        # 中的dense维度，只需要满足与self.V_a等长
        self.W_z = self.add_weight(shape=(self.units, self.cnn_encoder_k + 1),
                                   name='W_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint)
        # 这里采单向GRU 并认为encoder的hidden_size=self.units即与decoder的隐藏层维数相同
        # 因此在使用AttentionRNNCell时要注意将self.encoder_latent_dim设置与self.units一样
        self.U_z = self.add_weight(shape=(self.units, self.cnn_encoder_k + 1),
                                   name='U_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_contraint
                                   )

        self.built = True

    def call(self, inputs, states, constants):
        '''
        call函数 会在RNN中被调用然后被RNN改写 此时constant参数可用
        :param inputs: [wt; v_g] 维度为self.input_dim
        :param states: 前一步ht,mt
        :param constants: cnn_encoder outputs
        :return:
        '''
        h_tm = states[0]  # last hidden state
        m_tm = states[1]  # last memory cell
        self.v_seq = constants[0]  # [self.cnn_encoder_k, self.units] self.units=cnn_encoder_d
        """
            f-gate
        """
        ft = activations.sigmoid(
            K.dot(h_tm, self.W_f)
            + K.dot(inputs, self.U_f)
            + self.b_f
        )
        """
            i-gate
        """
        it = activations.sigmoid(
            K.dot(h_tm, self.W_i)
            + K.dot(inputs, self.U_i)
            + self.b_i
        )
        """
            o-gate
        """
        ot = activations.sigmoid(
            K.dot(h_tm, self.W_o)
            + K.dot(inputs, self.U_o)
            + self.b_o
        )
        """
            g-gate (sentinel gate)
        """
        gt = activations.sigmoid(
            K.dot(h_tm, self.W_g)
            + K.dot(inputs, self.U_g)
            + self.b_g
        )
        """
            at-renew input
        """
        at = activations.tanh(
            K.dot(h_tm, self.W_a)
            + K.dot(inputs, self.U_a)
            + self.b_a
        )
        """
            mt-memory cell
        """
        mt = m_tm * ft + it * at
        """
            ht-hidden state
        """
        ht = ot * activations.tanh(mt)
        """
            st-visual sentinel
        """
        st = gt * activations.tanh(mt)
        """
            ct-visual context
        """
        st = K.expand_dims(st, axis=1)
        # 将st合并进来一起计算权重参数[?, k+1, d] d=self.units 与论文的处理稍有不同
        self.v_expand = K.concatenate([self.v_seq, st], axis=1)
        # one_matrix = K.ones((self.cnn_encoder_k + 1, 1))
        vtt = K.dot(self.v_expand, self.W_z)
        dtt = K.repeat(K.dot(ht, self.U_z), self.cnn_encoder_k + 1)  # (?, k + 1, k + 1)
        tantt = K.tanh(vtt + dtt)

        zt = K.dot(tantt, self.W_h)

        alpha_t = activations.softmax(zt)  # (?, k + 1, 1)
        # alpha_t = K.expand_dims(alpha_t)  # (?, k + 1, 1)
        # 将st,v1,...,vk包括在内直接加权求和 与论文的处理稍有不同 (?, k + 1, units)
        # 输出(?, units)
        ct = K.squeeze(K.batch_dot(alpha_t, self.v_expand, axes=1), axis=1)  # batch_dot 针对 k + 1
        ht_plus_ct = ht + ct

        return ht_plus_ct, [ht, mt]
