import generator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import model as md
from params import Params
import os
import codecs
import pickle
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint

# 配置gpu参数
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# filename
filename = 'emb{0}_batch{1}_layer{2}_dropout{3}_g{4}_reduc{5}'. \
    format(Params.embedding_dim,
           Params.batch_size,
           Params.dense_layers,
           str(Params.dropout),
           Params.growth_rate,
           str(Params.reduction))

# 不同超参训练的模型的temp_data存在不同的文件夹下
model_temp_data_path = os.path.join(Params.temp_data_path, filename)
if not os.path.exists(model_temp_data_path):
    os.mkdir(model_temp_data_path)

# 数据处理
input_words_count = \
    generator.word_count(is_count=Params.is_count,
                         input_data_path=Params.text_remove_path,
                         output_data_path=os.path.join(model_temp_data_path,
                                                       'input_words_count.pkl'))

sen_len_count = \
    generator.sentence_len(input_data_path=Params.text_remove_path,
                           output_data_path=os.path.join(model_temp_data_path,
                                                         'sen_len_count.pkl'))

# 生成input_token_index/reverse_input_word_index
input_token_index, reverse_input_word_index = \
    generator.get_token(input_words_count=input_words_count,
                        input_word_num=Params.dec_word_num)

# 生成训练train_lst
train_lst, valid_lst, test_lst = \
    generator.generate_filename_list(total_num=Params.img_total_num,
                                     train_prob=Params.img_train_prob,
                                     valid_num=Params.img_valid_num,
                                     data_path=Params.img_data_path)

# 训练集生成器
share_params = {
    'max_seq_length': Params.max_dec_len,
    'word_num': Params.dec_word_num,
    'return_one_hot': Params.return_one_hot,
    'image_size': (Params.image_size, Params.image_size),
    'preprocess_method': Params.preprocess_method,
    'img_data_path': Params.img_data_path,
    'text_data_path': Params.text_data_path
}

train_generator = generator.train_fit_generator(train_lst=train_lst,
                                                token_index=input_token_index,
                                                batch_size=Params.batch_size,
                                                **share_params)
# 验证集的数据
valid_data = generator.get_batch_data(batch_lst=valid_lst,
                                      token_index=input_token_index,
                                      **share_params)

# 保存数据以备测试
with codecs.open(os.path.join(model_temp_data_path, 'input_token_index.pkl'), 'wb') as fi:
    pickle.dump(input_token_index, fi)
with codecs.open(os.path.join(model_temp_data_path, 'reverse_input_word_index.pkl'), 'wb') as fir:
    pickle.dump(reverse_input_word_index, fir)
with codecs.open(os.path.join(model_temp_data_path, 'train_lst.pkl'), 'wb') as ftr:
    pickle.dump(train_lst, ftr)
with codecs.open(os.path.join(model_temp_data_path, 'valid_lst.pkl'), 'wb') as fva:
    pickle.dump(valid_lst, fva)
with codecs.open(os.path.join(model_temp_data_path, 'test_lst.pkl'), 'wb') as fte:
    pickle.dump(test_lst, fte)


# 训练函数
def train_model_fit_generator(load_weight_path=None,
                              save_model_path='./models/',
                              load_weights_by_name=True):
    """
    根据rnn_type, model_type的组合总共有8种BaseLineModels模型
    :param load_weight_path: 加载resnet权重的路径/在'continue train'
        中可以用于加载已经训练好的模型[str]
    :param save_model_path: 保存训练模型的路径[str]
    :param load_weights_by_name: 是否通过by_name的方式加载初始化权重[boolean]
    :return:
    """

    model = md.pervasive_attention(blocks=Params.blocks,
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
                                   clip=Params.clip
                                   )

    # 初始化模型权重, 进行fine-tuning
    model.load_weights(filepath=load_weight_path, by_name=load_weights_by_name)
    print('model weights is initial with densenet121.')
    # 保存log文件的路径
    model_base_name = save_model_path + '/epochs{epoch:02d}_val_loss_{val_loss:.2f}.h5'
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    save_log_dir = os.path.join(Params.log_path, filename)
    if not os.path.exists(save_log_dir):
        os.mkdir(save_log_dir)

    model.fit_generator(generator=train_generator,
                        validation_data=valid_data,
                        steps_per_epoch=Params.steps_per_epoch,
                        epochs=Params.epochs,
                        workers=Params.workers,
                        callbacks=
                        [TensorBoard(log_dir=save_log_dir),
                         ModelCheckpoint(
                             filepath=model_base_name,
                             monitor='val_loss',
                             verbose=1,
                             period=10,
                             mode='auto',
                             save_weights_only=True
                         )])


# train model
# gru_before
if __name__ == '__main__':
    weight_path = \
        './densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    train_model_fit_generator(load_weight_path=weight_path,
                              save_model_path=os.path.join('./models/', filename),
                              load_weights_by_name=True)
