from collections import Counter
import math
import numpy as np
import tensorflow as tf
import Setting


class Token_container:
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.token_dict_re = {value: key for key, value in self.token_dict.items()}
        self.verb_size = len(self.token_dict)

    def id_to_token(self, token_id):
        # 由id找到token
        return self.token_dict_re[token_id]

    def token_to_id(self, token):
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def encode(self, _tokens):
        token_ids = [self.token_to_id('[CLS]')]
        for token in _tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.token_to_id('[SEP]'))
        return token_ids

    def decode(self, _token_ids):
        spec = {'[CLS]', '[SEP]'}
        result = []
        for token_id in _token_ids:
            if self.id_to_token(token_id) in spec:
                continue
            result.append(self.id_to_token(token_id))
        return ''.join(result)


# 禁用词
disallowed_words = Setting.DISALLOWED_WORDS
# 句子最大长度
max_len = Setting.MAX_LEN
# 最小词频
min_word_frequency = Setting.MIN_WORD_FREQUENCY
# mini batch 大小
batch_size = Setting.BATCH_SIZE

with open(Setting.DATASET_PATH, mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.replace('：', ':') for line in lines]  # 将所有冒号变为英文冒号

poetry = []  # 诗词列表
for line in lines:
    if line.count(':') != 1:
        continue
    __, last_half = line.split(':')  # 只保留诗词，不要题目
    b_flag = False
    for dis_word in disallowed_words:
        if dis_word in last_half:
            b_flag = True
            break
    if b_flag:
        continue
    if len(line) > max_len:
        continue
    poetry.append(last_half.replace('\n', ''))

counter = Counter()
for line in poetry:
    counter.update(line)

# 过滤低频词
tokens = [(token, count) for token, count in counter.items() if count >= min_word_frequency]
tokens = sorted(tokens, key=lambda x: -x[1])
tokens = [token for token, count in tokens]
tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + tokens
# 创建词典映射
token_id_dict = dict(zip(tokens, range(len(tokens))))
token_container = Token_container(token_id_dict)
np.random.shuffle(poetry)


class PoetryDataSetGenerator:
    def __init__(self, data, random=False):
        self.data = data
        self.batch_size = batch_size
        self.step = int(math.floor(len(self.data) / self.batch_size))
        self.random = random

    def padding(self, data, length=None, padding=None):
        # 计算填充长度
        if length is None:
            length = max(map(len, data))
        if padding is None:
            padding = token_container.token_to_id('[PAD]')
        result = []
        for line in data:
            padding_length = length - len(line)
            if padding_length > 0:
                result.append(np.concatenate([line, [padding] * padding_length]))
            else:
                result.append(line[:length])
        return np.array(result)

    def __len__(self):
        return self.step

    def __iter__(self):
        total = len(self.data)
        if self.random:
            np.random.shuffle(self.data)
        for start in range(0, len(self.data), batch_size):
            end = min(start + self.batch_size, total)
            batch_data = []
            # 对古诗进行编码
            for line in self.data[start:end]:
                batch_data.append(token_container.encode(line))
            # 填充到同一长度
            batch_data = self.padding(batch_data)
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], token_container.verb_size)
            del batch_data

    def for_fit(self):
        while True:
            yield from self.__iter__()
