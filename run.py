import tensorflow as tf
from dataset import token_container
import Setting
import tool

# 加载训练好的模型
print('===================Loading Model===================')
model = tf.keras.models.load_model(Setting.BEST_MODEL_PATH)
# 随机生成一首诗
print(tool.generate_random_poetry(token_container, model))
# # 给出部分信息的情况下，随机生成剩余部分
print(tool.generate_random_poetry(token_container, model, s='一江春水向东流'))
# 生成藏头诗
print(tool.generate_acrostic(token_container, model, head='好好学习天天向上'))