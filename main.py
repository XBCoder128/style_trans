import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import trange
import numpy as np
import time
import func
import model
from config import *

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

content_path = 'style17.jpeg'
style_path = 'style20.jpg'

content_image = func.load_img(content_path)
style_image = func.load_img(style_path)

content_layers = ['block4_conv2']
style_layers = ['block2_conv1',
                'block2_conv2',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1',]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

extractor = model.StyleContentModel(style_layers, content_layers)

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

def style_content_loss(outputs):
    style_outputs = outputs['style'] # 用来表示style信息的网络层的输出，这里已经计算过Gram矩阵了
    content_outputs = outputs['content'] # 用来表示content信息的网络层的输出，内容信息不需要计算Gram
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])

    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
image = tf.Variable(content_image + tf.random.truncated_normal(content_image.shape, mean=0.0, stddev=0.08))

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs)
        loss += total_variation_weight * func.total_variation_loss(image)

    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(func.clip_0_1(image))

for n in trange (epochs * steps_per_epoch):
    train_step(image)

plt.imshow(image.read_value()[0])
plt.show()
print (image.read_value()[0].shape)
Eimg = tf.image.convert_image_dtype (image.read_value()[0], tf.uint8)
Eimg = tf.image.encode_jpeg (Eimg)
tf.io.write_file ('F13.jpg', Eimg)