import tensorflow as tf
import matplotlib.pyplot as plt

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # 加载我们的模型。 加载已经在 imagenet 数据上预训练的 VGG
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layers(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model
