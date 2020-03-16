import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False
# 要固定卷积核，因此vgg不可被训练

'''
for i in vgg.layers:
    # 枚举模型中的所有层
    print(i.name)
'''

# 创建一张256*256的纯灰色图片，之后再叠加一个服从截断正态分布的噪声，并设置为可训练
input_img = tf.Variable(
    tf.ones(
        [1, 256, 256, 3]
    ) * 0.5 + tf.random.truncated_normal(
        [1, 256, 256, 3],
        mean=0.0,
        stddev=0.08
    )
    , trainable=True
)

# 可以来看看最开始的图片
plt.imshow(input_img.numpy()[0, ...])
plt.show()



# 分别看一看 block2_conv1 的101号卷积核  block5_conv4 的85号卷积核
# output = vgg.get_layer("block2_conv1").output
output = vgg.get_layer("block5_conv4").output
print(output.shape)

# model = tf.keras.Model(vgg.input, output[..., 101:102])
model = tf.keras.Model(vgg.input, output[..., 84:85])


pic = model(input_img)
print(pic.shape)
plt.imshow(pic.numpy()[0, ..., 0], cmap='gray')
plt.show()

optimizer = tf.keras.optimizers.RMSprop()

@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        mod_out = model(input_img)
        loss = - tf.reduce_mean(mod_out)
    grad = tape.gradient(loss, input_img)
    optimizer.apply_gradients([(grad, input_img)])
    return grad


for i in trange(300):
    grad = train_step(input_img)
    # print(grad)

tf.clip_by_value(input_img, 0., 1.)

plt.imshow(input_img.numpy()[0, ...])
plt.show()
