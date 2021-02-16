import tensorflow as tf
from custom_layers import ReflectionPadding2D, ResizeLayer, SqueezeExciteBlock


class DSR_Base(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DSR_Base, self).__init__(**kwargs)

        # Encoder
        self.conv1_1   = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv1_bn1 = tf.keras.layers.BatchNormalization()
        self.conv1_2   = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv1_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool1  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2_1   = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv2_bn1 = tf.keras.layers.BatchNormalization()
        self.conv2_2   = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv2_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool2  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv3_1   = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv3_bn1 = tf.keras.layers.BatchNormalization()
        self.conv3_2   = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv3_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool3  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv4_1   = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv4_bn1 = tf.keras.layers.BatchNormalization()
        self.conv4_2   = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.conv4_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool4  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        # Bottleneck
        self.diconv1     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=1, padding='same', activation='relu', kernel_initializer='he_normal')
        self.diconv1_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv2     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=2, padding='same', activation='relu', kernel_initializer='he_normal')
        self.diconv2_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv3     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=4, padding='same', activation='relu', kernel_initializer='he_normal')
        self.diconv3_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv4     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=8, padding='same', activation='relu', kernel_initializer='he_normal')
        self.diconv4_bn1 = tf.keras.layers.BatchNormalization()

        self.diconcat    = tf.keras.layers.Concatenate(axis=-1)

        # Decoder
        self.ups1        = tf.keras.layers.UpSampling2D((2,2))
        self.deconv1_1   = tf.keras.layers.Conv2D(512, (2,2), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv1_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv1_se1 = SqueezeExciteBlock()
        self.concat1     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv1_2   = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv1_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv1_3   = tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv1_bn3 = tf.keras.layers.BatchNormalization()

        self.ups2        = tf.keras.layers.UpSampling2D((2,2))
        self.deconv2_1   = tf.keras.layers.Conv2D(256, (2,2), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv2_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv2_se1 = SqueezeExciteBlock()
        self.concat2     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv2_2   = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv2_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv2_3   = tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv2_bn3 = tf.keras.layers.BatchNormalization()

        self.ups3        = tf.keras.layers.UpSampling2D((2,2))
        self.deconv3_1   = tf.keras.layers.Conv2D(128, (2,2), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv3_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv3_se1 = SqueezeExciteBlock()
        self.concat3     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv3_2   = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv3_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv3_3   = tf.keras.layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv3_bn3 = tf.keras.layers.BatchNormalization()

        self.ups4        = tf.keras.layers.UpSampling2D((2,2))
        self.deconv4_1   = tf.keras.layers.Conv2D(64, (2,2), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv4_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv4_se1 = SqueezeExciteBlock()
        self.concat4     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv4_2   = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv4_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv4_3   = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')
        self.deconv4_bn3 = tf.keras.layers.BatchNormalization()

        self.outputs     = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        # Encoder
        c1_1   = self.conv1_1(inputs)
        c1_bn1 = self.conv1_bn1(c1_1, training=training)
        c1_2   = self.conv1_2(c1_bn1)
        c1_bn2 = self.conv1_bn2(c1_2, training=training)
        mp1    = self.maxpool1(c1_bn2)

        c2_1   = self.conv2_1(mp1)
        c2_bn1 = self.conv2_bn1(c2_1, training=training)
        c2_2   = self.conv2_2(c2_bn1)
        c2_bn2 = self.conv2_bn2(c2_2, training=training)
        mp2    = self.maxpool2(c2_bn2)

        c3_1   = self.conv3_1(mp2)
        c3_bn1 = self.conv3_bn1(c3_1, training=training)
        c3_2   = self.conv3_2(c3_bn1)
        c3_bn2 = self.conv3_bn2(c3_2, training=training)
        mp3    = self.maxpool3(c3_bn2)

        c4_1   = self.conv4_1(mp3)
        c4_bn1 = self.conv4_bn1(c4_1, training=training)
        c4_2   = self.conv4_2(c4_bn1)
        c4_bn2 = self.conv4_bn2(c4_2, training=training)
        mp4    = self.maxpool4(c4_bn2)

        # BottleNeck
        bt1 = self.diconv1(mp4)
        bt1 = self.diconv1_bn1(bt1)
        bt2 = self.diconv2(mp4)
        bt2 = self.diconv2_bn1(bt2)
        bt3 = self.diconv3(mp4)
        bt3 = self.diconv3_bn1(bt3)
        bt4 = self.diconv4(mp4)
        bt4 = self.diconv4_bn1(bt4)
        btc = self.diconcat([bt1,bt2,bt3,bt4])

        # Decoder
        x = self.ups1(btc)
        x = self.deconv1_1(x)
        x = self.deconv1_bn1(x, training=training)
        x = self.deconv1_se1(x)
        x = self.concat1([x,c4_1])
        x = self.deconv1_2(x)
        x = self.deconv1_bn2(x, training=training)
        x = self.deconv1_3(x)
        x = self.deconv1_bn3(x, training=training)

        x = self.ups2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_bn1(x, training=training)
        x = self.deconv2_se1(x)
        x = self.concat2([x,c3_1])
        x = self.deconv2_2(x)
        x = self.deconv2_bn2(x, training=training)
        x = self.deconv2_3(x)
        x = self.deconv2_bn3(x, training=training)

        x = self.ups3(x)
        x = self.deconv3_1(x)
        x = self.deconv3_bn1(x, training=training)
        x = self.deconv3_se1(x)
        x = self.concat3([x,c2_1])
        x = self.deconv3_2(x)
        x = self.deconv3_bn2(x, training=training)
        x = self.deconv3_3(x)
        x = self.deconv3_bn3(x, training=training)

        x = self.ups4(x)
        x = self.deconv4_1(x)
        x = self.deconv4_bn1(x, training=training)
        x = self.deconv4_se1(x)
        x = self.concat4([x,c1_1])
        x = self.deconv4_2(x)
        x = self.deconv4_bn2(x, training=training)
        x = self.deconv4_3(x)
        x = self.deconv4_bn3(x, training=training)

        outputs = self.outputs(x)
        return outputs

class DSR_Reflect(tf.keras.Model):
    def __init__(self, **kwargs):
        super(DSR_Reflect, self).__init__(**kwargs)

        # Encoder
        self.conv1_1   = tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv1_pd1 = ReflectionPadding2D()
        self.conv1_bn1 = tf.keras.layers.BatchNormalization()
        self.conv1_2   = tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv1_pd2 = ReflectionPadding2D()
        self.conv1_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool1  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2_1   = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv2_pd1 = ReflectionPadding2D()
        self.conv2_bn1 = tf.keras.layers.BatchNormalization()
        self.conv2_2   = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv2_pd2 = ReflectionPadding2D()
        self.conv2_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool2  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv3_1   = tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv3_pd1 = ReflectionPadding2D()
        self.conv3_bn1 = tf.keras.layers.BatchNormalization()
        self.conv3_2   = tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv3_pd2 = ReflectionPadding2D()
        self.conv3_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool3  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv4_1   = tf.keras.layers.Conv2D(512, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv4_pd1 = ReflectionPadding2D()
        self.conv4_bn1 = tf.keras.layers.BatchNormalization()
        self.conv4_2   = tf.keras.layers.Conv2D(512, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.conv4_pd2 = ReflectionPadding2D()
        self.conv4_bn2 = tf.keras.layers.BatchNormalization()
        self.maxpool4  = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        # Bottleneck
        self.diconv1     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=1, padding='valid', activation='relu', kernel_initializer='he_normal')
        self.diconv1_pd1 = ReflectionPadding2D()
        self.diconv1_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv2     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=2, padding='valid', activation='relu', kernel_initializer='he_normal')
        self.diconv2_pd1 = ReflectionPadding2D((2,2))
        self.diconv2_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv3     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=4, padding='valid', activation='relu', kernel_initializer='he_normal')
        self.diconv3_pd1 = ReflectionPadding2D((4,4))
        self.diconv3_bn1 = tf.keras.layers.BatchNormalization()

        self.diconv4     = tf.keras.layers.Conv2D(256, (3,3), dilation_rate=8, padding='valid', activation='relu', kernel_initializer='he_normal')
        self.diconv4_pd1 = ReflectionPadding2D((8,8))
        self.diconv4_bn1 = tf.keras.layers.BatchNormalization()

        self.diconcat    = tf.keras.layers.Concatenate(axis=-1)

        # Decoder
        self.ups1        = ResizeLayer()
        self.deconv1_1   = tf.keras.layers.Conv2D(512, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv1_pd1 = ReflectionPadding2D()
        self.deconv1_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv1_se1 = SqueezeExciteBlock()
        self.concat1     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv1_2   = tf.keras.layers.Conv2D(512, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv1_pd2 = ReflectionPadding2D()
        self.deconv1_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv1_3   = tf.keras.layers.Conv2D(512, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv1_pd3 = ReflectionPadding2D()
        self.deconv1_bn3 = tf.keras.layers.BatchNormalization()

        self.ups2        = ResizeLayer()
        self.deconv2_1   = tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv2_pd1 = ReflectionPadding2D()
        self.deconv2_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv2_se1 = SqueezeExciteBlock()
        self.concat2     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv2_2   = tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv2_pd2 = ReflectionPadding2D()
        self.deconv2_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv2_3   = tf.keras.layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv2_pd3 = ReflectionPadding2D()
        self.deconv2_bn3 = tf.keras.layers.BatchNormalization()

        self.ups3        = ResizeLayer()
        self.deconv3_1   = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv3_pd1 = ReflectionPadding2D()
        self.deconv3_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv3_se1 = SqueezeExciteBlock()
        self.concat3     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv3_2   = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv3_pd2 = ReflectionPadding2D()
        self.deconv3_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv3_3   = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv3_pd3 = ReflectionPadding2D()
        self.deconv3_bn3 = tf.keras.layers.BatchNormalization()

        self.ups4        = ResizeLayer()
        self.deconv4_1   = tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv4_pd1 = ReflectionPadding2D()
        self.deconv4_bn1 = tf.keras.layers.BatchNormalization()
        self.deconv4_se1 = SqueezeExciteBlock()
        self.concat4     = tf.keras.layers.Concatenate(axis=-1)
        self.deconv4_2   = tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv4_pd2 = ReflectionPadding2D()
        self.deconv4_bn2 = tf.keras.layers.BatchNormalization()
        self.deconv4_3   = tf.keras.layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_initializer='he_normal')
        self.deconv4_pd3 = ReflectionPadding2D()
        self.deconv4_bn3 = tf.keras.layers.BatchNormalization()

        self.outputs     = tf.keras.layers.Conv2D(3, (1,1), activation='sigmoid')

    def call(self, inputs, training=False, **kwargs):
        # Encoder
        c1_1   = self.conv1_1(inputs)
        c1_pd1 = self.conv1_pd1(c1_1)
        c1_bn1 = self.conv1_bn1(c1_pd1, training=training)
        c1_2   = self.conv1_2(c1_bn1)
        c1_pd2 = self.conv1_pd2(c1_2)
        c1_bn2 = self.conv1_bn2(c1_pd2, training=training)
        mp1    = self.maxpool1(c1_bn2)

        c2_1   = self.conv2_1(mp1)
        c2_pd1 = self.conv2_pd1(c2_1)
        c2_bn1 = self.conv2_bn1(c2_pd1, training=training)
        c2_2   = self.conv2_2(c2_bn1)
        c2_pd2 = self.conv2_pd2(c2_2)
        c2_bn2 = self.conv2_bn2(c2_pd2, training=training)
        mp2    = self.maxpool2(c2_bn2)

        c3_1   = self.conv3_1(mp2)
        c3_pd1 = self.conv3_pd1(c3_1)
        c3_bn1 = self.conv3_bn1(c3_pd1, training=training)
        c3_2   = self.conv3_2(c3_bn1)
        c3_pd2 = self.conv3_pd2(c3_2)
        c3_bn2 = self.conv3_bn2(c3_pd2, training=training)
        mp3    = self.maxpool3(c3_bn2)

        c4_1   = self.conv4_1(mp3)
        c4_pd1 = self.conv4_pd1(c4_1)
        c4_bn1 = self.conv4_bn1(c4_pd1, training=training)
        c4_2   = self.conv4_2(c4_bn1)
        c4_pd2 = self.conv4_pd2(c4_2)
        c4_bn2 = self.conv4_bn2(c4_pd2, training=training)
        mp4    = self.maxpool4(c4_bn2)

        # BottleNeck
        bt1 = self.diconv1(mp4)
        bt1 = self.diconv1_pd1(bt1)
        bt1 = self.diconv1_bn1(bt1)
        bt2 = self.diconv2(mp4)
        bt2 = self.diconv2_pd1(bt2)
        bt2 = self.diconv2_bn1(bt2)
        bt3 = self.diconv3(mp4)
        bt3 = self.diconv3_pd1(bt3)
        bt3 = self.diconv3_bn1(bt3)
        bt4 = self.diconv4(mp4)
        bt4 = self.diconv4_pd1(bt4)
        bt4 = self.diconv4_bn1(bt4)
        btc = self.diconcat([bt1,bt2,bt3,bt4])

        # Decoder
        x = self.ups1(btc, c4_pd1.shape[1:-1])
        x = self.deconv1_1(x)
        x = self.deconv1_pd1(x)
        x = self.deconv1_bn1(x, training=training)
        x = self.deconv1_se1(x)
        x = self.concat1([x,c4_pd1])
        x = self.deconv1_2(x)
        x = self.deconv1_pd2(x)
        x = self.deconv1_bn2(x, training=training)
        x = self.deconv1_3(x)
        x = self.deconv1_pd3(x)
        x = self.deconv1_bn3(x, training=training)

        x = self.ups2(x, c3_pd1.shape[1:-1])
        x = self.deconv2_1(x)
        x = self.deconv2_pd1(x)
        x = self.deconv2_bn1(x, training=training)
        x = self.deconv2_se1(x)
        x = self.concat2([x,c3_pd1])
        x = self.deconv2_2(x)
        x = self.deconv2_pd2(x)
        x = self.deconv2_bn2(x, training=training)
        x = self.deconv2_3(x)
        x = self.deconv2_pd3(x)
        x = self.deconv2_bn3(x, training=training)

        x = self.ups3(x, c2_pd1.shape[1:-1])
        x = self.deconv3_1(x)
        x = self.deconv3_pd1(x)
        x = self.deconv3_bn1(x, training=training)
        x = self.deconv3_se1(x)
        x = self.concat3([x,c2_pd1])
        x = self.deconv3_2(x)
        x = self.deconv3_pd2(x)
        x = self.deconv3_bn2(x, training=training)
        x = self.deconv3_3(x)
        x = self.deconv3_pd3(x)
        x = self.deconv3_bn3(x, training=training)

        x = self.ups4(x, c1_pd1.shape[1:-1])
        x = self.deconv4_1(x)
        x = self.deconv4_pd1(x)
        x = self.deconv4_bn1(x, training=training)
        x = self.deconv4_se1(x)
        x = self.concat4([x,c1_pd1])
        x = self.deconv4_2(x)
        x = self.deconv4_pd2(x)
        x = self.deconv4_bn2(x, training=training)
        x = self.deconv4_3(x)
        x = self.deconv4_pd3(x)
        x = self.deconv4_bn3(x, training=training)

        outputs = self.outputs(x)
        return outputs

class Discriminator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.conv1    = tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=[192,256,3])
        self.conv1_lr = tf.keras.layers.LeakyReLU(0.2)
        self.conv1_bn = tf.keras.layers.BatchNormalization()
        self.conv1_mp = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv2    = tf.keras.layers.Conv2D(64, (3,3), padding='same')
        self.conv2_lr = tf.keras.layers.LeakyReLU(0.2)
        self.conv2_bn = tf.keras.layers.BatchNormalization()
        self.conv2_mp = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv3    = tf.keras.layers.Conv2D(64, (3,3), padding='same')
        self.conv3_lr = tf.keras.layers.LeakyReLU(0.2)
        self.conv3_bn = tf.keras.layers.BatchNormalization()
        self.conv3_mp = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.conv4    = tf.keras.layers.Conv2D(128, (3,3), padding='same')
        self.conv4_lr = tf.keras.layers.LeakyReLU(0.2)
        self.conv4_bn = tf.keras.layers.BatchNormalization()
        self.conv4_mp = tf.keras.layers.MaxPooling2D(pool_size=(2,2))

        self.flatten  = tf.keras.layers.Flatten()
        self.dense1  = tf.keras.layers.Dense(100, activation="tanh")
        self.dense2  = tf.keras.layers.Dense(2, activation="tanh")
        self.outputs = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.conv1_lr(x)
        x = self.conv1_bn(x)
        x = self.conv1_mp(x)

        x = self.conv2(x)
        x = self.conv2_lr(x)
        x = self.conv2_bn(x)
        x = self.conv2_mp(x)

        x = self.conv3(x)
        x = self.conv3_lr(x)
        x = self.conv3_bn(x)
        x = self.conv3_mp(x)

        x = self.conv4(x)
        x = self.conv4_lr(x)
        x = self.conv4_bn(x)
        x = self.conv4_mp(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        outputs = self.outputs(x)
        return outputs
