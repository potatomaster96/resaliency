import tensorflow as tf
import tensorflow.keras.backend as K


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'REFLECT')


class SymmetricPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0] ], 'SYMMETRIC')


class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)

    def call(self, inputs, shape):
        return tf.image.resize(inputs, shape, method='nearest')


class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.eps = epsilon
        super(AdaIN, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({ 'epsilon': self.eps })
        return config

    def call(self, inputs):
        content, style, alpha = inputs
        axes = [1, 2]
        c_mean, c_var = tf.nn.moments(content, axes=axes, keepdims=True)
        s_mean, s_var = tf.nn.moments(style, axes=axes, keepdims=True)
        c_std, s_std = tf.math.sqrt(c_var + self.eps), tf.math.sqrt(s_var + self.eps)
        adain = s_std * (content - c_mean) / c_std + s_mean
        return alpha * adain + (1-alpha) * content


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters=32, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid', activation='selu')
        self.rpad1 = ReflectionPadding2D((1,1))
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='valid', activation='selu')
        self.rpad2 = ReflectionPadding2D((1,1))
        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.rpad1(x)
        x = self.conv2(x)
        x = self.rpad2(x)
        x = self.add([x, inputs])
        return x


class PUPSBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, **kwargs):
        super(PUPSBlock, self).__init__(**kwargs)
        self.concat   = tf.keras.layers.Concatenate(axis=-1)
        self.conv     = tf.keras.layers.Conv2D(filter_size, 1, strides=1)
        self.resblock = ResidualBlock(filters=filter_size)
        # self.upsample = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.upsample = ResizeLayer()
        self.outputs  = tf.keras.layers.Conv2D(output_shape, 3, padding='valid', activation='selu')

    @tf.function
    def call(self, inputs, shape):
        x = self.concat(inputs)
        x = self.conv(x)
        x = self.resblock(x)
        x = self.upsample(x, shape)
        x = self.outputs(x)
        return x


class SqueezeExciteBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, axis=-1, **kwargs):
        super(SqueezeExciteBlock, self).__init__(**kwargs)
        self.ratio = ratio
        self.axis = axis

    def build(self, input_shape):
        super(SqueezeExciteBlock, self).build(input_shape)
        self.GlobalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.GlobalAvgPooling.build(input_shape)
        self.filters = input_shape[self.axis]
        self.dense1 = tf.keras.layers.Dense(self.filters // self.ratio, activation='relu', use_bias=False)
        self.dense2 = tf.keras.layers.Dense(self.filters, activation='sigmoid', use_bias=False)

    def call(self, inputs, **kwargs):
        avg = self.GlobalAvgPooling(inputs)
        se_shape = (inputs.shape[0], 1, 1, self.filters)
        se = tf.reshape(avg, se_shape)
        d1 = self.dense1(se)
        d2 = self.dense2(d1)
        x = tf.multiply(inputs, d2)
        return x


class BatchAttNorm(tf.keras.layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(BatchAttNorm, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
        # if self.axis == -1: self.data_format = 'channels_last'
        # else: self.data_format = 'channel_first'

    def build(self, input_shape):
        if len(input_shape) != 4: raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
        super(BatchAttNorm, self).build(input_shape)

        self.GlobalAvgPooling = tf.keras.layers.GlobalAveragePooling2D("channels_last")
        self.GlobalAvgPooling.build(input_shape)

        self.weight = self.add_weight(name='weight', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(1), trainable=True)
        self.bias = self.add_weight(name='bias', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(0), trainable=True)
        self.weight_readjust = self.add_weight(name='weight_readjust', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(0), trainable=True)
        self.bias_readjust   = self.add_weight(name='bias_readjust', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(-1), trainable=True)

    def call(self, inputs):
        avg = self.GlobalAvgPooling(inputs)
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)
        bn_weights = self.weight * attention
        out_bn = super(BatchAttNorm, self).call(inputs)

        if K.int_shape(inputs)[0] is None or K.int_shape(inputs)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
            # self.bias  = self.bias[None, None, None, :]
        return out_bn * bn_weights + self.bias


class BatchInstanceNormalization(tf.keras.layers.Layer):
    """Batch Instance Normalization Layer (https://arxiv.org/abs/1805.07925)."""

    def __init__(self, epsilon=1e-5):
        super(BatchInstanceNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.rho = self.add_weight(
            name='rho',
            shape=input_shape[-1:],
            initializer='ones',
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0),
            trainable=True)

        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)

        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        batch_mean, batch_sigma = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
        x_batch = (x - batch_mean) * (tf.math.rsqrt(batch_sigma + self.epsilon))

        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_ins = (x - ins_mean) * (tf.math.rsqrt(ins_sigma + self.epsilon))

        return (self.rho * x_batch + (1 - self.rho) * x_ins) * self.gamma + self.beta
