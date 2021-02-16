import tensorflow as tf
from tensorflow.keras.applications import MobileNet, VGG19
from color_convert import rgb_to_lab
import tensorflow.keras.backend as K
from utils import _gaussian_kernel


def build_nima_model():
    base_model = MobileNet(input_shape=(224, 224, 3), weights=None, include_top=False, pooling='avg')
    x = tf.keras.layers.Dropout(0)(base_model.output)
    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    nima_model = tf.keras.Model(inputs=base_model.inputs, outputs=x)
    nima_model.load_weights("weights_mobilenet_aesthetic_0.07.hdf5")
    nima_model.trainable = False
    return nima_model


def build_vgg_extractor():
    vgg = VGG19(include_top=False, weights='imagenet')
    content_layers = ['block1_conv1','block1_conv2','block2_conv1','block2_conv2','block3_conv1']
    lossModel = tf.keras.Model(inputs=vgg.input, outputs=[vgg.get_layer(x).output for x in content_layers])
    lossModel.trainable = False
    return lossModel


def frequency_saliency_detection(oriImg):
    blur = _gaussian_kernel(21, 5, 1, "float32")
    output_list = []
    temp = rgb_to_lab(oriImg)
    l = tf.expand_dims(temp[:,:,:,0], axis=-1)
    a = tf.expand_dims(temp[:,:,:,1], axis=-1)
    b = tf.expand_dims(temp[:,:,:,2], axis=-1)
    l = l/100
    a = (a + 86.185) / 184.439
    b = (b + 107.863) / 202.345
    labImg = tf.concat([l,a,b],axis=-1)
    index = 0

    for i in range(labImg.shape[0]):
        img = labImg[0]
        l = img[:,:,0]; lm = K.mean(l);
        a = img[:,:,1]; am = K.mean(a);
        b = img[:,:,2]; bm = K.mean(b);
        new_l = (l-lm)**2
        new_a = (a-am)**2
        new_b = (b-bm)**2
        sm = (new_l+new_a+new_b)
        sm = tf.clip_by_value(sm,0,1)
        thresholdVal = (2/(sm.shape[0]*sm.shape[1])) * K.sum(sm)
        safe_exp = tf.where(sm <= thresholdVal, 1.0, sm)
        sm = tf.where(sm<=thresholdVal, 0., safe_exp)
        sm = tf.expand_dims(sm,0)
        sm = tf.expand_dims(sm,3)
        blurred = tf.nn.depthwise_conv2d(tf.cast(sm,"float32"), blur, [1,1,1,1], 'SAME')
        blurred = tf.clip_by_value(blurred,0,1)
        index += 1
        output_list.append(blurred)
    return tf.concat(output_list, axis=0)
