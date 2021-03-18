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
    output_list = []
    temp = rgb_to_lab(oriImg)
    l = temp[:,:,:,0]/255
    a = temp[:,:,:,1]/255
    b = temp[:,:,:,2]/255
    labImg = tf.stack([l,a,b],3)
    for index,img in enumerate(labImg):
        l = img[:,:,0]; lm = K.mean(l);
        a = img[:,:,1]; am = K.mean(a);
        b = img[:,:,2]; bm = K.mean(b);
        new_l = (l-lm)**2
        new_a = (a-am)**2
        new_b = (b-bm)**2
        sm = (new_l+new_a+new_b)
        sm = tf.expand_dims(sm,0)
        sm = tf.expand_dims(sm,3)
        output_list.append(sm[0])
    return tf.stack(output_list)
