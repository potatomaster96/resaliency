import tensorflow as tf
from models import DSR_Base, Discriminator
from helper import build_nima_model, build_vgg_extractor, frequency_saliency_detection


class DSRNetwork(tf.keras.Model):
    def __init__(self, loss_weights=None, **kwargs):
        super(DSRNetwork, self).__init__(**kwargs)
        self.loss_weights = self.build_weights(loss_weights)
        self.generator = DSR_Base()
        self.discriminator = Discriminator()
        self.nima = build_nima_model()
        self.extractor = build_vgg_extractor()
        self.bce = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def mse(self, c_true, c_pred):
        loss = tf.math.reduce_mean(tf.math.square(c_true - c_pred))
        return loss

    @tf.function
    def mae(self, c_true, c_pred):
        loss = tf.math.reduce_mean(tf.math.squared_difference(c_true, c_pred))
        return loss

    @tf.function
    def perceptual(self, c_true, c_pred):
        vggX = self.extractor(c_pred)
        vggY = self.extractor(c_true)
        loss = [self.mse(x,y) for x,y in zip(vggX,vggY)]
        loss = tf.math.add_n(loss)
        return loss / 5.

    @tf.function
    def saliency(self, saliency_pred, guiding_sal):
        # saliency_pred = frequency_saliency_detection(c_pred)
        loss = self.bce(guiding_sal, saliency_pred)
        return loss

    @tf.function
    def color(self, groundtruth, predicted):
        style_color_mean, style_color_var = tf.nn.moments(groundtruth, [1, 2])
        gen_color_mean, gen_color_var     = tf.nn.moments(predicted, [1, 2])
        color_sigmaS = tf.math.sqrt(style_color_var)
        color_sigmaG = tf.math.sqrt(gen_color_var)
        l2_mean  = tf.math.reduce_sum(tf.math.square(gen_color_mean - style_color_mean))
        l2_sigma = tf.math.reduce_sum(tf.math.square(color_sigmaG - color_sigmaS))
        return (l2_mean + l2_sigma)

    @tf.function
    def color2(self, c_true, c_pred):
        loss = tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_pred, labels=c_true))
        return loss

    @tf.function
    def color3(self, c_true, c_pred):
        hsv_true = tf.image.rgb_to_hsv(c_true)
        hsv_pred = tf.image.rgb_to_hsv(c_pred)
        lab_true = tf_image.rgb_to_lab(c_true)
        lab_pred = tf_image.rgb_to_lab(c_pred)

        # loss_hsv = self.color(hsv_true, hsv_pred)
        loss_hsv = self.color(hsv_true[:,:,:,1:], hsv_pred[:,:,:,1:])
        loss_lab = self.color(lab_true, lab_pred)
        loss = loss_hsv + loss_lab
        return loss

    @tf.function
    def hue(self, c_true, c_pred, guiding_sal):
        hsv_true = tf.image.rgb_to_hsv(c_true)
        hsv_pred = tf.image.rgb_to_hsv(c_pred)
        gray_gd  = tf.image.rgb_to_grayscale(guiding_sal)
        # print('gray_gd', gray_gd.shape)
        loss = tf.abs(hsv_true[:,:,:,0] - hsv_pred[:,:,:,0])
        loss *= tf.squeeze(gray_gd, axis=-1)
        return tf.reduce_mean(loss)

    @tf.function
    def aesthetics(self, c_pred):
        mobileNet_input = tf.keras.applications.mobilenet.preprocess_input(c_pred)
        nima_output = self.nima(tf.image.resize(mobileNet_input,(224,224)))
        normalized = nima_output/tf.reduce_sum(nima_output)
        aesthetic_score = tf.reduce_sum(normalized*tf.range(1.0,11.0))/10
        loss = tf.square(1 - aesthetic_score)
        return loss

    def call(self, inputs, guiding_sal, **kwargs):
        concated_inputs = tf.concat([inputs, guiding_sal], axis=-1)
        outputs = self.generator(concated_inputs)
        outputs = tf.clip_by_value(outputs, 0., 1.)
        return outputs

    def train_batch(self, inputs, guiding_sal, **kwargs):
        concated_inputs = tf.concat([inputs, guiding_sal], axis=-1)
        retargetted     = self.generator(concated_inputs)
        retargetted     = tf.clip_by_value(retargetted, 0., 1.)
        saliency_pred   = frequency_saliency_detection(retargetted)

        real = self.discriminator(inputs)
        fake = self.discriminator(retargetted)

        disc_loss = tf.reduce_mean(fake**2) + tf.reduce_mean((real-1)**2)
        discriminator_loss = disc_loss * self.loss_weights["discriminator"]

        generator_loss  = tf.reduce_mean((fake-1)**2) * self.loss_weights["gen_adv"]
        perceptual_loss = self.perceptual(inputs, retargetted) * self.loss_weights["perceptual"]
        saliency_loss   = self.saliency(saliency_pred, guiding_sal) * self.loss_weights["saliency"]
        hue_loss        = self.hue(inputs, retargetted, guiding_sal) * self.loss_weights["hue"]
        aesthetics_loss = self.aesthetics(retargetted) * self.loss_weights["aesthetics"]

        return generator_loss, perceptual_loss, saliency_loss, hue_loss, aesthetics_loss, discriminator_loss

    def get_gen_trainables(self):
        return self.generator.trainable_variables

    def get_disc_trainables(self):
        return self.discriminator.trainable_variables

    def build_weights(self, loss_weights):
        if loss_weights == None:
            loss_weights = {
                "gen_adv"       : 1.0,
                "perceptual"    : 1.0,
                "saliency"      : 1.0,
                "hue"           : 1.0,
                "aesthetics"    : 1.0,
                "discriminator" : 1.0
            }
        return loss_weights
