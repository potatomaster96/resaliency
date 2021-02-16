import os
import cv2
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm

import tensorflow as tf

from utils import read_image
from networks import DSRNetwork

# ======================================
# argument parser

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs',     dest='epochs',     type=int,   default=1000, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int,   default=2,    help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int,   default=None, help='image resolution during training')
parser.add_argument('--lr',         dest='lr',         type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--lr_gen',     dest='lr_gen',     type=float, default=1e-4, help='initial learning rate for generator')
parser.add_argument('--lr_disc',    dest='lr_disc',    type=float, default=1e-4, help='initial learning rate for discriminator')
parser.add_argument('--eval_rate',  dest='eval_rate', default=100,  help='evaluating and saving checkpoints every # epoch')
parser.add_argument('--ckpt_dir',   dest='ckpt_dir', default='checkpoints', help='directory for checkpoints')
args = parser.parse_args()

# ======================================
# set tensorflow to use gpu
my_devices = tf.config.experimental.list_physical_devices(device_type='GPU')
print("Available GPU Devices:", len(my_devices))
if len(my_devices) != 0:
    gpu = my_devices[0] # default to 1st gpu
    tf.config.experimental.set_visible_devices(devices=gpu, device_type='GPU')
    tf.config.experimental.set_memory_growth(gpu, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.keras.backend.set_learning_phase (True)
tf.config.experimental_run_functions_eagerly(True)

# ======================================
# set keras tensor format to channel last
tf.keras.backend.set_image_data_format('channels_last')

# ======================================
# set training parameters
epochs             = args.epochs
learning_rate_gen  = args.lr_gen
learning_rate_disc = args.lr_disc
learning_rate      = learning_rate_gen if learning_rate_gen == learning_rate_disc else args.lr
lr_decay_rate      = 5e-5
patch_size         = args.patch_size
batch_size         = args.batch_size
eval_rate          = args.eval_rate
ckpt_dir           = args.ckpt_dir

# weights parameters
loss_weights = {
    "gen_adv"       : 0.0, # 0.5
    "perceptual"    : 1.0, # 5.0
    "saliency"      : 0.0, # 2.0
    "hue"           : 0.0, # 5.0
    "aesthetics"    : 10.0,
    "discriminator" : 1.0
}

# ======================================
# Prepare images
train_sr_data      = glob.glob('SAM_dataset/img_resize/*')
train_saliency_map = glob.glob('SAM_dataset/guide_sal/*')
real_image         = glob.glob('SAM_dataset/resize_beauty/train/*')
ori_sal_path       = 'SAM_dataset/ori_sal/*'
checkpoint_dir     = "checkpoints/"

# ======================================
# Set up training parameters
tf.random.set_seed(3) # 3324

writer_path = "logs/"
if not os.path.exists(writer_path): os.makedirs(writer_path)
writer = tf.summary.create_file_writer(writer_path)

# ==================================================================================
# Selecting images for training
train_src = train_sr_data[:1000]
train_ref = train_saliency_map[:1000]

# Load images from path
print("Loading training images")
training_src = [read_image(img_path, patch_size) for img_path in tqdm(train_src)]
training_ref = [read_image(img_path, patch_size) for img_path in tqdm(train_ref)]

# convert image lists to numpy arrays
training_src = np.asarray(training_src)
training_ref = np.asarray(training_ref)

# Convert to tf dataset
tf_train_data = tf.data.Dataset.from_tensor_slices((training_src, training_ref))

# ==================================================================================
# optimizers
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate, decay_steps=1, decay_rate=lr_decay_rate
)

lr_schedule_gen = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate_gen, decay_steps=1, decay_rate=lr_decay_rate
)

lr_schedule_disc = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate_disc, decay_steps=1, decay_rate=lr_decay_rate
)

optimizer      = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
optimizer_gen  = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

# ==================================================================================
# set up training data iteration
tf_train_data = tf_train_data.repeat().shuffle(30).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_iter = iter(tf_train_data)

# ==================================================================================
# DSRNetwork
# pass weights as parameters during init, if no values are passed,
# all values are defaulted to 1.0
model = DSRNetwork(loss_weights)

# Checkpoint
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
else:
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)

# ==================================================================================

@tf.function
def train_step_1(input_images, guiding_sal, step=0):
    with tf.GradientTape() as dsr_tape, tf.GradientTape() as disc_tape:
        loss = model.train_batch(input_images, guiding_sal)
    gen_trainables = model.get_gen_trainables()
    disc_trainables = model.get_disc_trainables()
    if (step+1)%2 == 0:
        dsr_gradients = dsr_tape.gradient(loss[:-1], gen_trainables)
        optimizer_gen.apply_gradients(zip(dsr_gradients, gen_trainables))
    if (step+1)%2 != 0:
        disc_gradients = disc_tape.gradient(loss[-1], disc_trainables)
        optimizer_disc.apply_gradients(zip(disc_gradients, disc_trainables))
    return loss

@tf.function
def train_step_2(input_images, guiding_sal, step=0):
    with tf.GradientTape() as dsr_tape, tf.GradientTape() as disc_tape:
        loss = model.train_batch(input_images, guiding_sal)
    dsr_gradients  = dsr_tape.gradient(loss[:-1], model.generator.trainable_variables)
    disc_gradients = disc_tape.gradient(loss[-1], model.discriminator.trainable_variables)
    optimizer_gen.apply_gradients(zip(dsr_gradients, model.generator.trainable_variables))
    optimizer_disc.apply_gradients(zip(disc_gradients, model.discriminator.trainable_variables))
    return loss

@tf.function
def train_step_3(input_images, guiding_sal, step=0):
    with tf.GradientTape() as tape:
        loss = model.train_batch(input_images, guiding_sal)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# ==================================================================================
# begin training
with writer.as_default():
    for i in tqdm(range(epochs)):
        input_images, guiding_sal = next(train_iter)
        g_loss, p_loss, s_loss, h_loss, a_loss, d_loss = train_step_1(input_images, guiding_sal, i)
        dsr_loss = g_loss + p_loss + s_loss + h_loss + a_loss
        with tf.name_scope("Generator") as scope:
            tf.summary.scalar("Gen Adversarial Loss", g_loss, step=i)
            tf.summary.scalar("Perceptual Loss", p_loss, step=i)
            tf.summary.scalar("Saliency Loss", s_loss, step=i)
            tf.summary.scalar("Hue Loss", h_loss, step=i)
            tf.summary.scalar("Aesthetics Loss", a_loss, step=i)
            tf.summary.scalar("Generator Loss", dsr_loss, step=i)
        with tf.name_scope("Discriminator") as scope:
            tf.summary.scalar("Discriminator Loss", d_loss, step=i)
        writer.flush()
        ckpt.step.assign_add(1)
        if (i+1) % eval_rate == 0:
            print('Writing example images...')
            input_images, guiding_sal = next(train_iter)
            outputs = model(input_images, guiding_sal)
            with tf.name_scope("Inputs") as scope:
                tf.summary.image("Input Images", input_images, step=i)
                tf.summary.image("Guiding Saliency Map", guiding_sal, step=i)
            tf.summary.image("Output Images", outputs, step=i)
            print('Writing complete')
        if (i+1) % eval_rate == 0 or (i+1) == epochs:
            print('Saving model checkpoint...')
            ckpt_path = manager.save()
            print('Saved model checkpoint to %s' % ckpt_path)
