# *** imports *** #
import tensorflow as tf
import wandb
import shutil
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import pathlib
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import random
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization

# *** General definitions *** #

AUTOTUNE = tf.data.AUTOTUNE
#denote subject wished to be trained / tested on.
dataset = '00000'

#initiate Weights & Biases for logging of metrics, time etc.
wandb.init(project="Alcohol Generative", name=dataset, group='individual_cycleGAN_fourview')



# Path to dataset. The dataset must be fed such as the image pairs are concatenated to one image.
dataset_path = f"/datasets/{dataset}/"

# General parameters defined for training.
BUFFER_SIZE = 500
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10
EPOCHS = 200
OUTPUT_CHANNELS = 3

# Convert the string path to a pathlib.Path object and select a random image
PATH = pathlib.Path(dataset_path)
list(PATH.iterdir())
files = os.listdir(PATH / f'train')
random_file = random.choice(files)
sample_image = tf.io.read_file(str(PATH / f'train/{random_file}'))
sample_image = tf.io.decode_jpeg(sample_image)


# *** Functions to load images, augmentation *** #

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  #each image is now splitted in two, seperating the RGB from the thermal image.
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image

# following the pix2pix paper, we resize, random_crop and normalize the images aswell as applying mirroring.
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image, real_image
def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])
  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image
  
#loading images for the train dataset
#all the different datasplits have been predefined when processing the datasets
def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

#loading images for the test dataset
def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
val_dataset = val_dataset.map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(BUFFER_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

#choose an example image
sample_RGB, sample_THERMAL = next(iter(train_dataset))



# *** Functions defining the generator and descriminator *** #

#define generator part. Defined for modification possibilities. Could have been loaded from tensorflow.
def resnet_block(input_layer, size, kernel_size=3, strides=1):
    initializer = tf.random_normal_initializer(0., 0.02)
    # Ensure input_layer is a tensor
    if not isinstance(input_layer, tf.Tensor):
        input_layer = tf.convert_to_tensor(input_layer)
    x = layers.Conv2D(size, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(input_layer)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(size, kernel_size, strides=strides, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization(axis=-1)(x)
    return layers.add([input_layer, x])

def resnet_generator():
    inputs = layers.Input(shape=[256, 256, 3])
    x = inputs
    initializer = tf.random_normal_initializer(0., 0.02)
    # Downsampling
    x = layers.Conv2D(64, 7, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization(axis=-1)(x)
    x = layers.Activation('relu')(x)
    for size in [128, 256]:
        x = layers.Conv2D(size, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
    # Resnet blocks
    for _ in range(9):
        x = resnet_block(x, 256)
    # Upsampling
    for size in [128, 64]:
        x = layers.Conv2DTranspose(size, 3, strides=2, padding='same', kernel_initializer=initializer)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = layers.Activation('relu')(x)
    outputs = layers.Conv2D(3, 7, strides=1, padding='same', kernel_initializer=initializer, activation='tanh')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

generator_g = resnet_generator()
generator_f = resnet_generator()

#defining the discriminator
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)


# *** Defining losses and optimizers *** #

#defining general loss object
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  # Real loss: compare discriminator's output for real images with a tensor of ones (real label). How well is it classifying real images as real
  real_loss = loss_obj(tf.ones_like(real), real)
  # Generated loss: compare discriminator's output for fake images with a tensor of zeros (fake label). How well is it classifying fake as fake
  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  # Total discriminator loss: sum of real loss and generated loss
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

#generators BCE loss
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

#cycle-loss
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return LAMBDA * loss1

#l1 loss to make sure the image are close to the wanted prediction.
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss


# Define optimizers
learning_rate = 2e-4
generator_g_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

if os.path.exists(checkpoint_path):
  shutil.rmtree(checkpoint_path)

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                          generator_f=generator_f,
                          discriminator_x=discriminator_x,
                          discriminator_y=discriminator_y,
                          generator_g_optimizer=generator_g_optimizer,
                          generator_f_optimizer=generator_f_optimizer,
                          discriminator_x_optimizer=discriminator_x_optimizer,
                          discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')


#generate imagees for either display purpose or test purpose.
def generate_images(model, test_input, orig=None, display=False):
  if orig is None:
    prediction = model(test_input)
    if(display):
      plt.figure(figsize=(12, 12))

      display_list = [test_input[0], prediction[0]]
      title = ['Input Image', 'Predicted Image']

      for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
      plt.show()

  else:
    prediction = model(test_input)
    if(display):
      plt.figure(figsize=(12, 12))

      display_list = [test_input[0], orig[0], prediction[0]]
      title = ['Input', 'Orignal', 'Prediction']
      
      for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i],fontsize=20)
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.02)
      plt.show()
  return prediction


# *** Training *** #

# Define the train steps. Get predictions, get loss, update gradients.
@tf.function
def train_step(real_x, real_y):
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)
    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)
    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    RGBcycle = calc_cycle_loss(real_x, cycled_x)
    ThermalCycle = calc_cycle_loss(real_y, cycled_y)
    total_cycle_loss = RGBcycle + ThermalCycle
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,discriminator_y.trainable_variables))
  return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, total_cycle_loss, gen_g_loss, gen_f_loss, RGBcycle, ThermalCycle


step = 0
best_loss = 99999
total_val_l1_loss = 0
total_val_l1_loss_RGB = 0
num_val_steps = 0
for epoch in range(EPOCHS):
  start = time.time()
  n = 0
  for image_x, image_y in train_dataset:
    step += 1
    total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss, total_cycle_loss, gen_g_loss, gen_f_loss, RGBcycle, ThermalCycle = train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1
    wandb.log({
        "train/total_gen_g_loss": total_gen_g_loss,
        "train/total_gen_f_loss": total_gen_f_loss,
        "train/disc_x_loss": disc_x_loss,
        "train/disc_y_loss": disc_y_loss,
        "train/total_cycle_loss": total_cycle_loss,
        "train/gen_g_loss": gen_g_loss,
        "train/gen_f_loss": gen_f_loss,
        "train/RGBcycle": RGBcycle,
        "train/ThermalCycle": ThermalCycle
    }, step=step)
  clear_output(wait=True)
  prediction = generate_images(generator_g, sample_RGB, sample_THERMAL, display=True)
  l1_loss = tf.reduce_mean(tf.abs(sample_THERMAL - prediction))
  wandb.log({
      "val/L1 Loss Validation Example": l1_loss
  }, step=step)

  for inp, tar in val_dataset:
      predicted = generate_images(generator_g, inp, tar)
      predictedRGB = generate_images(generator_f, tar, inp)
      l1_loss = tf.reduce_mean(tf.abs(tar - predicted))
      l1_loss_RGB = tf.reduce_mean(tf.abs(inp - predictedRGB))
      total_val_l1_loss += l1_loss
      total_val_l1_loss_RGB += l1_loss_RGB
      num_val_steps += 1
  average_val_loss = total_val_l1_loss / num_val_steps
  average_val_loss_RGB = total_val_l1_loss_RGB / num_val_steps
  print(f'Average_l1_loss thermal:{average_val_loss}, Average_l1_loss RGB:{average_val_loss_RGB}')
  wandb.log({
    'val/Average L1 Loss thermal': average_val_loss,
    'val/Average L1 Loss RGB': average_val_loss_RGB,
  }, step=step)

  if (epoch + 1) % 1 == 0:
    if average_val_loss < best_loss:
      best_loss = average_val_loss
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))
      wandb.log({
          "train/Best Model": best_loss
      }, step=step)

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))



# *** Testing *** #

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

# Generate using test dataset and get loss scores.
for z, (inp, orig) in enumerate(test_dataset):
  prediction = generator_g(inp)
  l1_loss = tf.reduce_mean(tf.abs(orig - prediction))
  wandb.log({'test/L1 Loss Test Example': l1_loss})
  plt.figure(figsize=(12, 12))

  display_list = [inp[0], orig[0], prediction[0]]
  title = ['Input', 'Orignal', 'Prediction']
  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i],fontsize=20)
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.02)
  directory = f'/pictures_cycle/{dataset}/test'
  if not os.path.exists(directory):
    os.makedirs(directory)
  plt.savefig(f'{directory}/{z}.png')
  #plt.show()
  plt.close()

wandb.finish()


