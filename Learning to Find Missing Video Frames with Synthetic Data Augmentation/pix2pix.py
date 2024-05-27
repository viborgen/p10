# *** imports *** #
import tensorflow as tf
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import wandb
import glob
import random

# *** General definitions *** #

#denote subject wished to be trained / tested on.
dataset = '00000' 

#initiate Weights & Biases for logging of metrics, time etc.
wandb.init(project="Alcohol Generative", name=dataset, group='fourview-stacked')

# Path to dataset. The dataset must be fed such as the image pairs are concatenated to one image.
dataset_path = f"/datasets/{dataset}/"

# General parameters defined for training.
BUFFER_SIZE = 500
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100

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
  # Resize to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)
  # Randomly crop back to 256x256
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

#loading images for the validation dataset
def load_image_val(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

#create tensors and shuffle dataset
train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
val_dataset = val_dataset.shuffle(BUFFER_SIZE)
val_dataset = val_dataset.map(load_image_val)
val_dataset = val_dataset.batch(BATCH_SIZE)


# *** Functions defining the generator and descriminator *** #

#defining generator for modification purposes. Could have just importet a standard Unet from keras, which is where these implementations are from aswell.
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)
  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer=initializer,use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result


up_model = upsample(3, 4)
up_result = up_model(down_result)


def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  
    downsample(128, 4),  
    downsample(256, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
    downsample(512, 4),  
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4, apply_dropout=True),  
    upsample(512, 4),  
    upsample(256, 4),  
    upsample(128, 4),  
    upsample(64, 4),  
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
#gen_output = generator(inp[tf.newaxis, ...], training=False)


# *** Defining losses and optimizers *** #

# defining the generator loss, both L1 and cross entropy, used for the adverserial loss.
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
  # The goal is to make the discriminator believe that the generated images are real (label=1).
  # tf.ones_like creates a tensor of ones with the same shape as disc_generated_output.
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  # Mean absolute error, helps ensure the image is close to the target image
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  # Combine the adversarial loss and the L1 loss to form the total generator loss.
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)
  return total_gen_loss, gan_loss, l1_loss

#utilizing patchGAN. Following the same downsampling layers as the generator, but then utilizes convolutions to look at individual patches of the image.
def Discriminator(): 
  initializer = tf.random_normal_initializer(0., 0.02)
  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar])  
  down1 = downsample(64, 4, False)(x)  
  down2 = downsample(128, 4)(down1)  
  down3 = downsample(256, 4)(down2)  
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  
  conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)  
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  
  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()

def discriminator_loss(disc_real_output, disc_generated_output):
  # Real loss: compare discriminator's output for real images with a tensor of ones (real label). How well is it classifying real images as real
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  # Generated loss: compare discriminator's output for fake images with a tensor of zeros (fake label). How well is it classifying fake as fake
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  # Total discriminator loss: sum of real loss and generated loss
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss

#define the optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#define checkpoints
checkpoint_dir = f'./training_checkpoints/{dataset}'
checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt{dataset}")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)


#Generate images for display and test-purposes.
def generate_images(model, test_input, tar, step=None, test=None):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input', 'Original', 'Prediction']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i],fontsize=20)
    # Getting the pixel values in the [0, 1] range to plot.
    #plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.subplots_adjust(wspace=0.02)
  if step != None:
    os.makedirs(f'pictures/dataset/{dataset}', exist_ok=True)  # Create the directory if it doesn't exist
    plt.savefig(f'pictures/dataset/{dataset}/{step}.jpg')
  if test != None:
    os.makedirs(f'pictures/dataset/{dataset}/test', exist_ok=True)  # Create the directory if it doesn't exist
    n = 1
    while os.path.exists(f'pictures/dataset/{dataset}/test/{n}.jpg'):
        n += 1
    plt.savefig(f'pictures/dataset/{dataset}/test/{n}.jpg')
  #plt.show()
  return prediction


for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# *** Training *** #

# defining a train step
@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

# Function to display an image
def display_image(image):
    plt.figure(figsize=(5, 5))
    plt.imshow(image[0] * 0.5 + 0.5)
    plt.axis('off')
    plt.show()


# defining the training loop
def fit(train_ds, test_ds, steps):
  best_loss = 1000000
  # Initial selection
  example_input, example_target = next(iter(test_ds.take(1)))
  display_image(example_input)

  # Ask the user if they want to select a new example image for the testing.
  user_input = input("Do you want to select a new image? (yes/no): ")

  while user_input.lower() != 'yes':
      display.clear_output(wait=True)
      example_input, example_target = next(iter(test_ds.take(1)))
      display_image(example_input)
      user_input = input("Do you want to select a new image? (yes/no): ")
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    # we want to validate every 10th step in the beginning up to 100 steps, then every 500 steps untill step 5000 and then every 1000 step for the rest.
    if (step < 100 and step % 10 == 0) or (step >= 100 and step < 5000 and step % 500 == 0) or (step >= 5000 and step % 1000 == 0):
      display.clear_output(wait=True)
      nrsteps = 10 if step < 100 else 500 if step < 5000 else 1000
      if step != 0:
        duration = time.time()-start
        print(f'Time taken for {nrsteps} steps: {duration} sec\n')
      start = time.time()

      val_predicted = generate_images(generator, example_input, example_target, step)
      val_l1_loss = tf.reduce_mean(tf.abs(target - val_predicted))
      print(f'l1_loss:{val_l1_loss} - above example')
      wandb.log({
        'val/Example L1 Loss': val_l1_loss,
      }, step=step)

      total_val_l1_loss = 0
      num_steps = 0
      for inp, tar in test_ds:
          predicted = generator(inp, training=True)
          l1_loss = tf.reduce_mean(tf.abs(tar - predicted))
          total_val_l1_loss += l1_loss
          num_steps += 1
      average_val_loss = total_val_l1_loss / num_steps
      print(f'Average_l1_loss:{average_val_loss}')
      wandb.log({
        'val/Average L1 Loss': average_val_loss,
      }, step=step)
      print(f"Step: {step}")

    gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, step)
    wandb.log({
      'train/Gen Total Loss': gen_total_loss,
      'train/Gen Gan Loss': gen_gan_loss,
      'train/Gen L1 Loss': gen_l1_loss,
      'train/Disc Loss': disc_loss
    }, step=step)
    # Training step
    if (step+1) % nrsteps / 100 == 0:
      print('.', end='', flush=True)


    # Save (checkpoint) the model every 200 steps
    if (step + 1) % 200 == 0:#5000 == 0:
      #checkpoint.save(file_prefix=checkpoint_prefix)
      if gen_total_loss < best_loss:
            best_loss = gen_total_loss
            for checkpoint_file in glob.glob(checkpoint_prefix + '*'):
              os.remove(checkpoint_file)
            checkpoint.save(file_prefix=checkpoint_prefix)
            print(f"Saved best model with loss {best_loss} at step {step}")

#running the train loop
fit(train_dataset, val_dataset, steps=100001)


# *** Testing *** #

# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#generate the average l1 loss for the test dataset
total_l1_loss = 0
num_steps = 0
for inp, tar in test_dataset:
    predicted = generator(inp, training=True)
    l1_loss = tf.reduce_mean(tf.abs(tar - predicted))
    total_l1_loss += l1_loss
    num_steps += 1
average_loss = total_l1_loss / num_steps
print(average_loss)
wandb.log({"Test/Average L1 Loss": average_loss})


wandb.finish()