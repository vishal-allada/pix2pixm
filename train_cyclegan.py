import tensorflow as tf

from tensorflow_examples.models.pix2pix import pix2pix

import os
import time
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

import tensorflow as tf
import pathlib
from PIL import Image
import numpy as np

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    #image = tf.image.decode_png(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [256, 256])
    image = tf.cast(image, tf.uint8)
    # image = image * 2 - 1  # Scale to [-1, 1]
    return image

def load_images_from_folder(folder_path,label):
    data_root = pathlib.Path(folder_path)
    image_paths = list(data_root.glob('*'))
    image_paths = [str(path) for path in image_paths]
    labels = [label] * len(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths,labels))
    print(f"Total number of elements: {len(list(dataset))}")
    dataset = dataset.map(lambda x,y:(load_image(x),y), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

# Adjust this path to match your dataset location
base_path = './data/'
# train_horses = load_images_from_folder(f'{base_path}/trainA',label=0)
# train_zebras = load_images_from_folder(f'{base_path}/trainB',label=1)
# test_horses = load_images_from_folder(f'{base_path}/testA',label=0)
# test_zebras = load_images_from_folder(f'{base_path}/testB',label=1)

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

# train_horses = train_horses.cache().map(
#     preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
#     BUFFER_SIZE).batch(BATCH_SIZE)

# train_zebras = train_zebras.cache().map(
#     preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
#     BUFFER_SIZE).batch(BATCH_SIZE)

# test_horses = test_horses.map(
#     preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
#     BUFFER_SIZE).batch(BATCH_SIZE)

# test_zebras = test_zebras.map(
#     preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
#     BUFFER_SIZE).batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./artifacts/model2_21_feb"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                        generator_f=generator_f,
                        discriminator_x=discriminator_x,
                        discriminator_y=discriminator_y,
                        generator_g_optimizer=generator_g_optimizer,
                        generator_f_optimizer=generator_f_optimizer,
                        discriminator_x_optimizer=discriminator_x_optimizer,
                        discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

checkpoint_number = 7  # Change this to restore a specific checkpoint
checkpoint_path = f"{checkpoint_path}/ckpt-{checkpoint_number}"

if tf.io.gfile.exists(checkpoint_path + ".index"):
    ckpt.restore(checkpoint_path).expect_partial()
    print(f"Checkpoint {checkpoint_number} restored successfully!")
else:
    print(f"Checkpoint {checkpoint_number} not found!")


os.makedirs("./artifacts/testA", exist_ok=True)

c=0
for hou in os.listdir("./data"):
    house_number = int(hou.split("_")[1])
    # if house_number > 10:
    #     continue
    if os.path.exists(f"./data/{hou}/cat"):
        for pet in os.listdir(f"./data/{hou}/cat"):
            # for img in os.listdir(
            #     f"/opt/ml/processing/apt/efficientnet_data/{hou}/cat/{pet}"
            # ):
            #     print(img)
            
            test_horses = load_images_from_folder(f"./data/{hou}/cat/{pet}",label=0)
            # test_zebras = load_images_from_folder(f'{base_path}/testB',label=1)


            test_horses = test_horses.map(
                preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
                BUFFER_SIZE).batch(BATCH_SIZE)

            def generate_and_save_images(model, test_input,index):
                # Generate image
                prediction = model(test_input, training=False)

                # Convert tensor to numpy array
                predicted_image = prediction[0].numpy()

                # Convert from [-1, 1] to [0, 255]
                predicted_image = (predicted_image * 127.5 + 127.5).astype(np.uint8)
                image = Image.fromarray(predicted_image)
                # Save using OpenCV (BGR format)
                os.makedirs(f"./artifacts/testA/{hou}/cat/{pet}/",exist_ok=True)
                # cv2.imwrite(f'/opt/ml/processing/apt/testA/{hou}/cat/{pet}/generated_image_{index}.jpg', predicted_image)
                image.save(f"./artifacts/testA/{hou}/cat/{pet}/generated_image_{index}.jpg")
            # for img, _ in test_horses.take(1):
            #     plt.imshow(img[0].numpy().astype("uint8"))
            #     plt.show()

            for inp in test_horses.take(10):
                generate_and_save_images(generator_g, inp,c)
                c=c+1
        print(f"{hou} is completed")

    if os.path.exists(f"./data/{hou}/dog"):
        for pet in os.listdir(f"./data/{hou}/dog"):
            # for img in os.listdir(
            #     f"/opt/ml/processing/apt/efficientnet_data/{hou}/dog/{pet}"
            # ):
            #     print(img)
            test_horses = load_images_from_folder(f"./data/{hou}/dog/{pet}",label=0)
            # test_zebras = load_images_from_folder(f'{base_path}/testB',label=1)


            test_horses = test_horses.map(
                preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
                BUFFER_SIZE).batch(BATCH_SIZE)

            def generate_and_save_images(model, test_input,index):
                # Generate image
                prediction = model(test_input, training=False)

                # Convert tensor to numpy array
                predicted_image = prediction[0].numpy()

                # Convert from [-1, 1] to [0, 255]
                predicted_image = (predicted_image * 127.5 + 127.5).astype(np.uint8)
                image = Image.fromarray(predicted_image)
                # Save using OpenCV (BGR format)
                os.makedirs(f"./artifacts/testA/{hou}/dog/{pet}/",exist_ok=True)
                # cv2.imwrite(f'/opt/ml/processing/apt/testA/{hou}/dog/{pet}/generated_image_{index}.jpg', predicted_image)
                image.save(f"./artifacts/testA/{hou}/dog/{pet}/generated_image_{index}.jpg")

            for inp in test_horses.take(10):
                generate_and_save_images(generator_g, inp,c)
                c=c+1
        print(f"{hou} is completed")


