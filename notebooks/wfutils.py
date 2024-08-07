# View an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
from pathlib import Path
import os

from tensorflow.keras.preprocessing import image as image_utils


def open_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = Path(target_dir,target_class)

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it using matplotlib
  img = mpimg.imread(target_folder.as_posix() + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print(f"Image shape: {img.shape}") # show the shape of the image

  return img


def predict_random_image(model, target_dir, class_names):

    #print("Target Dir: ", target_dir)
    i =0
    random_folder_str = random.sample(os.listdir(target_dir), 1)[0]
    random_folder = os.path.join(target_dir, random_folder_str)
    while not os.path.isdir(random_folder): 
        # print("random folder/file", random_folder.as_posix())
        i+=1
        random_folder_str = random.sample(os.listdir(target_dir), 1)[0]
        random_folder = os.path.join(target_dir, random_folder_str)
        if i > 10 :
            break
    #print("RandomFolder", random_folder)
    # Get a random image path
    random_image = random.sample(os.listdir(random_folder), 1)

    # Read in the image and plot it using matplotlib
    img = mpimg.imread(random_folder + "/" + random_image[0])
    ##plt.imshow(img)
    plt.title(random_folder)
    plt.axis("off")

    #resize and reshape
    image = image_utils.load_img(random_folder + "/" + random_image[0], color_mode="rgb", target_size=(224,224)) 
    plt.imshow(image)   
    image = image_utils.img_to_array(image)
    print(f"Image shape: {image.shape}") # show the shape of the image
    
    image = image/255 # normalize
    image = image.reshape(1, 224, 224, 3) # reshape
    print(f"Image shape: {image.shape}") # show the shape of the image
    prediction = model.predict(image)
    print("Prediction: ", class_names[np.argmax(prediction[0])])

    return prediction


#Check for file errors

import os
from PIL import Image

def check_for_bad_images(source_path):
    #folder_path = 'data\img'
    failed_files = []
    for fldr in os.listdir(source_path):
        sub_folder_path = os.path.join(source_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            print('** Path: {}  **'.format(file_path), end="\r", flush=True)
            try:
                im = Image.open(file_path)
                im.close()
            except:
                failed_files.append(file_path)
            #rgb_im = im.convert('RGB')
            # if filee.split('.')[1] not in extensions:
            #     extensions.append(filee.split('.')[1])
    return failed_files

def save_pic(figure_name, filepath, extension="png", resolution=300):
   path = os.path.join(filepath, figure_name + "." + extension )
   plt.savefig(path, format=extension, dpi=resolution)       


def plot_binary_image(image):
  plt.imshow(image, cmap="binary")
  plt.axis("off")

def plot_multiple_images(images, n_cols=None):
  n_cols = n_cols or len(images)
  n_rows = (len(images) - 1) // n_cols + 1
  if images.shape[-1] == 1:
      images = np.squeeze(images, axis=-1)
  plt.figure(figsize=(n_cols, n_rows))
  for index, image in enumerate(images):
      plt.subplot(n_rows, n_cols, index + 1)
      plt.imshow(image, cmap="binary")
      plt.axis("off")


from tqdm import tqdm
import tensorflow as tf
#from tensorflow import keras

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
  generator, discriminator = gan.layers
  for epoch in tqdm(range(n_epochs)):
      print("Epoch {}/{}".format(epoch + 1, n_epochs)) 
      batch_num = 0
      for X_batch in dataset:
          #print("Num batches seen: ", dataset.total_batches_seen, end = "\r")
          batch_num += 1
          print("Batch number: ", batch_num, end='\r')
          # to accomodate for the final batch that might have fewer good images reset batch size to actual value
          if X_batch[0].shape[0] != batch_size:
              print(f"Batch size was {X_batch[0].shape[0]}")
              break
          X_batch =tf.cast(X_batch[0], tf.float32)  #CJA added so that concat will work
          # phase 1 - training the discriminator
          
          noise = tf.random.normal(shape=[batch_size, codings_size])
          #noise
          generated_images = generator(noise)
          X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
          y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
          discriminator.trainable = True
          discriminator.train_on_batch(X_fake_and_real, y1)
          # phase 2 - training the generator
          noise = tf.random.normal(shape=[batch_size, codings_size])
          y2 = tf.constant([[1.]] * batch_size)
          discriminator.trainable = False
          gan.train_on_batch(noise, y2)
      plot_multiple_images(generated_images, 8)                     
      plt.show()                                                    