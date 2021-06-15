from PIL import Image
import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os
import streamlit as st 

# Define the batch size, 32 is a good default
BATCH_SIZE = 32

labels_csv = pd.read_csv("./labels.csv")
labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
unique_breeds = np.unique(labels)

# Create a simple function to return a tuple (image, label)
def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label

# Define image size
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """

  # Read in image file
  image = tf.io.read_file(image_path)
  # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
  image = tf.image.decode_jpeg(image, channels=3)
  # Convert the colour channel values from 0-225 values to 0-1 values
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image to our desired size (224, 244)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  # If the data is a test dataset, we probably don't have labels
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(BATCH_SIZE)
  return data_batch


# Turn prediction probabilities into their respective label (easier to understand)
def get_pred_label(prediction_probabilities):
  """
  Turns an array of prediction probabilities into a label.
  """
  return unique_breeds[np.argmax(prediction_probabilities)]


# Create a function to unbatch a batched dataset
def unbatchify(data):
  """
  Takes a batched dataset of (image, label) Tensors and returns separate arrays
  of images and labels.
  """
  images = []
  labels = []
  # Loop through unbatched data
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels

# Function to load Our Model 
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


# Function to Read and Manupilate Images
def load_images(imgList):
    result = []
    for img in imgList :
        img_path = save_uploadedfile(img)
        result.append(img_path)
    return result

def save_uploadedfile(uploadedfile):
    
    with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:

         f.write(uploadedfile.getbuffer())
         #st.success("Saved File:{} to tempDir".format(uploadedfile.name))

    return "./tempDir/{}".format(uploadedfile.name)



try:
    # Use the full page instead of a narrow central column
    st.set_page_config(layout="wide")

    col1, col2 = st.beta_columns(2)

    col1.write("""## Upload Your Dog Image """)
    

    # Uploading the File to the Page
    uploadFile = col1.file_uploader(label="Upload image", type=['jpg', 'png'], accept_multiple_files=True)

    #Uoloading Our Model
    model = load_model('./20210326-07371616740640-All-images-Adam.h5')

    # Checking the Format of the page
    if uploadFile is not None:

        custom_image_paths = load_images(uploadFile)

        
        # Turn custom image into batch (set to test data because there are no labels)
        custom_data = create_data_batches(custom_image_paths, test_data=True)

        # Make predictions on the custom data
        custom_preds = model.predict(custom_data)
        
        # Get custom image prediction labels
        custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
        

        for i, image in enumerate(custom_image_paths):
            
            col2.write(f''' ### Your dog is a : *{str(custom_pred_labels[i])}*''')
            img = Image.open(image)
            col2.image(img)
            
            
            
    else:
        st.write("Make sure you image is in JPG/PNG Format.")


except ValueError:
    st.error('Please enter a valid input')