import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def import_img(img_path,img_rows):
    '''
    Imports the `img` from the directory and resizes it while keeping the same aspect ratio.

    Args:
        `img_path`: Path of the image to be imported in the directory.
        `img_rows`: The height of the image you want to resize it into.
        (Can't set the width of the imported image as it is self computed by this function to keep the aspect ratio constant)
    
    Returns:
        image imported from the directory with the chosen size.
    '''
    # Load image from the directory
    img = tf.keras.preprocessing.image.load_img(img_path)
    # Convert the image to numeric arrays
    img = tf.keras.preprocessing.image.img_to_array(img)
    # Extract the height, width, color channels from the image
    (height, width, channels) = img.shape
    # Compute the width of the image while keeping the same aspect ratio
    img_cols = img_rows*(width/height)
    # Resize the image into the new size
    img = tf.image.resize(img,size=(img_rows,int(img_cols)))
    return img

def preprocess_img(img):
    '''
    Preprocesses the `img` to make it ready and most learnable for the VGG19 model.

    Args:
        `img`: input image to be made ready to be sent in the VGG19 model.
    
    Returns:
        `img` after preprocessing, ready to be sent into the VGG19 model.
    '''    
    # Add a batch dimension
    img = tf.expand_dims(img,axis=0)
    # Preprocess the image
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

def show(*images,figsize=[10,10],normalise_all = True):
    '''
    Plot all the `images` passed into the function.

    Args:
        `images`: Is a `*args` variable and is a list that contains all the images passed to the function.
        `figsize`: The size of the figure in which all the subplots are made.
        `normalise_all`: If true, divides all the pixel values of all the images by 255.
    '''
    # Get the number of images passed to the function
    no_of_images = len(images)
    # Calculate the number of rows required to plot all the images passed
    rows = int((no_of_images+1)/2) if no_of_images%2 else int(no_of_images/2)
    # Set the size of the figure
    plt.figure(figsize=figsize)
    # Plot the images using plt.subplot
    for i,img in enumerate(images):
        plt.subplot(rows,2,i+1)
        plt.axis(False)
        plt.imshow(img/255) if normalise_all else plt.imshow(img) 
    plt.show()
    
def deprocess_image(img):
    '''
    Deprocesses the `img` by removing the zero-center and converting BGR to RGB for plotting the image.

    Args;
        `img`: input image for deprocessing.
    '''
    # squeeze any batch dimension for plotting
    img = tf.squeeze(img).numpy()
    # Removing the zero-center which was needed by VGG19 model
    img[:, :, 0] = img[:, :, 0] + 103.939
    img[:, :, 1] = img[:, :, 1] + 116.779
    img[:, :, 2] = img[:, :, 2] + 123.68
    # Convert the colour channels from BGR to RGB
    img = img[:, :, ::-1]
    # Clip all pixels values less than 0 to 0 and pixels values greater than 255 to 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img