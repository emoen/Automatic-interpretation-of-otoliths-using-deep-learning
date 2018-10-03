from keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import scipy
from scipy.misc import imsave, imresize

def shrink_img_grayscale(new_shape, x):
    """reads images as grayscale given by path in x and resizes them to new_shape returning array of numpy arrays of images

    >>> initial_shape = (299, 299, 1)
    >>> x = ['./batch1/2006_02098sep1_t01a.jpg', './batch1/2006_02098sep1_t01b.jpg', './batch1/2006_02098sep1_t02a.jpg']
    >>> shrink_img_grayscale( initial_shape, x).shape
    (3, 299, 299, 1)
    """

    x_otolith_rescaled = np.empty(shape=(len(x),)+new_shape)
    for ex in range(0, len(x)):
        pil_img = load_img(x[ex], grayscale=True)
        smaller_img = pil_img.resize( (new_shape[1], new_shape[0]))
        x_otolith_rescaled[ex] = img_to_array(smaller_img)
    return x_otolith_rescaled

def shrink_img_rgb( new_shape, x):
    """reads images as grayscale given by path in x and resizes them to new_shape returning array of numpy arrays of images

    >>> initial_shape = (400, 400, 3)
    >>> x = ['./batch1/2006_02098sep1_t01a.jpg', './batch1/2006_02098sep1_t01b.jpg', './batch1/2006_02098sep1_t02a.jpg']
    >>> shrink_img_rgb( initial_shape, x).shape
    (3, 400, 400, 3)
    """

    x_otolith_rescaled = np.empty(shape=(len(x),)+new_shape)
    for ex in range(0, len(x)):
        pil_img = load_img(x[ex], grayscale=False)
        an_img = img_to_array(pil_img)
        x_otolith_rescaled[ex] = scipy.misc.imresize( an_img, new_shape)
    return x_otolith_rescaled 


def shrink_img_rgb_cropped_images( new_shape, x ):
    """read all the images given by the path of x
       assuming all images are cropped so that they are
       less than or equal to 1300 x 1944
       Past each image to a 1300 x 1944 white image and
       resize images to new_shape. Add these images
       to x_otolith_rescaled. After all images
       are cropped - x_otolith_rescaled has shape
       (training_examples + (new_shape) and
       return x_otolith_rescaled

    >>> initial_shape = (299, 299, 3)
    >>> x = ['./batch1/2006_02098sep1_t01a.jpg', './batch1/2006_02098sep1_t01b.jpg', './batch1/2006_02098sep1_t02a.jpg']
    >>> shrink_img_rgb( initial_shape, x).shape
    (3, 299, 299, 3)
    """

    x_otolith_rescaled = np.empty(shape=(len(x),)+new_shape)
    for ex in range(0, len(x)):
        pil_img = load_img(x[ex],grayscale=False)
        img_1300_1944 = Image.new('RGB', (1300, 1944), (255, 255, 255)) # White
        img_1300_1944.paste(pil_img, (0,0))
        an_img = img_to_array( img_1300_1944 )
        x_otolith_rescaled[ex] = scipy.misc.imresize( an_img, new_shape)
    return x_otolith_rescaled

if __name__ == '__main__':
    import doctest
    doctest.testmod()

