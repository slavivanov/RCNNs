import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import bcolz
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

np.set_printoptions(precision=4, linewidth=100)

def get_batch(path, image_size, batch_size=32):
    ''' Create a keras data generator, that flows the images from the specified 
    dir.
    '''
    data_generator = ImageDataGenerator()
    return data_generator.flow_from_directory(directory=path,
                                              batch_size=batch_size,
                                              target_size=image_size,
                                              shuffle=False)

def get_all_generator_samples(data_gen):
    ''' Gets all samples from a keras data generator.'''
    current_index = 0
    samples = []
    data_gen.reset()
    while current_index < data_gen.n:
        samples.extend(data_gen.next()[0])
        current_index += data_gen.batch_size
        
    # The last batch might have looped from the beginning
    data_gen.reset()
    return np.asarray(samples[:data_gen.n])

def display_image_regions(img, regions, ground_truth_box=None, figsize=(10,10)):
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor=np.random.rand(3,1), linewidth=1)
        ax.add_patch(rect)
    
    if ground_truth_box is not None:
        x, y, w, h = ground_truth_box
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=4, linestyle='dotted')
        ax.add_patch(rect)
        
    plt.show()


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def plot_files(filepaths, images_per_row=None, figsize=(12,6), 
               rows=1, interp=False, titles=None):
    images = [Image.open(filepath) for filepath in filepaths]
    plot_images(images, images_per_row=images_per_row, figsize=figsize, 
               rows=rows, interp=interp, titles=titles)
        
def plot_images(images, images_per_row=None, figsize=(12,6), 
               rows=1, interp=False, titles=None):
    num_files = len(images)
    # If not images_per_row is not specified, display it all on one row
    if not images_per_row:
        images_per_row = num_files
    
    index = 0
    while index < num_files:
        start = index
        # Don't go outside the array
        end = min(index + images_per_row, num_files)
        if titles is not None:
            plot_titles = titles[start:end]
        else: 
            plot_titles = None
        plots(images[start:end], figsize=figsize, rows=rows, interp=interp, 
              titles=plot_titles)
        index += images_per_row

        
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
