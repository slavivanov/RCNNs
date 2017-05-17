#import keras
#from keras.preprocessing.image import ImageDataGenerator
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

def add_region(ax, region, gt=False, class_id_to_name=None):
    x, y, w, h = region[:4]

    if gt:
        color = 'red'
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor=color, linewidth=4, linestyle='dotted')
    else:
        color = np.random.rand(3, 1)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)

    # print the name
    try:
        class_id = region[4]
    except IndexError:
        class_id = None
        
    if class_id:
        if class_id_to_name:
            class_name = class_id_to_name[class_id]
        else:
            class_name = class_id
        if gt:
            ax.text(x, y, class_name, fontsize=12, backgroundcolor='white',
                   bbox=dict(facecolor=color, alpha=0.5))
        else:
            ax.text(x + w, y + h, class_name, fontsize=12, backgroundcolor='white',
                   bbox=dict(facecolor=color, alpha=0.5))


def display_image_regions(img, regions, ground_truth_regions=None,
                          class_id_to_name=None,
                          figsize=(10, 10)):
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    ax.imshow(img)
    for region in regions:
        add_region(ax, region, False, class_id_to_name)
    if ground_truth_regions is not None:
        boxes = (ground_truth_regions)
        # single box or many boxes
        if len(np.asarray(ground_truth_regions).shape) == 1:
            # if single, make it a single-item list
            ground_truth_regions = [ground_truth_regions]
        for box in ground_truth_regions:
            add_region(ax, box, True, class_id_to_name)
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

def get_image_size(file):
    ''' Gets the size of an image.
    Args:
        file: the fullpath to the image.
    Returns:
        Size image, tuple of (w, h).
    '''
    return Image.open(file).size

def resize_bbox(bbox, original_size, new_size):
    ''' Resize a single bbox. '''
    output_width, output_height = new_size
    img_width, img_height = original_size

    # how much to scale each bbox dimension
    img_scale_x = output_width / img_width
    img_scale_y = output_height / img_height

    # Scale bbox
    bbox[0] *= img_scale_x
    bbox[1] *= img_scale_y
    bbox[2] *= img_scale_x
    bbox[3] *= img_scale_y

    return bbox
