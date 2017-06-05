import numpy as np
from PIL import Image
import bcolz
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

np.set_printoptions(precision=4, linewidth=100)


def add_region(ax, region, gt=False, class_id_to_name=None):
    """ Adds a region to an Axes object.
    Args:
        ax: Matplotlib Axes object to add the region to.
        region: A box to draw. Defined as 
            (x, y, w, h, probability, class_id).
            - probability: [0,1] confidence in the box. Optional. 
            Doesn't apply for ground truth boxes. 
            - class_id: The class_id of this box. Optional.
        gt: bool, Is this a ground truth box (drawn in different 
            style). Also for gt boxes, region should be 
            (x, y, w, h, class_id).
        class_id_to_name: A dict mapping from a class_id to 
            class_name.
    """
    x, y, w, h = region[:4]
    # Which element is the region class
    class_index = 4
    box_text = ''
    # print the name and probability
    # GT boxes have no probability
    if not gt:
        class_index = 5
        try:
                probability = region[4]
                box_text += '{:.2f}'.format(probability)
        except IndexError:
            pass

    try:
        class_id = region[class_index]
        if class_id_to_name:
            class_id = class_id_to_name[int(class_id)]
        box_text += ' ' + class_id
    except IndexError:
        pass

    if gt:
        color = 'red'
        rect_args = {'edgecolor': color, 'linewidth': 4, 'linestyle': 'dotted'}
        text_x, text_y = x, y
    else:
        color = np.random.rand(3, )
        rect_args = {'edgecolor': color, 'linewidth': 2}
        text_x, text_y = x + w, y + h
        
    rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, **rect_args)
    ax.add_patch(rect)
    ax.text(text_x, text_y, box_text, fontsize=12, backgroundcolor='white',
            bbox=dict(facecolor=color, alpha=0.5))


def display_image_regions(img, regions=None, ground_truth_regions=None,
                          class_id_to_name=None,
                          figsize=(10, 10)):
    """ Display regions on image.
    Args:
        img: PIL Image.
        regions: Proposal regions to draw on the image. A sequence of 
            (x, y, w, h, probability, class_id). Probablity and 
            class_id are optional. Default is None for no prosals.
        ground_truth_regions: The ground truth regions for this image.
            A sequence of (x, y, w, h, probability, class_id). Class_id 
            is optional. Default is None for no GT regions.
        class_id_to_name: A dict mapping from class_id to class name.
        figsize: The size of the drawn figure. w,h tuple in inches. 
    """
    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
    ax.imshow(img)
    if regions is not None:
        for region in regions:
            add_region(ax, region, False, class_id_to_name)
    if ground_truth_regions is not None:
        # single box or many boxes
        if len(np.asarray(ground_truth_regions).shape) == 1:
            # if single, make it a single-item list
            ground_truth_regions = [ground_truth_regions]
        for box in ground_truth_regions:
            add_region(ax, box, True, class_id_to_name)
    plt.show()


def plot_files(filepaths, images_per_row=None, figsize=(12,6), 
               rows=1, titles=None):
    """ Draw several images (read from files) on the same 
        plot, in rows. 
    Args:
        filepaths: Sequence of the filepaths to the images.
        images_per_row: How many images to display in each row. 
            Default is None, to display all in one row.
        figsize: The size of the drawn figure. w,h tuple in inches. 
        rows: Number of rows.
        titles: Sequence of text titles for each image.
    """
    images = [Image.open(filepath) for filepath in filepaths]
    plot_images(images, images_per_row=images_per_row, figsize=figsize, 
               rows=rows, titles=titles)


def plot_images(images, images_per_row=None, figsize=(12,6), 
                rows=1, titles=None):
     """ Draw several images on the same plot, in rows. 
    Args:
        Images: Sequence of PIL images.
        images_per_row: How many images to display in each row. 
            Default is None, to display all in one row.
        figsize: The size of the drawn figure. w,h tuple in inches. 
        rows: Number of rows.
        titles: Sequence of text titles for each image.
    """
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
        _plots(images[start:end], figsize=figsize, rows=rows,
              titles=plot_titles)
        index += images_per_row

        
def _plots(ims, figsize=(12,6), rows=1, titles=None):
     """ Draw several images on the same plot, in rows. 
    Args:
        ims: Sequence of PIL images.
        images_per_row: How many images to display in each row. 
            Default is None, to display all in one row.
        figsize: The size of the drawn figure. w,h tuple in inches. 
        rows: Number of rows.
        titles: Sequence of text titles for each image.
    """
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation='none')


def get_image_size(file):
    """ Gets the size of an image.
    Args:
        file: the fullpath to the image.
    Returns:
        Size image, tuple of (w, h).
    """
    return Image.open(file).size


def open_image_arr(file):
    """ Open an image as np array.
    Args:
        file: the fullpath to the image.
    Returns:
        np.array of the image. Sized (h, w)
    """
    return np.asarray(Image.open(file))
