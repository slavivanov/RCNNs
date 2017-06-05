import numpy as np

import torch
import torchvision

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])


def normalize_images(batch_images):
    """ Normalize a batch of images.
    Args:
        batch_images: ndarray of image data, dims: (batch_index, H, W, channels)
    Returns:
        A torch tensor of normalized images (0 to 1),
         with dims: (batch_index, C, H, W).
    """
    normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN,
                                                 std=IMAGENET_STD)
    images_normalized = []
    for image in batch_images:
        image_normalized = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize])(image)
        images_normalized.append(image_normalized)
    if len(images_normalized) == 1:
        return torch.cat(images_normalized)
    else:
        return torch.stack(images_normalized)

    IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])

def normalize_images(batch_images):
    """ Normalize a batch of images.
    Args:
        batch_images: ndarray of image data, dims: (batch_index, H, W, channels)
    Returns:
        A torch tensor of normalized images (0 to 1),
         with dims: (batch_index, C, H, W).
    """
    normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN,
                                                 std=IMAGENET_STD)
    images_normalized = []
    for image in batch_images:
        image_normalized = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize])(image)
        images_normalized.append(image_normalized)
    if len(images_normalized) == 1:
        return torch.cat(images_normalized)
    else:
        return torch.stack(images_normalized)

    
def denormalize(tensor):
    ''' Denormalize a tensor of images normalized by 
        normalize_images(). 
        
    Args:
        tensor: Tensor, with dims: (batch_index, channels, H, W). 
    Returns:
        An ndarray of unnormalized images ([0, 255], uint8).
            With dims (batch_index, H, W, C)
    '''
    for image in tensor:
        for t, m, s in zip(image, IMAGENET_MEAN, IMAGENET_STD):
            t.mul_(s).add_(m).mul_(255.)
    tensor = tensor.permute(0, 2, 3, 1)
    return tensor.cpu().numpy().astype('uint8')


def flip_boxes(boxes, image_width):
    """ Flips xyxy boxes horizontally.
    Args:
        boxes: ndarray of (box_index, x1, y1, x2, y2)
        image_width: the width of the image.
    Returns:
        The horizontally flipped boxes. Ndarray of 
            (box_index, x1, y1, x2, y2).
    """
    oldx1 = boxes[:, 0].copy()
    oldx2 = boxes[:, 2].copy()
    boxes[:, 0] = image_width - oldx2 - 1
    boxes[:, 2] = image_width - oldx1 - 1
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    return boxes


def get_new_size(im_size, min_size=600, max_size=1000):
    """ Figure the size of an image so that:
            1. its aspect ratio is preserved.
            2. its shortest side is min_size, given that:
            3. the longest side is no more than max_size. 
    Args:
        im_size: a sequence, size of the image.
        min_size: The minimum size of the shortest side.
        max_size: The max size of the longest side.
    Returns:
        The new size: a sequence, the new size of the image.
        Resize ratio: Int, the product of new_size / old_size.
    """
    im_size = np.array(im_size).astype('float')
    # Short side should be equal to min_size
    # (including enlarging the image).
    short_side = np.min(im_size)
    resize_ratio = min_size / short_side
    new_size = im_size * resize_ratio
    # However, we want the longer side to be no longer
    # than max_size.
    long_side = np.max(new_size)
    if long_side > max_size:
        resize_ratio = max_size / long_side
        new_size = new_size * resize_ratio

    return new_size.astype(int), resize_ratio


def x1y1x2y2_to_xywh(regions):
    """ From (x1, y1, x2, y2) to (x1, y1, w, h)."""
    regions_transformed = regions.copy()
    regions_transformed[:, 2:4] = regions[:, 2:4] - regions[:, 0:2] + 1
    return regions_transformed


def xywh_to_x1y1x2y2(regions):
    """ From (x, y, w, h) to (x1, y1, x2, y2)."""
    regions_transformed = regions.copy()
    regions_transformed[:, 2:4] = regions[:, 0:2] + regions[:, 2:4] - 1
    return regions_transformed
