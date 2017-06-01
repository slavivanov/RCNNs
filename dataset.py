import _init_paths
import sys
import os
import importlib
import shutil
import numpy as np
import threading
# import keras
import tqdm
#from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import scipy
import scipy.io as scipyio

import selectivesearch
importlib.reload(selectivesearch)
from selectivesearch import get_selective_search_regions

import shutil
import copy
import PIL

from IPython.core.debugger import Tracer

import bbox_transform; importlib.reload(bbox_transform)
from bbox_transform import *

from roi_pooling.modules.roi_pool import RoIPool as RoIPool_GPU

import rcnn_utils; importlib.reload(rcnn_utils); 
from rcnn_utils import *

import pascal_voc_reader; importlib.reload(pascal_voc_reader)

import data_utils; importlib.reload(data_utils); 
from data_utils import *

import utils.general; importlib.reload(utils.general); 
from utils.general import *

import datasets.ds_utils as ds_utils; importlib.reload(ds_utils);

import utils.cython_bbox as cython_bbox; importlib.reload(cython_bbox);
from utils.cython_bbox import bbox_overlaps

import roi_data_layer.roidb_mine; importlib.reload(roi_data_layer.roidb_mine);
from roi_data_layer.roidb_mine import add_bbox_regression_targets

import roi_data_layer.minibatch as minibatch; importlib.reload(minibatch);
from roi_data_layer.minibatch import _sample_rois
from torch.utils.data.dataset import Dataset as torch_dataset


class Dataset(torch_dataset):
    def __init__(self, cache_dir):
        super().__init__()
        self.set_cache_dir(cache_dir)
        
    def __len__(self):
        raise NotImplementedError
        
    def __getitem__(self, n):
        raise NotImplementedError
        
    def load_cache(self, filename):
        ''' Load data from cache.'''
        fullpath = os.path.join(self.cache_dir, filename)
        if os.path.exists(fullpath + '.bcolz'):
            return load_array(fullpath + '.bcolz')
        elif os.path.exists(fullpath + '.pkl'):
            with open(fullpath + '.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError

    def save_cache(self, filename, to_save, use_pickle=False):
        ''' Save data to cache.'''
        fullpath = os.path.join(self.cache_dir, filename)
        if use_pickle or type(to_save) == dict:
            with open(fullpath + '.pkl', 'wb') as f:
                pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            save_array(fullpath + '.bcolz', to_save)

    def set_cache_dir(self, dirs):
        '''Set the directory where to save cached data.'''
        # create the directories if needed
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        self.cache_dir = dirs

    def clear_cache(self, filename=None):
        ''' Delete a specific file in cache dir or all of 
            them if unspecified. '''
        if filename:
            shutil.rmtree(os.path.join(self.cache_dir, filename))
        elif self.cache_dir:
            shutil.rmtree(self.cache_dir)
            # recreate the folder
            self.set_cache_dir(self.cache_dir)


class Pascal_VOC(Dataset):
    def __init__(self, 
                 data_path=None, set_name=None, year=None, 
                 mode=None):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = '../VOC/VOCdevkit/'

        if set_name:
            self.set_name = set_name
        else:
            self.set_name = 'train'

        if year:
            self.year = year
        else:
            self.year = '2007'

        cache_dir = os.path.join('intermediate/voc/',
                                 self.year, self.set_name)
        super().__init__(cache_dir)
        if mode:
            self.mode = mode
        else:
            self.mode = 'train'
            
        self.sample_index = 0
        self.min_prop_box_side = 10
        self.get_samples()
        
    def __len__(self):
        return len(self.samples)
        
    def get_samples(self):
        self.get_class_mapping()
        try:
            # Load from cache
            self.samples = self.load_cache('samples')
            self.targets_mean = self.load_cache('targets_mean')
            self.targets_std = self.load_cache('targets_std')
            self.get_filenames()
        except FileNotFoundError:
            # Read from the dataset and cache

            self.samples = pascal_voc_reader.get_data(
                self.data_path,
                set_name=self.set_name,
                year=self.year)

            self.get_filenames()

            # adds the file sizes to the samples
            print('Getting images size')
            self.get_image_sizes()

            # format the ground truth boxes for samples
            print('Formatting GT boxes')
            self.get_gt_boxes()

            # add the ss roi proposals to the samples
            print('Getting ROI proposals')
            self.get_proposals()

            print('Preparing GT and ROI overlaps.')
            self.get_overlaps()

            print('Flipping samples')
            self.flip_samples()
            # Update the filenames after we have flipped the samples
            self.get_filenames()

            print('Computing Regression Targets')
            (self.targets_mean,
             self.targets_std) = add_bbox_regression_targets(self.samples)
            # These will be needed to compute unnormalize the computed bboxes
            self.save_cache('targets_mean', self.targets_mean)
            self.save_cache('targets_std', self.targets_std)

            self.save_cache('samples', self.samples, use_pickle=True)

    def get_class_mapping(self):
#        classes = ['__background__', 'person', 'bottle', 'tvmonitor',
#            'boat', 'chair', 'sofa', 'cow', 'diningtable',
#            'train', 'pottedplant', 'aeroplane', 'motorbike',
#            'car', 'cat', 'dog', 'bird', 'bicycle',
#            'horse', 'bus', 'sheep']
        classes = ('__background__',  # always index 0
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')
        self.class_name_to_id = dict(zip(classes, range(len(classes))))
        self.class_id_to_name = {v:k for k,v in self.class_name_to_id.items()}
            
    def add_background_class(self):
        if '__background__' not in self.class_name_to_id.keys():
            # Add the background class with index 0
            self.class_name_to_id = {k: v + 1 for k, v
                                     in self.class_name_to_id.items()}
            self.class_name_to_id['bg'] = 0

    def get_filenames(self):
        self.filenames = [sample['filename']
                          for sample in self.samples]

        self.filepath = [sample['filepath']
                         for sample in self.samples]

        self.num_classes = len(self.class_name_to_id)

    def get_gt_boxes(self):
        samples_boxes = [sample['bboxes'] for sample in self.samples]

        for sample_index, sample_boxes in enumerate(samples_boxes):
            resize_ratio = self.samples[sample_index]['resize_ratio']
            xyxy_boxes = np.zeros((len(sample_boxes), 5))
            for box_index, box in enumerate(sample_boxes):
                # Indexes start at 1 in Pascal VOC
                x1 = (box['x1'] - 1) * resize_ratio
                y1 = (box['y1'] - 1) * resize_ratio
                x2 = (box['x2'] - 1) * resize_ratio
                y2 = (box['y2'] - 1) * resize_ratio

                class_name = box['class']
                class_index = self.class_name_to_id[class_name]

                xyxy_boxes[box_index] = np.array([x1, y1, x2, y2,
                                                  class_index])

            self.samples[sample_index]['gt_boxes_xyxy'] = xyxy_boxes
            xywh_boxes = x1y1x2y2_to_xywh(xyxy_boxes)
            self.samples[sample_index]['gt_boxes_xywh'] = xywh_boxes

    def get_proposals(self):
        fullpath = os.path.join(
            self.data_path,
            'VOC' + self.year,
            'SelectiveSearchRoIs',
            'voc_' + self.year + '_' + self.set_name + '.mat')
        saved_boxes = scipyio.loadmat(fullpath)

        images = saved_boxes['images']
        image_boxes = saved_boxes['boxes'][0]
        filenames_dict = {v: k for k, v in enumerate(self.filenames)}

        for image, boxes in zip(images, image_boxes):
            # Find the fullpath for the image
            filename = image[0][0] + '.jpg'
            # boxes are saved as y1, x1, y2, x2
            boxes = boxes[:, (1, 0, 3, 2)] - 1

            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep]

            keep = ds_utils.filter_small_boxes(
                boxes,
                min_size=self.min_prop_box_side)
            boxes = boxes[keep]

            # Find the index of this image
            sample_index = filenames_dict[filename]

            resize_ratio = self.samples[sample_index]['resize_ratio']
            boxes = boxes * resize_ratio

            self.samples[sample_index][
                'roi_proposals_xyxy'] = np.array(boxes)
            self.samples[sample_index][
                'roi_proposals_xywh'] = x1y1x2y2_to_xywh(np.array(boxes))

    def get_overlaps(self):
        for i, sample in enumerate(self.samples):
            boxes = sample['roi_proposals_xyxy']
            num_boxes = boxes.shape[0]
            overlaps = np.zeros(
                (num_boxes, self.num_classes), dtype=np.float32)
            gt_boxes = sample['gt_boxes_xyxy']
            if len(gt_boxes) > 0:
                gt_xyxy = gt_boxes[:, :4]
                gt_classes = gt_boxes[:, 4].astype(int)
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_xyxy.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            # max overlap with gt over classes (columns)
            max_overlaps = overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = overlaps.argmax(axis=1)
            self.samples[i]['max_classes'] = max_classes
            self.samples[i]['max_overlaps'] = max_overlaps

            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)

            overlaps = scipy.sparse.csr_matrix(overlaps)
            self.samples[i]['gt_overlaps'] = overlaps

    def flip_samples(self):
        num_images = len(self.samples)
        for index in range(num_images):
            new_sample = copy.deepcopy(self.samples[index])
            # Flip gt boxes
            new_sample['gt_boxes_xyxy'] = flip_boxes(
                new_sample['gt_boxes_xyxy'], new_sample['width'])
            new_sample['gt_boxes_xywh'] = x1y1x2y2_to_xywh(
                new_sample['gt_boxes_xyxy'])
            # Flip ROIs boxes
            new_sample['roi_proposals_xyxy'] = flip_boxes(
                new_sample['roi_proposals_xyxy'], new_sample['width'])
            new_sample['roi_proposals_xywh'] = x1y1x2y2_to_xywh(
                new_sample['roi_proposals_xyxy'])

            new_sample['flipped'] = True

            self.samples.append(new_sample)

    def get_image_sizes(self, min_size=600, max_size=1000):
        ''' Load all image sizes.'''
        # Read from the dataset and cache
        for sample_index, sample in enumerate(self.samples):
            width, height = PIL.Image.open(sample['filepath']).size
            new_size, resize_ratio = get_new_size((width, height))
            self.samples[sample_index]['width'] = new_size[0]
            self.samples[sample_index]['height'] = new_size[1]
            self.samples[sample_index]['resize_ratio'] = resize_ratio
    
    def __getitem__(self, index):
        batch_rois=64
        sample = self.samples[index]
        # get image
        image = self.get_image(sample)

        image_arr = np.asarray(image, dtype='uint8')
        image_normalized = normalize_images([image])

        if self.mode is 'train':
            # get fg & bg rois and targets
            # how many of the rois should be Foreground rois
            FG_FRACTION = 0.25
            rois_per_image = batch_rois
            fg_rois_per_image = np.round(FG_FRACTION * rois_per_image)
            (labels, overlaps, im_rois, bbox_targets,
             bbox_inside_weights) = _sample_rois(sample,
                                                 fg_rois_per_image,
                                                 rois_per_image,
                                                 self.num_classes)
            # shuffle rois, targets and labels
            shuffle_indexes = np.random.permutation(im_rois.shape[0])
            im_rois = im_rois[shuffle_indexes]
            bbox_targets = bbox_targets[shuffle_indexes]
            labels = labels[shuffle_indexes]

            # Roi pooling needs the index of the image in the batch
            # to be the first param of a ROI.
            zeros_column = np.zeros((im_rois.shape[0], 1))
            im_rois = np.concatenate([zeros_column, im_rois], axis=1)
            # Get one target for each roi
            targets = bbox_targets.reshape((-1, self.num_classes, 4)
                                           )[range(len(im_rois)), labels, :]
            targets = torch.Tensor(targets)
            labels = torch.FloatTensor(labels.astype(float))
            # Combine the targets and the classes
            targets_all = torch.cat([targets, labels.float()], dim=1) 

            im_rois = im_rois.astype(float)
            return (index, image_normalized, im_rois, targets_all)

        elif self.mode is 'test':
            # Roi pooling needs the index of the image in the batch
            # to be the first param of a ROI.
            im_rois = sample['roi_proposals_xyxy']
            zeros_column = np.zeros((im_rois.shape[0], 1))
            im_rois = np.concatenate([zeros_column, im_rois], axis=1)
            im_rois = im_rois.astype(float)
            return (index, image_normalized, im_rois)

        return (index, image_normalized, im_rois, targets_all)
    
    @staticmethod
    def get_image(sample):
        image = PIL.Image.open(sample['filepath'])
        try:
            if sample['flipped']:
                image = PIL.ImageOps.mirror(image)
        except KeyError:
            pass
        new_size = (np.array(image.size) * sample['resize_ratio']
                   ).astype(int)
        return image.resize(new_size)
