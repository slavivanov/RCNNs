import pascal_voc_reader
from utils import *

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225])

class RCNN_All_Data(object):        
    def __init__(self, image_size):
        self.im_size = image_size
        self.load_pascal_dataset()
        self.load_indexes()
        # The following calls depend on load_indexes being called first
        self.load_samples()
        self.load_features()
        self.load_gt_boxes()
        self.load_proposals_rois() 

    def unload_set(self, set_name='valid'):
        ''' Unload train, test or valid set from memory. '''
        if set_name is 'train':
            del self.train_samples
            del self.train_features
            del self.train_gt_boxes
            del self.train_positive_rois
            del self.train_positive_rois_targets
            del self.train_negative_rois
            del self.train_negative_rois_targets
        elif set_name is 'valid':
            del self.valid_samples
            del self.valid_features
            del self.valid_gt_boxes
            del self.valid_positive_rois
            del self.valid_positive_rois_targets
            del self.valid_negative_rois
            del self.valid_negative_rois_targets
        elif set_name is 'train':
            pass
        
    def load_pascal_dataset(self):
        data_path = './data/'

        (self.samples, self.classes_count,
         self.class_mapping) = pascal_voc_reader.get_data(data_path)
        self.add_background_class()
        self.class_id_to_name = {v: k for k, v
                                 in self.class_mapping.items()}

        self.filenames = [sample['filepath']
                          for sample in self.samples]

        self.num_classes = len(self.classes_count)

    def add_background_class(self):
        if 'bg' not in self.classes_count:
            self.classes_count['bg'] = 0
            # Add the background class with index 0
            self.class_mapping = {k: v + 1 for k, v
                                  in self.class_mapping.items()}
            self.class_mapping['bg'] = 0

    def load_indexes(self):
        filename = ('intermediate/voc/' +
                    'voc_enough_samples_mask_{}x{}.bcolz').format(
            self.im_size[0], self.im_size[1])
        self.enough_samples_mask = load_array(filename)

        self.valid_indexes = load_array(
            'intermediate/voc/voc_valid_indexes.bcolz')
        self.train_indexes = load_array(
            'intermediate/voc/voc_train_indexes.bcolz')

    def load_samples(self):
        filename = 'intermediate/voc/voc_images_{}x{}.bcolz'.format(
            self.im_size[0], self.im_size[1])
        all_samples = load_array(filename)

        all_samples = all_samples[self.enough_samples_mask]
        self.train_samples = all_samples[self.train_indexes]
        self.valid_samples = all_samples[self.valid_indexes]
        del all_samples

    def load_gt_boxes(self):
        filename = 'intermediate/voc/voc_gt_boxes_{}x{}.bcolz'.format(
            self.im_size[0], self.im_size[1])
        truth_boxes = load_array(filename)

        truth_boxes = truth_boxes[self.enough_samples_mask]
        self.train_gt_boxes = truth_boxes[self.train_indexes]
        self.valid_gt_boxes = truth_boxes[self.valid_indexes]

    def load_features(self):
        filename = ('intermediate/voc/' + 
            'voc_precomputed_features_{}x{}_all.bcolz').format(
            self.im_size[0], self.im_size[1])
        self.train_features = load_array(filename)

        filename = ('intermediate/voc/' + 
                    'voc_precomputed_features_valid_{}x{}.bcolz').format(
            self.im_size[0], self.im_size[1])
        self.valid_features = load_array(filename)
        
    def load_proposals_rois(self):
#         filename = 'intermediate/voc/voc_rois_{}x{}_scale_750.bcolz'.format(
#             self.im_size[0], self.im_size[1])
#         all_regions = load_array(filename)

#         all_regions = all_regions[self.enough_samples_mask]
#         self.train_rois = np.asarray(all_regions)[self.train_indexes]
#         self.valid_rois = np.asarray(all_regions)[self.valid_indexes]
        self.load_positive_rois()
        self.load_negative_rois()
        self.load_targets_mean_std()

    def load_positive_rois(self):
        # load positive rois and targets
        filename = ('intermediate/voc/' +
                    'voc_positive_rois_{}x{}.bcolz').format(
            self.im_size[0], self.im_size[1])
        all_positive_regions=load_array(filename)

        all_positive_regions = all_positive_regions[self.enough_samples_mask]
        self.train_positive_rois=all_positive_regions[self.train_indexes]
        self.valid_positive_rois=all_positive_regions[self.valid_indexes]

        filename=('intermediate/voc/' +
                    'voc_positive_rois_targets_normalized_{}x{}.bcolz'
                   ).format(self.im_size[0], self.im_size[1])
        all_positive_regions_targets=load_array(filename)

        all_positive_regions_targets=(
            all_positive_regions_targets[self.enough_samples_mask])
        self.train_positive_rois_targets=(
            all_positive_regions_targets[self.train_indexes])
        self.valid_positive_rois_targets=(
            all_positive_regions_targets[self.valid_indexes])

    def load_negative_rois(self):
        # load negative rois and targets
        filename = ('intermediate/voc/' +
                    'voc_negative_rois_{}x{}.bcolz').format(
            self.im_size[0], self.im_size[1])
        all_negative_regions = load_array(filename)

        filename = ('intermediate/voc/' +
                    'voc_negative_rois_targets_{}x{}.bcolz').format(
            self.im_size[0], self.im_size[1])
        all_negative_regions_targets = load_array(filename)

        all_negative_regions = all_negative_regions[self.enough_samples_mask]
        self.train_negative_rois = all_negative_regions[self.train_indexes]
        self.valid_negative_rois = all_negative_regions[self.valid_indexes]

        all_negative_regions_targets = (
            all_negative_regions_targets[self.enough_samples_mask])
        self.train_negative_rois_targets = (
            all_negative_regions_targets[self.train_indexes])
        self.valid_negative_rois_targets = (
            all_negative_regions_targets[self.valid_indexes])

    def load_targets_mean_std(self):
        filename = 'intermediate/voc/roi_targets_mean_{}x{}.bcolz'.format(
            self.im_size[0], self.im_size[1])
        self.targets_mean = load_array(filename)

        filename = 'intermediate/voc/roi_targets_std_{}x{}.bcolz'.format(
            self.im_size[0], self.im_size[1])
        self.targets_std = load_array(filename)

        
class RCNN_Set(object):
    def __init__(self, data, set_name='train'):
        super().__init__()
        self.index = 0
        # Create a referrence in this instance 
        # to all attributes of the dataset.
        # This can be better done with singleton.
        for k,v in data.__dict__.items():
            setattr(self, k, v)

        if set_name is 'train':
            self.setup_train()
        elif set_name is 'valid':
            self.setup_valid()
        elif set_name is 'test':
            pass
        self.sample_count = len(self.images)


    def setup_train(self):
        self.images = self.train_samples
        self.features = self.train_features
        self.gt_boxes = self.train_gt_boxes
        self.positive_rois = self.train_positive_rois
        self.positive_targets = self.train_positive_rois_targets
        self.negative_rois = self.train_negative_rois
        self.negative_targets = self.train_negative_rois_targets

    def setup_valid(self):
        self.images = self.valid_samples
        self.features = self.valid_features
        self.gt_boxes = self.valid_gt_boxes
        self.positive_rois = self.valid_positive_rois
        self.positive_targets = self.valid_positive_rois_targets
        self.negative_rois = self.valid_negative_rois
        self.negative_targets = self.valid_negative_rois_targets

    def reset_index(self, source='train'):
        self.index = 0

    def next_batch(self,
                   roi_batch_size=128,
                   images_per_batch=2,
                   use_features=False,
                   images_only=False,
                   loop_over=True):
        ''' Get a batch of training samples. 
        Args:
            source: Which dataset to get a batch from: train, valid or test.
            roi_batch_size: Final batch size of ROIs.
            images_per_batch: How many images to comprise in each batch. Note that 
                the number of ROIs per image is = roi_batch_size / images_per_batch.
            features: Instead of the images supplies the computed CNN features.
            images_only: Whether to retrieve only images or ROIs per image.
            loop_over: Whether to start over at the end of the dataset.
        Returns:
            A tuple of (batch_images, batch_rois, batch_rois_targets).
        '''
        # Where we would stop
        end_index = self.index + images_per_batch
        # If we are at the end of the dataset
        if end_index > self.sample_count:
            # Are we starting from the beginning
            if loop_over:
                self.index = 0
                end_index = self.index + images_per_batch
            else:
                # Just get the last samples
                end_index = self.sample_count

        if use_features:
            # Instead of images we use the image CNN features
            batch_images = self.features[self.index:end_index]
        else:
            # Get images
            batch_images = self.images[self.index:end_index]
            # Normalize and convert to pytorch tensors
            batch_images_normalized = self.normalize_images(batch_images)

            if len(batch_images_normalized) == 0:
                batch_images = torch.FloatTensor([])
            else:
                batch_images = torch.stack(batch_images_normalized)

            # If we only need the images
            if images_only:
                # move index for the next batch
                self.index += images_per_batch
                return batch_images

        batch_rois, batch_targets = self.get_batch_rois(
            end_index, roi_batch_size, images_per_batch)

        # finally, move index for the next batch
        self.index = end_index

        return (batch_images, batch_rois, batch_targets)

    def get_batch_rois(self, end_index, roi_batch_size, images_per_batch):
    # How many rois we need per image
        num_rois_per_image = roi_batch_size // images_per_batch
        num_positive_rois = num_rois_per_image // 4
        num_negative_rois = num_rois_per_image - num_positive_rois

        batch_rois = []
        batch_rois_targets = []
        # get the rois for each image
        batch_index = 0
        for image_index in range(self.index, end_index):
            # the positive rois and their targets
            # that we are going to use for this sample
            image_rois, image_targets = self.get_rois_type(
                'positive', image_index, num_positive_rois)

            negative_rois, negative_targets = self.get_rois_type(
                'negative', image_index, num_negative_rois)

            # combine the negative and the positive rois
            # and targets for this image
            image_rois = np.concatenate([image_rois, negative_rois])
            image_targets = np.concatenate([image_targets, negative_targets])

            # Add the image index to the rois
            batch_index_arr = batch_index * np.ones((image_rois.shape[0], 1))
            image_rois = np.hstack((batch_index_arr, image_rois))

            # shuffle the rois and targets for this image
            shuffle_mask = np.random.permutation(len(image_rois))
            image_rois = image_rois[shuffle_mask]
            image_targets = image_targets[shuffle_mask]

            # add them to the batch rois
            batch_rois.extend(image_rois)
            batch_rois_targets.extend(image_targets)

        return (np.array(batch_rois), np.array(batch_rois_targets))

    def get_rois_type(self, rois_type,
                      image_index,
                      rois_num_limit):
        if rois_type is 'positive':
            rois_pool = self.positive_rois
            targets_pool = self.positive_targets
        elif rois_type is 'negative':
            rois_pool = self.negative_rois
            targets_pool = self.negative_targets
        # All positive rois for this image
        image_rois = self.transform_regions_bottom_right(rois_pool[image_index])
        image_rois_targets = targets_pool[image_index]
        # shuffle rois
        shuffle_mask = np.random.permutation(len(image_rois))

        final_rois = np.asarray(
            image_rois)[shuffle_mask][:rois_num_limit]
        final_targets = np.asarray(
            image_rois_targets)[shuffle_mask][:rois_num_limit]
        return (final_rois, final_targets)

    def normalize_images(self, batch_images):
        # Normalize the images.
        normalize = torchvision.transforms.Normalize(mean=IMAGENET_MEAN,
                                                     std=IMAGENET_STD)
        images_normalized = []
        for image in batch_images:
            image_normalized = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                normalize])(image)
            images_normalized.append(image_normalized)

        return images_normalized

    @staticmethod
    def transform_regions_bottom_right(regions):
        ''' From (x, y, w, h) to (x1, y1, x2, y2).'''
        regions_transformed = np.zeros(np.array(regions).shape)
        for index, region in enumerate(regions):
            x1, y1, w, h = region[:4]
            w = max(w, 1)
            h = max(h, 1)
            x2 = x1 + w - 1
            y2 = y1 + h - 1

            extra_data = list(region[4:])
            regions_transformed[index] = np.array([x1, y1, x2, y2] + extra_data)
        return regions_transformed
    
    
    @staticmethod
    def transform_regions_width_height(regions):
        ''' From (x1, y1, x2, y2) to (x1, y1, w, h).'''
        regions_transformed = np.zeros(np.array(regions).shape)
        for index, region in enumerate(regions):
            x1, y1, x2, y2 = region[:4]
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            extra_data = list(region[4:])
            regions_transformed[index] = np.array([x1, y1, w, h] + extra_data)
        return regions_transformed