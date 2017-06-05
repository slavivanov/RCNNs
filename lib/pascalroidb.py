import os
import PIL
import numpy as np
import datasets.pascal_voc as pascal_voc
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.minibatch import get_minibatch
from utils.config import cfg
from dataset import Dataset
from data_utils import normalize_images, get_new_size


class PascalRoiDB(Dataset):
    """ Representation of the Pascal VOC dataset."""
    def __init__(self, data_path=None, set_name=None, year=None,
                 mode=None, shuffle_samples=None):
        """
        Args:
            data_path: Location of the VOC Devkit. Default is ../VOC/VOCdevkit/
            set_name: Part of the dataset to use. E.g: train, trainval, test.
                Default: train.
            year: Dataset from which year are we using. E.g: 2007, 2012. String
                Default: 2007
            mode: Are we using the dataset to train or test. String.
                Default: train
            shuffle_samples: Whether to shuffle the samples in the dataset.
                Default: False.
        """
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

        if mode:
            self.mode = mode
        else:
            self.mode = 'train'

        if shuffle_samples:
            self.shuffle_samples = shuffle_samples
        else:
            self.shuffle_samples = False

        cache_dir = os.path.join('intermediate/voc_roidb/',
                                 self.year, self.set_name)
        super().__init__(cache_dir)

        self.sample_index = 0
        self.get_samples()

    def get_samples(self):
        """ Get samples from the dataset files. Use cache if avalable. """
        self.num_classes = 21
        try: 
            self._sample_indexes = self.load_cache(
                'sample_indexes_' + str(self.shuffle_samples))
            self.bbox_means = self.load_cache('bbox_means')
            self.bbox_stds = self.load_cache('bbox_stds')   
            self.samples = self.load_cache('samples')
            self.imdb = self.load_cache('imdb')
            print('Loaded from cache.')
        except FileNotFoundError:
            print('No cache exists, loading from devkit')
            imdb = pascal_voc.pascal_voc(self.set_name, self.year,
                                         self.data_path)
            cfg.TRAIN.PROPOSAL_METHOD = 'selective_search'
            imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
            roidb = self.get_training_roidb(imdb)
            self.save_cache('imdb', imdb, use_pickle=True)

            roidb = self.filter_roidb(roidb)
            (self.bbox_means,
             self.bbox_stds) = rdl_roidb.add_bbox_regression_targets(roidb)
            self.save_cache('bbox_means', self.bbox_means)
            self.save_cache('bbox_stds', self.bbox_stds)

            self.samples = roidb
            self.save_cache('samples', self.samples, use_pickle=True)

            self._shuffle_roidb_inds()
            print('done')
        
    def __len__(self):
        """ How many samples are in the dataset. """
        return len(self.samples)

    @staticmethod
    def get_image(sample):
        """ Load and resize an sample image.
        Args:
            sample: A single sample to load.
        Returns:
            Resized PIL.Image.
        """
        image = PIL.Image.open(sample['image'])
        try:
            if sample['flipped']:
                image = PIL.ImageOps.mirror(image)
        except KeyError:
            pass
        new_size, _ = get_new_size(image.size)
        return image.resize(new_size)

    def get_training_roidb(self, imdb):
        """Returns a roidb (Region of Interest database) for use in training.
        Args:
            imdb: A sequence of samples.
        Returns:
            The roidb for the given samples.
        """
        if cfg.TRAIN.USE_FLIPPED and self.mode is not 'test':
            print('Appending horizontally-flipped training examples...')
            imdb.append_flipped_images()
            print('done')

        print('Preparing training data...')
        rdl_roidb.prepare_roidb(imdb)
        print('done')

        return imdb.roidb

    @staticmethod
    def filter_roidb(roidb):
        """Remove roidb entries that have no usable RoIs.
        Returns:
            filtered_roidb: Roidb with only entries that have rois.
        """
        def is_valid(entry):
            # Valid images have:
            #   (1) At least one foreground RoI OR
            #   (2) At least one background RoI
            overlaps = entry['max_overlaps']
            # find boxes with sufficient overlap
            fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                               (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
            # image is only valid if such boxes exist
            valid = len(fg_inds) > 0 or len(bg_inds) > 0
            return valid

        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after))
        return filtered_roidb

    @staticmethod
    def _get_rois_blob(im_rois, im_scale_factors):
        """Converts RoIs into network inputs.

        Arguments:
            im_rois (ndarray): R x 4 matrix of RoIs in original image
                coordinates.
            im_scale_factors (list): scale factors as returned by
                _get_image_blob.

        Returns:
            blob (ndarray): R x 5 matrix of RoIs in the image pyramid
        """
        rois, levels = PascalRoiDB._project_im_rois(im_rois, im_scale_factors)
        rois_blob = np.hstack((levels, rois))
        return rois_blob.astype(np.float32, copy=False)

    @staticmethod
    def _project_im_rois(im_rois, scales):
        """Project image RoIs into the image pyramid built by _get_image_blob.

        Arguments:
            im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
            scales (list): scale factors as returned by _get_image_blob

        Returns:
            rois (ndarray): R x 4 matrix of projected RoI coordinates
            levels (list): image pyramid levels used by each projected RoI
        """
        im_rois = im_rois.astype(np.float, copy=False)

        if len(scales) > 1:
            widths = im_rois[:, 2] - im_rois[:, 0] + 1
            heights = im_rois[:, 3] - im_rois[:, 1] + 1

            areas = widths * heights
            scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
            diff_areas = np.abs(scaled_areas - 224 * 224)
            levels = diff_areas.argmin(axis=1)[:, np.newaxis]
        else:
            levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

        # rois = im_rois * scales[levels]
        rois = im_rois * scales

        return rois, levels

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb. Creates
        shuffled _sample_indexes."""
        if self.shuffle_samples:
            if cfg.TRAIN.ASPECT_GROUPING:
                widths = np.array([r['width'] for r in self.samples])
                heights = np.array([r['height'] for r in self.samples])
                horz = (widths >= heights)
                vert = np.logical_not(horz)
                horz_inds = np.where(horz)[0]
                vert_inds = np.where(vert)[0]
                inds = np.hstack((
                    np.random.permutation(horz_inds),
                    np.random.permutation(vert_inds)))
                inds = np.reshape(inds, (-1, 2))
                row_perm = np.random.permutation(np.arange(inds.shape[0]))
                inds = np.reshape(inds[row_perm, :], (-1,))
                self._sample_indexes = inds
            else:
                self._sample_indexes = np.random.permutation(
                    np.arange(len(self.samples)))
        else:
            self._sample_indexes = np.arange(len(self.samples))
            
        self.save_cache('sample_indexes_' + str(self.shuffle_samples), 
                        self._sample_indexes)
           
    def __getitem__(self, index):
        """ Retrieves a batch by index.
        Args:
            index: The index of the batch to get.
        Returns:
            batch: Sequence of Torch tensors and ndarrays.
        """
        if self.mode == 'train':
            return self.train_batch(index)
        elif self.mode == 'test':
            return self.test_batch(index)
        else:
            raise ValueError('Unknown mode: {:s}'.format(self.mode))
            
    def test_batch(self, index):
        """ Retrieves a test-mode batch.
        Args:
            index: The index of the batch to get.
        Returns:
            index (int): The index of the retrived batch. (Useful when getting
                batches via next())
            original_size_h_w (2 element ndarray): The original height and width
                of the image in the batch.
            resize_factor (float): How was the original size changed to get the
                tensor in 'image'.
            image (Torch Tensor): Normalized image data, with dims:
                (batch_index, C, H, W).
            rois (Torch Tensor): All regions of interest for this image,
                rescaled to the 'image' size. With dims:
                (rois_index, x1, y2, x2, y2).
        """
        batch_index = self._sample_indexes[index]
        sample = self.samples[batch_index]

        image_pil = self.get_image(sample)
        image_arr = np.asarray(image_pil, dtype='uint8')
        image = normalize_images([image_arr])

        original_size = (sample['width'], sample['height'])
        _, resize_factor = get_new_size(original_size)
        original_size_h_w = np.array((sample['height'], sample['width']))
        
        # The roidb may contain ground-truth rois (for example, if the roidb
        # comes from the training or val split). We only want to evaluate
        # detection on the *non*-ground-truth rois. We select those the rois
        # that have the gt_classes field set to 0, which means there's no
        # ground truth.
        box_proposals = sample['boxes'][sample['gt_classes'] == 0]
        
        rois = self._get_rois_blob(box_proposals, [resize_factor])
        

        return index, original_size_h_w, resize_factor, image, rois
            
    def train_batch(self, index):
        """ Retrieves a train-mode batch.
       Args:
           index: The index of the batch to get.
       Returns:
           index (int): The index of the retrived batch. (Useful when getting
               batches via next())
           image (Torch Tensor): Normalized image data, with dims:
               (batch_index, C, H, W).
           im_rois (Torch Tensor): Regions of interest for this image, rescaled
                to the 'image' size. With dims: (rois_index, x1, y2, x2, y2).
           bbox_targets (Torch Tensor): The targets for each RoI. With dims:
                (roi_index, num_classes * 4).
           labels (Torch Tensor): The GT label for each ROI. With dims:
                (roi_index, 1).
           bbox_inside_weights (Torch Tensor): Which part of 'bbox_targets'
                applies to the roi. Each row is all 0 except the 4 targets that
                apply to the roi. With dims: (roi_index, num_classes * 4).
       """
        batch_index = self._sample_indexes[index]
        sample = self.samples[batch_index]
        batch = get_minibatch([sample], self.num_classes)

        # shuffle rois, targets and labels
        shuffle_indexes = np.random.permutation(batch['rois'].shape[0])
        im_rois = batch['rois'][shuffle_indexes].astype(float)
        bbox_targets = batch['bbox_targets'][shuffle_indexes]
        bbox_inside_weights = batch['bbox_inside_weights'][shuffle_indexes]
        labels = batch['labels'][shuffle_indexes].astype(float)

        # This was for the other vgg
        # image = batch['data']
        image_pil = self.get_image(sample)
        image_arr = np.asarray(image_pil, dtype='uint8')
        image = normalize_images([image_arr])

        return (index, image, im_rois, bbox_targets, labels, bbox_inside_weights)