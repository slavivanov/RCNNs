import numpy as np
import PIL
import torch
from torch.autograd import Variable
from bbox_transform import bbox_transform_inv, clip_boxes
from nms.nms_wrapper import nms
from pascalroidb import PascalRoiDB
from data_utils import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2
from display_utils import display_image_regions
import tqdm


class FastRCNNTest(object):
    def __init__(self, model=None, targets_mean=None, targets_std=None,
                 top_class_only=False,
                 class_detection_thresh=0.05,
                 nms_thresh=0.3,
                 num_classes=21,
                 show_gt_boxes=True,
                 class_id_to_name=None,
                 data_loader=None):
        """
        Args:
            model: The model which will evaluate the data.
            targets_mean: The mean of all targets.
            targets_std: The std of all targets.
            top_class_only: Whether to use only top class for each roi,
                or any class over a certain threshhold (class_detection_thresh).
            class_detection_thresh: If the softmax for a class is
                above class_detection_thresh, it's considered detected
                in the roi.
            nms_thresh: Combine same class boxes with IOU over this threshold.
            num_classes: The number of categories, including background.
            show_gt_boxes: Display ground truth boxes when visualizing
                detections.
            data_loader: A torch dataloader, over the samples. Default None to
                manually feed samples.
        """
        self.model = model
        self._targets_mean = targets_mean
        self._targets_std = targets_std
        self.top_class_only = top_class_only
        self.class_detection_thresh = class_detection_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        self.show_gt_boxes = show_gt_boxes
        self.class_id_to_name = class_id_to_name
        if data_loader:
            self.data_loader = data_loader
            self.reset_iterator()
        else:
            self.data_loader = None
            self.data_iterator = None

    def reset_iterator(self):
        """ Resets the iterator over the samples. """
        if self.data_loader:
            self.data_iterator = self.data_loader.__iter__()
        else:
            self.data_iterator = None


    def test_single(self, image, rois, image_size, image_resize_ratio):
        """ Test a single image on the net.
        Args:
            image: A preprocessed image or precomputed features of
                the image. As ndarray.
            rois: Rois sized for the the image.
                Ndarray: (image_index, x1, y1, x2, y2)
            image_size: The original image size.
            image_resize_ratio: What is the ratio that this image was resized on.
        """
        rois_np, dedup_inv_index = self.dedup_boxes(rois.numpy())
        image_var = Variable(image.cuda(), volatile=True)
        rois_var = Variable(torch.Tensor(rois_np).cuda(), volatile=True)

        # Run the img through the network
        out = self.model(image_var, rois_var)
        # predicted deltas
        deltas = out[1].data.cpu().numpy()
        deltas = self.unnormalize_deltas(deltas, self._targets_mean,
                                         self._targets_std)

        # transform rois using predicted deltas
        boxes = rois_np[:, 1:] / image_resize_ratio
        bboxes_inv_transformed = bbox_transform_inv(boxes, deltas)

        class_probas, class_indexes = torch.max(out[0], 1)
        indexes_np = np.squeeze(class_indexes.data.cpu().numpy())
        #     print('Total FG RoIs Detected: ', np.sum(indexes_np > 0))

        scores = out[0].data.cpu().numpy()
        scores = np.exp(scores)

        # clip rois to image size
        bboxes_inv_transformed = clip_boxes(bboxes_inv_transformed,
                                            image_size)

        scores = scores[dedup_inv_index, :]
        bboxes_inv_transformed = bboxes_inv_transformed[dedup_inv_index, :]

        # Non-maximum supression of similar boxes
        all_boxes = self._nms_boxes(bboxes_inv_transformed, scores)

        return all_boxes

    def test_next(self, display_detections=False):
        """ Test the next sample, and display the detections if specified.
        Returns:
            The boxes found in this sample.
        """
        batch = next(self.data_iterator)
        sample_index, image_size, image_resize_ratio, images, rois = batch
        sample_index = sample_index.cpu().numpy()[0]
        image_resize_ratio = image_resize_ratio.cpu().numpy()[0]
        rois = rois[0]
        image_size = image_size.cpu().numpy()[0]

        all_boxes = self.test_single(images, rois, image_size,
                                     image_resize_ratio)

        if display_detections:
            display_boxes, display_classes = self.get_display_boxes(all_boxes)
            sample = self.data_loader.dataset.samples[sample_index]
            if len(display_boxes) > 0:
                self.display_detections(
                    display_boxes,
                    display_classes,
                    sample)

        return all_boxes

    def test_all(self, use_tqdm=True):
        """ Test all samples.
        Args:
            use_tqdm: Whether to show tqdm progress bar.
        """

        n_batches = len(self.data_loader.dataset.samples)

        all_boxes = [[[] for _ in range(n_batches)]
                     for _ in range(self.data_loader.dataset.num_classes)]
        if use_tqdm:
            batch_iterator = tqdm.tqdm_notebook(range(n_batches))
        else:
            batch_iterator = range(n_batches)

        self.reset_iterator()
        for image_index in batch_iterator:
            image_boxes = self.test_next()

            for class_id, class_boxes in enumerate(image_boxes):
                all_boxes[class_id][image_index] = class_boxes

        return all_boxes

    def _nms_boxes(self, boxes, scores):
        """ Perform non-maximum supression of similar boxes/detections.
        Args:
            boxes: Rois for this image. Array (num_rois, num_classes * 4).
            scores: Class probabilities for each roi.
                Array (num_rois, num_classes).
        Returns:
            A list of NMSed class detections for this image.
        """
        all_boxes = [[] for _ in range(self.num_classes)]
        # skip j = 0, because it's the background class
        for class_id in range(1, self.num_classes):
            # Whether to use only the top class for each box or
            # all classes over a certain threshhold.
            if self.top_class_only:
                detection_criterion = (np.argmax(scores, axis=1) == class_id)
            else:
                detection_criterion = (
                    scores[:, class_id] > self.class_detection_thresh)
            class_detected_indexes = np.where(detection_criterion)[0]

            cls_scores = scores[class_detected_indexes, class_id]
            class_box_start = class_id * 4
            class_box_end = class_box_start + 4
            cls_boxes = boxes[class_detected_indexes,
                              class_box_start:class_box_end]

            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])
                                 ).astype(np.float32, copy=False)

            if len(cls_dets) > 1:
                keep = nms(cls_dets, self.nms_thresh, force_cpu=True)
                cls_dets = cls_dets[keep, :]
            all_boxes[class_id] = cls_dets
        return all_boxes

    def display_detections(self, rois, classes, sample):
        """ Display detected foreground rois for a sample.
        Args:
            rois: Detected RoIs as ndarray of (x, y, x2, y2)
            classes: Class labels of each RoI as ndarray of class_id.
            sample: A dict describing the sample that produced
                the rois and classes.
        """
        detected_roi = np.append(rois,
                                 classes[:, None],
                                 axis=1)
        if self.show_gt_boxes:
            gt_boxes = sample['boxes'][sample['gt_classes'] != 0]
            gt_boxes = x1y1x2y2_to_xywh(gt_boxes)
        else:
            gt_boxes = None

        resize_image = False
        if resize_image:
            image = PascalRoiDB.get_image(sample)
        else:
            image = PIL.Image.open(sample['image'])

        display_image_regions(image,
                              x1y1x2y2_to_xywh(detected_roi),
                              gt_boxes,
                              class_id_to_name=self.class_id_to_name)

    def display_precomputed_boxes(self, sample_index, all_boxes):
        """ Display the precomputed boxes for an image. 
        Args:
            sample_index: The index of the image for which to 
                display the detections.
            all_boxes: A list of detected rois for each class.
        """
        image_rois = [class_detections[sample_index]
                  for class_detections in all_boxes]

        image_rois_list = []
        image_classes = []
        for class_index, class_rois in enumerate(image_rois):
            if len(class_rois) > 0:
                classes = np.ones((class_rois.shape[0])) * class_index
                image_rois_list.extend(class_rois)
                image_classes.extend(classes)
        image_rois_list = np.array(image_rois_list)
        image_classes = np.array(image_classes)

        show_gt_boxes = False
        self.display_detections(image_rois_list, image_classes, 
                                  self.data_loader.dataset.samples[sample_index])
        
    @staticmethod
    def get_display_boxes(all_boxes):
        """ Transform all_boxes from for a single image to a list of
            boxes and the categories detected in them.
        Args:
            all_boxes: A list of detections for each class in an image.
                Each detection is (x1, y1, x2, y2, probability)
        Returns:
            Array of boxes coordinates.
            Array of the detected class for each box."""
        display_boxes = []
        display_classes = []
        for class_id, class_boxes in enumerate(all_boxes):
            for box in class_boxes:
                display_boxes.append(box)
                display_classes.append(class_id)
        return np.asarray(display_boxes), np.asarray(display_classes)


    @staticmethod
    def unnormalize_deltas(deltas, mean, std):
        """ Unnormalize deltas using mean and std.
        Args:
            deltas: A ndarray with size (batch_size, num_fg_classes * 4)
            mean: The mean of the normalized targets
            std: The std of the normalized targets
        Returns:
            The unnormalized deltas.
        """
        return deltas * std + mean

    @staticmethod
    def dedup_boxes(boxes):
        """Identify duplicate feature ROIs, so we only compute features
            on the unique subset.
        Args:
            boxes: array of (image_index, x1, y1, x2, y2)
        Returns:
        A tuple of (
            Deduplicated boxes, array of (image_index, x1, y1, x2, y2),
            Indexes to invert the deduplicated boxes to boxes with duplicates)
        """
        # When mapping from image ROIs to feature map ROIs, there's some aliasing
        # (some distinct image ROIs get mapped to the same feature ROI).
        spatial_scale = 1. / 16
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(boxes * spatial_scale).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        boxes = boxes[index, :]
        return boxes, inv_index
