import torch

import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import flatten
from roi_pooling.modules.roi_pool import RoIPool as RoIPool_GPU
from vgg import get_vgg_conv


class FastRCNNModel(torch.nn.Module):
    def __init__(self, cnn_weights=None, fc1_weights=None, fc2_weights=None,
                 roi_size=(7, 7, 512), dropout_p=0.5, num_categories=21):
        super(FastRCNNModel, self).__init__()
        self.dropout_p = dropout_p

        self.cnn = get_vgg_conv(skip_last_layers=1, pretrained=False)
        if cnn_weights:
            self.cnn.load_state_dict(cnn_weights)
        # freeze the bottommost cnn layers
        for index, param in enumerate(self.cnn.parameters()):
            if index <= 7:
                param.requires_grad = False

        self.roi_pooling = RoIPool_GPU(roi_size[0], roi_size[1],
                                       spatial_scale=1. / 16)

        input_size = 1
        for dim in roi_size:
            input_size *= dim
        self.fc1 = torch.nn.Linear(input_size, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        if fc1_weights:
            self.fc1.load_state_dict(fc1_weights)
        if fc2_weights:
            self.fc2.load_state_dict(fc2_weights)

        self.classifier = torch.nn.Linear(4096, num_categories)
        self.regressor = torch.nn.Linear(4096, num_categories * 4)

        self.weights_init()

    def forward(self, image, regions):
        cnn_features = self.cnn(image)
        rois = self.roi_pooling.forward(cnn_features, regions)
        x = flatten(rois)

        # FC1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # FC2
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Outputs
        class_probabilities = F.log_softmax(self.classifier(x))
        class_box_regression = self.regressor(x)

        return class_probabilities, class_box_regression

    def weights_init(self):
        for m in self.named_modules():
            name = m[0]
            module = m[1]
            class_name = module.__class__.__name__

            if name == 'classifier':
                module.weight.data.normal_(0.0, 0.01)
                module.bias.data.fill_(0.)
            elif name == 'regressor':
                module.weight.data.normal_(0.0, 0.001)
                module.bias.data.fill_(0.)
            


