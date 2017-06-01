import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F
from roi_pooling.modules.roi_pool import RoIPool as RoIPool_GPU

def denormalize(tensor):
    ''' Denormalize a normalized for VGG image. 
    Args:
        tensor: the image as a np array (w, h, channels)
    '''
    for t, m, s in zip(tensor, IMAGENET_MEAN, IMAGENET_STD):
        t.mul_(s).add_(m)
    return tensor


def get_vgg(skip_last_layers=0):
    ''' Returns a trained and ready to go VGG 16 network.
    Args:
        skip_last_layers: How many of the last layers to not include.
    Returns:
        A torch model.
    '''
    vgg = torchvision.models.vgg16(pretrained=True)

    if skip_last_layers > 0:
        layers = [layer for layer in vgg.classifier]
        last_index = -1 * skip_last_layers
        layers = layers[:last_index]
        vgg.classifier = torch.nn.Sequential(*layers)

    return vgg


def get_vgg_conv(skip_last_layers=0, weights_file=None):
    ''' Returns a trained and ready to go features only part of the
        VGG 16 network.
    Args:
        skip_last_layers: How many of the last layers to not include.
        weights_file: if provided the weights are loaded from this file, 
            otherwise use the default torch weigths for the mode.
    Returns:
        A torch Sequential model.
    '''
    if weights_file:
        vgg = torchvision.models.vgg16(pretrained=False)
        weights = torch.load(weights_file)
        soft_load_weights(vgg, weights)
    else:
        vgg = torchvision.models.vgg16(pretrained=True)
    
    layers = [layer for layer in vgg.features]

    if skip_last_layers > 0:
        last_index = -1 * skip_last_layers
        layers = layers[:last_index]

    model = torch.nn.Sequential(*layers)
        
    return model


def soft_load_weights(model, weights):
    ''' Loads any weights that have the same name in model and weights. 
        Not very safe in general but works for this case.
    '''
    own_state = model.state_dict()
    for name, param in weights.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


class Fast_RCNN_model(torch.nn.Module):
    def __init__(self, roi_size=(7, 7, 512), dropout_p=0.5, num_categories=21):
        super(Fast_RCNN_model, self).__init__()
        self.dropout_p = dropout_p
        
        # weights_file='vgg16-00b39a1b.pth'
        vgg = get_vgg()
        self.cnn = torch.nn.Sequential(*[layer for layer in vgg.features][:-1])

        # freeze the bottommost cnn layers
        for index, param in enumerate(self.cnn.parameters()):
            if index <= 7:
                param.requires_grad = False

        self.roi_pooling = RoIPool_GPU(roi_size[0], roi_size[1],
                                   spatial_scale=1. / 16)
#         self.roi_pooling = ROIPooling(roi_size[:2])

        input_size = 1
        for dim in roi_size:
            input_size *= dim
        self.fc1 = torch.nn.Linear(input_size, 4096)
        self.fc2 = torch.nn.Linear(4096, 4096)
        self.fc1.load_state_dict(vgg.classifier[0].state_dict())
        self.fc2.load_state_dict(vgg.classifier[3].state_dict())
        
        self.classifier = torch.nn.Linear(4096, num_categories)
        self.regressor = torch.nn.Linear(4096, num_categories * 4)

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

        return class_probabilities , class_box_regression


def fast_rcnn_weights_init(model):
    for m in model.named_modules():
        name = m[0]
        module = m[1]
        class_name = module.__class__.__name__

        if name == 'classifier':
            module.weight.data.normal_(0.0, 0.01)
            module.bias.data.fill_(0.)
        elif name == 'regressor':
            module.weight.data.normal_(0.0, 0.001)
            module.bias.data.fill_(0.)
            

def np_to_var(x, is_cuda=True, dtype=torch.FloatTensor):
    var = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        var = var.cuda()
    return var


def flatten(x):
    return x.view(x.size(0), -1)


def loss_criterion(outputs, labels, targets, preds_weights):
    ''' Compute classification and regression loss'''
#     true_class = targets[:, 4].long()
    true_class = labels.long()
    classification_loss = F.nll_loss(outputs[0], true_class)

    regression_loss_computed = regression_loss(outputs[1], targets, 
                                               preds_weights)

    loss = classification_loss + regression_loss_computed
    return loss


def regression_loss(preds, targets, preds_weights):
    ''' Regression Smooth L1 loss between predicted deltas for bboxes and 
        the ground truth targets.
    '''
#     true_classes = targets[:, 4].long()
#     true_deltas = targets[:, :4]
    true_deltas = targets

#     predicted_deltas = prepare_predicted_regression(preds, true_classes)
    predicted_deltas = preds * preds_weights
    return F.smooth_l1_loss(predicted_deltas, true_deltas)
    

def prepare_predicted_regression(preds, true_classes, 
                                 num_classes=21,
                                 target_dims = 4):
    ''' Converts regression predictions from preds for all classes  to 
        regression predictions for only the true class for each sample
        from the batch (batch_size, 1, targets_dim).
    Args:
        preds: Predictions for all classes 
            (batch_size, num_classes * targets_dim)
        true_classes: The true class for each sample (batch_size).
        num_classes: How many classes are there.
        target_dims: The number of dimensions in each regression preds,
            e.g. we have 4 for (x, y, w, h).
    Returns:
        Regression predictions for only the true class for each sample,
        (batch_size, 1, targets_dims).
    '''
    # We stack the true classes because the predicted regressions will 
    # have 3 dimensions, and we need the gather indexes in the same dims.
    true_class_stack = torch.stack([true_classes] * target_dims, 1)
    indexes = true_class_stack.view((-1, 1, target_dims))
    
    # Convert regression predictions from 
    # (batch_size, num_classes * target_dims) to
    # (batch_size, num_classes, target_dims)
    preds = preds.view(-1, num_classes, target_dims)
    
    # Gather the predictions using the indexes on the 1 dimension (y).
    # Gather works like:
    # output[x, y, z] = input[x, indexes[x, y, z], z]
    # where x, y, z are bound by the indexes size.
    preds_gathered = preds.gather(1, indexes)
    # remove the empty y dimension
    return preds_gathered.view((-1, target_dims))

def save_weights(model, weights_dir, epoch):
    weights_fname = 'weights-%s.pth' % (epoch)
    weights_fpath = os.path.join(weights_dir, weights_fname)
    torch.save({'state_dict': model.state_dict()}, weights_fpath)
    return weights_fpath

def load_weights(model, fpath):
    state = torch.load(fpath)
    model.load_state_dict(state['state_dict'])