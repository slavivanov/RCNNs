import torch
from torch.autograd import Variable
import torch.nn.functional as F


def get_parameters(net, lr):
    """ Get the network parameters that have gradients flowing through them,
        with a double learning rate for biases, and normal lr for weights.
    """
    biases = []
    weights = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            if 'bias' in name:
                biases.append(param)
            else:
                weights.append(param)
    return [{'params': biases, 'lr': lr * 2},
            {'params': weights, 'lr': lr},
           ]


def fast_rcnn_loss(outputs, labels, targets, preds_weights):
    """ Compute classification and regression loss"""
    true_class = labels.long()
    classification_loss = F.nll_loss(outputs[0], true_class)

    regression_loss_computed = regression_loss(outputs[1], targets,
                                               preds_weights)

    loss = classification_loss + regression_loss_computed
    return loss


def regression_loss(preds, targets, preds_weights):
    """ Regression Smooth L1 loss between predicted deltas for bboxes and
        the ground truth targets.
    """
    true_deltas = targets

#     predicted_deltas = prepare_predicted_regression(preds, true_classes)
    predicted_deltas = preds * preds_weights
    return F.smooth_l1_loss(predicted_deltas, true_deltas)


def prepare_predicted_regression(preds, true_classes,
                                 num_classes=21,
                                 target_dims = 4):
    """ Converts regression predictions from preds for all classes  to
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
    """
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