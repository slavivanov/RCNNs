import torch
import torchvision


def get_vgg(skip_last_layers=0, pretrained=False):
    """ Returns a trained and ready to go VGG 16 network.
    Args:
        skip_last_layers: How many of the last layers to not include.
        pretrained: Whether to load pretrained weights for this net.
    Returns:
        A torch model.
    """
    vgg = torchvision.models.vgg16(pretrained=pretrained)

    if skip_last_layers > 0:
        layers = [layer for layer in vgg.classifier]
        last_index = -1 * skip_last_layers
        layers = layers[:last_index]
        vgg.classifier = torch.nn.Sequential(*layers)

    return vgg


def get_vgg_conv(skip_last_layers=-1, pretrained=False):
    """ Returns a trained and ready to go features only part of the
        VGG 16 network.
    Args:
        skip_last_layers: How many of the last layers to not include.
        pretrained: Whether to load pretrained weights for this net.
    Returns:
        A torch Sequential model.
    """
    vgg = torchvision.models.vgg16(pretrained=pretrained)

    layers = [layer for layer in vgg.features]

    if skip_last_layers > 0:
        last_index = -1 * skip_last_layers
        layers = layers[:last_index]

    model = torch.nn.Sequential(*layers)

    return model


def fast_rcnn_weights():
    """ Returns the features, fc1 and fc2 weights for fast RCNN.
    """
    vgg = get_vgg(pretrained=True)
    selected_layers = [layer for layer in vgg.features][:-1]
    feat_weights = torch.nn.Sequential(*selected_layers).state_dict()

    fc1_weights = vgg.classifier[0].state_dict()
    fc2_weights = vgg.classifier[3].state_dict()

    return feat_weights, fc1_weights, fc2_weights