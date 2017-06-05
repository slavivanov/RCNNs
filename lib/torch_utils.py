import os
import torch
from torch.autograd import Variable


def np_to_var(x, is_cuda=True, dtype=torch.FloatTensor):
    """ Convert a np array to a Torch Variable.
    Args:
        x (ndarray): Data to convert.
        is_cuda (bool): Whether to transfer the data to the GPU.
        dtype (torch tensor class): The target type of the Variable.
    Returns:
        var (Variable): The data wrapped in a Torch Variable.
        """
    var = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        var = var.cuda()
    return var


def flatten(x):
    """ Flatten a Torch tensor to a single dimension.
    Args:
        x (Torch Tensor): The tensor or variable to flatten.
    Returns:
        x (Torch Tensor): The tensor flattened to a single dimension.
    """
    return x.view(x.size(0), -1)


def save_weights(model, weights_dir, epoch):
    """ Save the weights of a model.
    Args:
        model (Torch Module): The model whose weights to save.
        weights_dir: Dir where to save.
        epoch (string): Text to add to the filename of the saved weights.
    Returns:
        weights_fpath (string): The fullpath to the saved model weights.
    """
    weights_fname = 'weights-%s.pth' % (epoch)
    weights_fpath = os.path.join(weights_dir, weights_fname)
    torch.save({'state_dict': model.state_dict()}, weights_fpath)
    return weights_fpath


def load_weights(model, fpath):
    """ Load model weights from file.
        Args:
            model (Torch Module): The model to load weights for.
            fpath (string): File from where to load weights.
        Returns:
            None.
    """
    state = torch.load(fpath)
    model.load_state_dict(state['state_dict'])


def soft_load_weights(model, weights):
    """ Loads any weights that have the same name in model and weights.
        Not very safe in general but works for some cases.
    Args:
        model (Torch Module): The model to load weights for.
        weights (state_dict): Weights as a torch state_dict.
    Returns:
        None.
    """
    own_state = model.state_dict()
    for name, param in weights.items():
        if name not in own_state:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
