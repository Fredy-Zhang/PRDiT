"""3D ResNet model factory for FID / MMD evaluation.

Constructs a 3D ResNet of the requested depth and optionally loads pretrained
weights for feature extraction.  Supported depths: 10, 18, 34, 50, 101, 152,
200.
"""

import os
import sys

import torch
from torch import nn

# Import resnet from the local models directory by explicit file path, to avoid
# colliding with the project-root `models` package (PRDiT) that shares the name.
import importlib.util

_resnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "resnet.py")
_spec = importlib.util.spec_from_file_location("eval_resnet", _resnet_path)
resnet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resnet)


def generate_model(opt):
    """Construct a 3D ResNet and optionally load pretrained weights.

    Parameters
    ----------
    opt : argparse.Namespace
        Must expose ``model`` (str), ``model_depth`` (int),
        ``input_W/H/D`` (int), ``resnet_shortcut`` (str),
        ``no_cuda`` (bool), ``n_seg_classes`` (int),
        ``gpu_id`` (int), ``phase`` (str), ``pretrain_path`` (str),
        and ``new_layer_names`` (list of str).

    Returns
    -------
    model : torch.nn.Module
        Constructed (and optionally pretrained) ResNet.
    parameters : dict or generator
        ``{'base_parameters': …, 'new_parameters': …}`` when a pretrained
        checkpoint is loaded; ``model.parameters()`` otherwise.
    """
    assert opt.model in ['resnet']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                sample_input_W=opt.input_W,
                sample_input_H=opt.input_H,
                sample_input_D=opt.input_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)

    if not opt.no_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
            model = model.cuda()
            model = nn.DataParallel(model)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    # load pretrain
    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}

        return model, parameters

    return model, model.parameters()
