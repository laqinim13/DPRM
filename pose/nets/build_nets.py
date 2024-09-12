# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Author : Pan
# @E-mail : 390737991@qq.com
# @Date   : 2020-10-09 09:51:33
# --------------------------------------------------------
"""
from pose.nets.hrnet import pose_hrnet
from pose.nets.mobilenet import ir_mobilenet_v2, mobilenet_v2
from pose.nets.mobilenet import mobilenet_v3_large, mobilenet_v3_small
from pose.nets import pose_resnet
from pose.nets import pose_resnest
from pose.nets.litehrnet import litehrnet
from pose.nets.lpnet import lpnet


def build_nets(net_type: str, config, is_train=True):
    """
    :param net_type:
    :param config:
    :param is_train:
    :return:
    """
    net_type = net_type.lower()
    if net_type.startswith("resnet".lower()):
        model = pose_resnet.get_pose_net(config, is_train=is_train)
    elif net_type.endswith("pose_resnest"):
        model = pose_resnest.get_pose_net(config, is_train=is_train)
    elif net_type.startswith("HRNet".lower()):
        model = pose_hrnet.get_pose_net(config, is_train=is_train)
    elif net_type.startswith("LiteHRNet".lower()):
        model = litehrnet.get_pose_net(config, is_train=is_train)
    elif net_type == "ir_mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = ir_mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "mobilenet_v2":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult, out_feature=False)
    elif net_type == "mobilenet_v2_kd":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = mobilenet_v2.get_pose_net(config, is_train=is_train, width_mult=width_mult, out_feature=True)
    elif net_type == "mobilenet_v3_large":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = mobilenet_v3_large.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "mobilenet_v3_small":
        width_mult = config.MODEL.EXTRA.WIDTH_MULT
        model = mobilenet_v3_small.get_pose_net(config, is_train=is_train, width_mult=width_mult)
    elif net_type == "lpnet":
        model = lpnet.get_pose_net(config, is_train=is_train)
    else:
        raise Exception("Error:{}".format(config.MODEL.NAME))
    return model


if __name__ == "__main__":
    import torch
    from easydict import EasyDict
    from pybaseutils import yaml_utils
    from basetrainer.utils import torch_tools
    from basetrainer.utils.converter.pytorch2onnx import convert2onnx

    device = "cpu"
    batch_size = 1
    # config_file = "../../configs/coco/lpnet/lpnet_person_192_256.yaml"
    config_file = "../../configs/coco/hrnet/w32_adam_person_192_256.yaml"
    # config_file = "../../configs/coco/hrnet/w48_adam_hand_192_192.yaml"
    # config_file = "../../configs/coco/litehrnet/litehrnet18_hand_192_192.yaml"
    # config_file = "../../configs/coco/litehrnet/litehrnet30_hand_192_192.yaml"
    # config_file = "../../configs/coco/mobilenet/mobilenetv2_hand_192_192.yaml"
    config = EasyDict(yaml_utils.load_config(config_file))
    # input_size = config.MODEL.IMAGE_SIZE
    input_size = [192, 192]
    net_type = config.MODEL.NAME
    input_shape = (batch_size, 3, input_size[1], input_size[0])
    model = build_nets(net_type, config, is_train=False)
    model.eval()
    inputs = torch.randn(size=input_shape).to(device)
    output = model(inputs)
    torch_tools.nni_summary_model(model, batch_size=1, input_size=input_size, device=device)
    # torch_tools.torchinfo_summary(model, batch_size=1, input_size=input_size, device=device)
    print("===" * 10)
    print("inputs.shape:{}".format(inputs.shape))
    print("output.shape:{}".format(output.shape))
    print(net_type)
    file = "_".join([config.MODEL.NAME])
    # torch.save(model.state_dict(), file + ".pth")
    # convert2onnx(model, input_shape=input_shape, input_names=['input'], output_names=['output'], opset_version=11)
    # torch_tools.plot_model(model, output)
