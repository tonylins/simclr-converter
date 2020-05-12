import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import argparse
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

parser = argparse.ArgumentParser(description='SimCLR converter',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('tf_path', type=str, help='path of the input tensorflow file')
parser.add_argument('pth_path', type=str, help='path of the output pytorch path')

args = parser.parse_args()


def main():
    # 1. read tensorflow weight into a python dict
    vars_list = tf.train.list_variables(args.tf_path)
    vars_list = [v[0] for v in vars_list]
    # print('#vars:', len(vars_list))

    sd = {}
    ckpt_reader = tf.train.load_checkpoint(args.tf_path)
    for v in vars_list:
        sd[v] = ckpt_reader.get_tensor(v)

    sd.pop('global_step')

    # 2. convert the state_dict to PyTorch format
    conv_keys = [k for k in sd.keys() if k.split('/')[1].split('_')[0] == 'conv2d']
    conv_idx = []
    for k in conv_keys:
        mid = k.split('/')[1]
        if len(mid) == 6:
            conv_idx.append(0)
        else:
            conv_idx.append(int(mid[7:]))
    arg_idx = np.argsort(conv_idx)
    conv_keys = [conv_keys[idx] for idx in arg_idx]

    bn_keys = list(set([k.split('/')[1] for k in sd.keys() if k.split('/')[1].split('_')[0] == 'batch']))
    bn_idx = []
    for k in bn_keys:
        if len(k.split('_')) == 2:
            bn_idx.append(0)
        else:
            bn_idx.append(int(k.split('_')[2]))
    arg_idx = np.argsort(bn_idx)
    bn_keys = [bn_keys[idx] for idx in arg_idx]

    if '_1x' in args.tf_path:
        model = resnet50x1()
    elif '_2x' in args.tf_path:
        model = resnet50x2()
    elif '_4x' in args.tf_path:
        model = resnet50x4()
    else:
        raise NotImplementedError

    conv_op = []
    bn_op = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_op.append(m)
        elif isinstance(m, nn.BatchNorm2d):
            bn_op.append(m)

    for i_conv in range(len(conv_keys)):
        m = conv_op[i_conv]
        # assert the weight of conv has the same shape
        assert torch.from_numpy(sd[conv_keys[i_conv]]).permute(3, 2, 0, 1).shape == m.weight.data.shape
        m.weight.data = torch.from_numpy(sd[conv_keys[i_conv]]).permute(3, 2, 0, 1)

    for i_bn in range(len(bn_keys)):
        m = bn_op[i_bn]
        m.weight.data = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/gamma'])
        m.bias.data = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/beta'])
        m.running_mean = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/moving_mean'])
        m.running_var = torch.from_numpy(sd['base_model/' + bn_keys[i_bn] + '/moving_variance'])

    model.fc.weight.data = torch.from_numpy(sd['head_supervised/linear_layer/dense/kernel']).t()
    model.fc.weight.bias = torch.from_numpy(sd['head_supervised/linear_layer/dense/bias'])

    # 3. dump the PyTorch weights.
    torch.save({'state_dict': model.state_dict()}, args.pth_path)


if __name__ == '__main__':
    main()
