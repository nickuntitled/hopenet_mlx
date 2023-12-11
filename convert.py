import torch, argparse
from collections import OrderedDict
from mlx.utils import tree_unflatten, tree_flatten
from model import hopenet
import mlx.core as mx
import torch.utils.model_zoo as model_zoo

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation.')
    parser.add_argument('--weight', dest='weight', help='Path of model ImageNet weight.',
          default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    checkpoint = torch.load("./weights/resnet50-19c8e357.pth" if args.weight == '' else args.weight, map_location=torch.device('cpu'))
    new_state_arr = []
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        split_name = k.split('.')
        if len(split_name) >= 4:
            if split_name[2] == 'conv1' or split_name[2] == 'conv2' or split_name[2] == 'conv3':
                v = torch.permute(v, (0, 2, 3, 1))

            if (split_name[2] == 'bn1' or split_name[2] == 'bn2' or split_name[2] == 'bn3'):
                continue

            if (split_name[2] == 'downsample'):
                if (split_name[3] == '1'):
                    continue
                
                if (split_name[3] == '0'):
                    split_name = [split_name[0], split_name[1], 'downsample_conv', 'weight']
                    v = torch.permute(v, (0, 2, 3, 1))
        else:
            if split_name[0] == 'conv1' or split_name[0] == 'conv2' or split_name[0] == 'conv3':
                v = torch.permute(v, (0, 2, 3, 1))

            if (split_name[0] == 'bn1' or split_name[0] == 'bn2' or split_name[0] == 'bn3'):
                continue
        
        key_name = split_name[0]
        if key_name == 'layer1' or key_name == 'layer2' or key_name == 'layer3' or key_name == 'layer4':
            split_name.insert(1, 'layers')

        var_name = '.'.join(split_name)
        new_state_arr.append([var_name, mx.array(v.detach().numpy())])

    new_state_arr = tree_unflatten(new_state_arr)
    mx.savez("weights/resnet50.npz", **dict(tree_flatten(new_state_arr)))
    
    test = hopenet()
    test.load_state_dict("weights/resnet50.npz")

    print('[*] Convert Successfully')