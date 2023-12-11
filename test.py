import argparse
import numpy as np
import mlx.core as mx

from tqdm import tqdm
from rich.console import Console
from model import hopenet as hopenet_model
from datasets import DataLoader, AFLW2000

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=8, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--val_dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)
    parser.add_argument(
        '--val_data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/AFLW2000', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--val_filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/aflw2000_list.txt', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/AFLW2000', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/aflw2000_list.txt', type=str)
    parser.add_argument('--input_size', dest='input_size', help='input_size',
          default=128, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def evaluation(idx_tensor, hpe_model, X, y):
    img = X
    _ , cont_labels, _ = y

    # Continuous labels
    label_yaw   = cont_labels[:,0]
    label_pitch = cont_labels[:,1]
    label_roll  = cont_labels[:,2]
    
    # Forward Pass
    yaw, pitch, roll = hpe_model(img)

    # Wrapped loss
    yaw_predicted   = mx.softmax(yaw,   axis = -1)
    pitch_predicted = mx.softmax(pitch, axis = -1)
    roll_predicted  = mx.softmax(roll,  axis = -1)

    yaw_predicted   = mx.sum(yaw_predicted * idx_tensor,   1, stream = mx.cpu) * 3 - 99
    pitch_predicted = mx.sum(pitch_predicted * idx_tensor, 1, stream = mx.cpu) * 3 - 99
    roll_predicted  = mx.sum(roll_predicted * idx_tensor,  1, stream = mx.cpu) * 3 - 99

    yaw_error   = mx.sum(mx.abs(yaw_predicted - label_yaw),     stream = mx.cpu)
    pitch_error = mx.sum(mx.abs(pitch_predicted - label_pitch), stream = mx.cpu)
    roll_error  = mx.sum(mx.abs(roll_predicted - label_roll),   stream = mx.cpu)

    yaw_error   = mx.expand_dims(yaw_error, 0)
    pitch_error = mx.expand_dims(pitch_error, 0)
    roll_error  = mx.expand_dims(roll_error, 0)

    return mx.concatenate([yaw_error, pitch_error, roll_error], 0)

def test():
    args = parse_args()

    # Set Random Seed 0
    np.random.seed(0)
    mx.random.seed(0)

    # Load Rich Console
    console = Console()

    # Load Dataset
    console.log('Loading the dataset')
    if args.dataset == 'AFLW2000':
        val_dataset  = AFLW2000(args.data_dir, args.filename_list, target_size = args.input_size)
        test_loader  = DataLoader(val_dataset, args.batch_size)
        total_val_images    = len(val_dataset)
    else:
        raise NotImplementedError("The dataset another than AFLW2000 is not implemented yet.")

    # Load Model
    console.log('Initialization of model.')
    hopenet = hopenet_model(target_size = args.input_size, pretrained = False)

    # Load State Dict
    if args.snapshot != '':
        hopenet.load_state_dict(args.snapshot)
    else:
        console.log('You have to provide snapshot.')

    # Preparation
    mx.eval(hopenet.parameters())

    # bin
    idx_tensor = mx.array([idx for idx in range(66)])

    # Ready for training
    console.log('Ready for evaluation.')
    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    for i, (X, y) in tqdm(enumerate(test_loader())):
        result = evaluation(idx_tensor, hopenet, X, y)
        mx.eval(result)

        yaw, pitch, roll = result
        yaw_error   += yaw
        pitch_error += pitch
        roll_error  += roll

    yaw_error   = yaw_error / total_val_images
    pitch_error = pitch_error / total_val_images
    roll_error  = roll_error / total_val_images
    mean_error  = (yaw_error + pitch_error + roll_error) / 3.0

    yaw_error   = np.array(yaw_error).tolist()
    pitch_error = np.array(pitch_error).tolist()
    roll_error  = np.array(roll_error).tolist()
    mean_error  = np.array(mean_error).tolist()
    
    console.log(f'Test Result from {str(total_val_images)} images : MAE of Yaw {yaw_error:.3f} Pitch {pitch_error:.3f} Roll {roll_error:.3f} Average {mean_error:.3f}')

if __name__ == '__main__':
    test()
