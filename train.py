import argparse, os
import numpy as np
import json
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rich.console import Console
from model import hopenet as hopenet_model
from datasets import DataLoader, Pose_300WLP, AFLW2000

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation.')
    parser.add_argument('--val', dest='val', help='val.',
          default=0, type=int)
    parser.add_argument('--flip', dest='flip', help='flip.',
          default=0, type=int)
    parser.add_argument('--augment', dest='augment', help='augment.',
          default=0.5, type=float)
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=160, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=8, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
          default=0.00001, type=float)
    parser.add_argument('--val_dataset', dest='dataset', help='Dataset type.', default='AFLW2000', type=str)
    parser.add_argument(
        '--val_data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/AFLW2000', type=str)
    parser.add_argument(
        '--val_filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/aflw2000_list.txt', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='300W_LP', type=str)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300wlp_files.txt', type=str)
    parser.add_argument('--input_size', dest='target_size', help='target_size',
          default=224, type=int)
    parser.add_argument('--transfer', dest='transfer', help='transfer.',
          default=1, type=int)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output snapshots.', default = 'matrix', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)

    args = parser.parse_args()
    return args

def mse_loss(predict, target):
    return mx.mean(mx.square(predict - target))

def cross_entropy(predict, target):
    return mx.mean(nn.losses.cross_entropy(predict, target))

def loss_function(idx_tensor, f, console, epoch, total_epoch, iter, total_iter, hpe_model, X, y, alpha = 1):
    img = X
    labels, cont_labels, index = y

    # Binned labels
    label_yaw = labels[:,0]
    label_pitch = labels[:,1]
    label_roll = labels[:,2]

    # Continuous labels
    label_yaw_cont = cont_labels[:,0]
    label_pitch_cont = cont_labels[:,1]
    label_roll_cont = cont_labels[:,2]
    
    # Forward Pass
    yaw, pitch, roll = hpe_model(img)

    # Cross entropy loss
    loss_yaw   = cross_entropy(yaw, label_yaw)
    loss_pitch = cross_entropy(pitch, label_pitch)
    loss_roll  = cross_entropy(roll, label_roll)

    # Wrapped loss
    yaw_predicted   = mx.softmax(yaw,   axis = -1)
    pitch_predicted = mx.softmax(pitch, axis = -1)
    roll_predicted  = mx.softmax(roll,  axis = -1)

    yaw_predicted   = mx.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
    pitch_predicted = mx.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
    roll_predicted  = mx.sum(roll_predicted * idx_tensor, 1) * 3 - 99

    loss_reg_yaw   = mse_loss(yaw_predicted,   label_yaw_cont)
    loss_reg_pitch = mse_loss(pitch_predicted, label_pitch_cont)
    loss_reg_roll  = mse_loss(roll_predicted,  label_roll_cont)

    # Total Loss
    loss_yaw += alpha * loss_reg_yaw
    loss_pitch += alpha * loss_reg_pitch
    loss_roll += alpha * loss_reg_roll

    dict_loss = {
        "mode": "train",
        "epoch": epoch + 1,
        "total_epoch": total_epoch,
        "iter": iter + 1,
        "total_iter": total_iter,
        "loss_yaw": np.array(loss_yaw).tolist(),
        "loss_pitch": np.array(loss_pitch).tolist(),
        "loss_roll": np.array(loss_roll).tolist()
    }

    f.write(json.dumps(dict_loss))

    f.write("\n")
    
    if (iter+1) % 50 == 0:
        yaw_str   = np.array(loss_yaw).tolist()
        pitch_str = np.array(loss_pitch).tolist()
        roll_str  = np.array(loss_roll).tolist()
        console.log(f'Epoch [{epoch+1}/{total_epoch}] Iter [{iter+1}/{total_iter}] | Loss Yaw {yaw_str:.3f} Pitch {pitch_str:.3f} Roll {roll_str:.3f}')
    
    return loss_yaw + loss_pitch + loss_roll

def train():
    args = parse_args()
    
    # Epoch
    num_epochs = args.num_epochs

    # Set Random Seed 0
    np.random.seed(0)
    mx.random.seed(0)

    # Load Rich Console
    console = Console()

    # Create The snapshot Folder
    console.log('Create the snapshot folder.')
    if args.output_string != '':
        base_path = f'workdirs_{ args.output_string }'
    else:
        base_path = f'workdirs'
    
    output_path = f'{ base_path }/snapshots'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load Dataset
    console.log('Loading the dataset')
    pose_dataset = Pose_300WLP(args.data_dir, args.filename_list, target_size = args.target_size, 
                               augment = args.augment, flip = args.flip == 1)
    train_loader = DataLoader(pose_dataset, args.batch_size)
    total_iteration = len(train_loader)

    # Load Model
    console.log('Initialization of model.')
    hopenet = hopenet_model(target_size = args.target_size, pretrained = True) #(3, 66, batch_size = args.batch_size, target_size = args.target_size)

    # Load State Dict
    if args.snapshot != '':
        hopenet.load_state_dict(args.snapshot)

    # Preparation
    mx.eval(hopenet.parameters())

    # Set up Gradient
    console.log('Setting up gradient.')
    loss_and_grad_fn = nn.value_and_grad(hopenet, loss_function)

    # Config Optimizer
    console.log('Instantiate the optimizer')
    optimizer = optim.Adam(args.lr)

    # bin
    idx_tensor = mx.array([idx for idx in range(66)])

    # Ready for training
    console.log('Ready for training.')
    f = open(os.path.join(base_path, "training.log"), "a")
    for epoch in range(num_epochs):
        for i, (X, y) in enumerate(train_loader()):
            _ , grads = loss_and_grad_fn(idx_tensor, f, console, epoch, num_epochs, i, total_iteration, hopenet, X, y)
            optimizer.update(hopenet, grads)
            mx.eval(hopenet.parameters(), optimizer.state)

        hopenet.save_state_dict(os.path.join(output_path, f"{epoch+1}.npz"))

if __name__ == '__main__':
    train()
