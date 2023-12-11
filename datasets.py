import os
import mlx.core as mx
import numpy as np
from PIL import Image
import utils, random
import albumentations as A

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index_range = list(range(len(self.dataset)))
        self.index = 0
        self.total_iter = len(self.dataset) // self.batch_size

        if shuffle:
            random.shuffle(self.index_range)

    def __len__(self):
        return self.total_iter
    
    def __call__(self):
        for index in range(self.total_iter):
            indexes = self.index_range[index * self.batch_size:(index + 1) * self.batch_size]
            out = []
            for idx in indexes:
                output = self.dataset[idx]
                if len(out) == 0:
                    for i in range(len(output)):
                        out.append([])

                for i in range(len(output)):
                    temp_out = mx.expand_dims(output[i], axis = 0, stream = mx.cpu)
                    out[i].append(temp_out)

            for i in range(len(output)):
                out[i] = mx.concatenate(out[i], axis = 0, stream = mx.cpu)

            img_out = out[0]
            others = out[1:]

            yield img_out, others
    
class Pose_300WLP:
    def __init__(self, root_path, filename_path, img_ext='.jpg', annot_ext='.mat', 
        image_mode='RGB', crop_size = 160, target_size = 128, augment = 0.5, flip = False):

        self.data_dir = root_path
        self.transform = self.albu_augmentations(target_size, crop_size)
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        self.X_train, self.y_train = [], []
        self.bbox = []

        filename_list = utils.get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list

        self.image_mode = image_mode
        self.length = len(self.X_train)

        self.augment = augment
        self.flip = flip

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def albu_augmentations(self, target_size = 128, p = 0.5):
        albu_transformations = [A.GaussianBlur(p=p), #p=1.0
                                A.RandomBrightnessContrast(p=p), #p=1.0
                                A.augmentations.transforms.GaussNoise(p=p)] #p=1.0
        albu_transformations = [A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0),
            x]) for x in albu_transformations]

        transformations = A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0),
            A.augmentations.crops.transforms.RandomResizedCrop(target_size, target_size, (0.8, 1))
        ])

        resize_compose = A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0)
        ])
        
        return [*albu_transformations, transformations, resize_compose]

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        
        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)

        augment_or_not = np.random.random_sample()

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)

        # And convert to degrees.
        pitch = pose[0] * (180 / np.pi)
        yaw = pose[1] * (180 / np.pi)
        roll = pose[2] * (180 / np.pi)

        if augment_or_not <= self.augment and self.augment > 0:
            # Bounding box Augmentation
            rand = random.randint(1, 4) if self.flip else random.randint(2, 4)
            if rand == 1: # Flip
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                yaw = -yaw
                roll = -roll
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            elif rand == 2: # Random Shifting
                mid_x = int((x_max + x_min) / 2)
                mid_y = int((y_max + y_min) / 2)
                width = x_max - x_min
                height = y_max - y_min
                kx = np.random.random_sample() * 0.2 - 0.1
                ky = np.random.random_sample() * 0.2 - 0.1
                shiftx = mid_x + width * kx
                shifty = mid_y + height * ky
                x_min = shiftx - width/2
                x_max = shiftx + width/2
                y_min = shifty - height/2
                y_max = shifty + height/2
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            elif rand == 3: # Random Scaling
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
                img = np.array(img)
                img = self.transform[-2](image = img)['image']
            else:
                img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

            # Image Augmentation
            rand = random.randint(1, 4)
            img = np.array(img)
            if rand >= 1 and rand <= 3:
                img = self.transform[rand-1](image = img)['image']
        else:
            img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
            img = np.array(img)

        # finalize Transform
        img = self.transform[-1](image = img)['image']
        
        # Convert into MLX array
        img = mx.array(img) / 255
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = mx.array(binned_pose)
        cont_labels = mx.array([yaw, pitch, roll])
        index = mx.array([index])

        return img, labels, cont_labels, index

    def __len__(self):
        # 122,415
        return self.length
    
# AFLW2000 Dataset
class AFLW2000:
    # AFLW dataset with flipping
    def __init__(self, data_dir, filename_path, target_size = 224, image_mode='RGB', ad = 0.2):
        self.data_dir = data_dir
        self.transform_albu = A.Compose([
            A.augmentations.geometric.resize.Resize(target_size, target_size, p = 1.0)
        ])

        filename_list = utils.get_list_from_filenames(filename_path)
        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(self.X_train)

        self.img_ext = '.jpg'
        self.annot_ext = '.mat'
        self.ad = ad

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        print(f"[*] Loaded { self.length } images.")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext)) 
        img = img.convert(self.image_mode) 

        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt3d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = ad = self.ad
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        pose = utils.get_ypr_from_mat(mat_path)

        # And convert to degrees.
        pose = np.array(pose)

        # And convert to degrees.
        pitch = pose[0] * (180 / np.pi)
        yaw = pose[1] * (180 / np.pi)
        roll = pose[2] * (180 / np.pi)
        
        # Convert into MLX array
        img = np.array(img)
        img = self.transform_albu(image = img)['image']
        img = mx.array(img) / 255
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        labels = mx.array(binned_pose)
        cont_labels = mx.array([yaw, pitch, roll])
        index = mx.array([index])

        return img, labels, cont_labels, index