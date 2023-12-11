import mlx.core as mx
import mlx.nn as nn
from model import MaxPool2D, AvgPool2D, Bottleneck
from mlx.utils import tree_flatten

class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, block, layers, num_bins, target_size = 224):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.GroupNorm(64, 64)
        self.relu = nn.ReLU()
        self.maxpool = MaxPool2D(stride = 2, padding = 0) #nn.MaxPool2d(kernel_size=3, stride=2, padding=1) <- Comment due to my written MaxPool2D cannot do the same like in Hopenet
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if target_size   == 256:
            self.avgpool = AvgPool2D(8)
        elif target_size == 224:
            self.avgpool = AvgPool2D(7)
        elif target_size == 128:
            self.avgpool = AvgPool2D(4)
        elif target_size == 112:
            self.avgpool = AvgPool2D(4)
        elif target_size == 64:
            self.avgpool = AvgPool2D(2)
        else:
            raise NotImplementedError("Not implement to the other size. You have to use 64, 112, 128, 224, and 256.")
        
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

    def _make_layer(self, block, planes, blocks, stride=1, groups = None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = {
                'conv': nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                'bn': nn.GroupNorm(planes * block.expansion, planes * block.expansion),
            }

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups = groups))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], x.shape[-1])
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll
    
    def save_state_dict(self, file_path):
        mx.savez(file_path, **dict(tree_flatten(self.trainable_parameters())))

    def load_state_dict(self, file_path):
        self.load_weights(file_path)
    
def hopenet(target_size = 224, pretrained = False):
    hopenet_model = Hopenet(Bottleneck, [3, 4, 6, 3], 66, target_size = target_size)
    if pretrained:
        hopenet_model.load_state_dict("weights/resnet50.npz")

    return hopenet_model