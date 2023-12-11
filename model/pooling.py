import mlx.core as mx
import mlx.nn as nn

def pooling_2d(x, stride: int = 2, padding: int = 0, array_mode:str = 'BHWC', mode = 'average'):
    if stride < 0:
        raise ValueError("The stride should be more than zero")
    
    if padding < 0:
        raise ValueError("The padding should be more than zero")
    
    if padding > 0:
        if array_mode == 'BCHW':
            x = mx.pad(x, [(0,0), (0,0), (padding, padding), (padding,padding)], 0)
        else:
            x = mx.pad(x, [(0,0), (padding, padding), (padding,padding), (0,0)], 0)
            
    if array_mode == 'BCHW':
        B, C, H, W = x.shape
        axis = (3,5)

        if W == stride or H == stride:
            taken = x.reshape(B, C, 1, stride, 1, stride)
        else:
            taken = x.reshape(B, C, W//stride, stride, H//stride, stride)
    else:
        B, H, W, C = x.shape
        axis = (2,4)

        if W == stride or H == stride:
            taken = x.reshape(B, 1, stride, 1, stride, C)
        else:
            taken = x.reshape(B, W//stride, stride, H//stride, stride, C)
        
    if mode == 'average':
        taken = taken.mean(axis)
    elif mode == 'max':
        taken = taken.max(axis)
    else:
        raise ValueError("Invalid Type of Pooling")
    
    return taken

def avg_pool_2d(x, stride: int = 2, padding: int = 0, array_mode:str = 'BHWC'):
    return pooling_2d(x, stride, padding, array_mode, mode = 'average')

def max_pool_2d(x, stride: int = 2, padding: int = 0, array_mode:str = 'BHWC'):
    return pooling_2d(x, stride, padding, array_mode, mode = 'max')

class AvgPool2D(nn.Module):
    def __init__(self, stride: int = 2, padding: int = 0, array_mode:str = 'BHWC'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.array_mode = array_mode
        
    def __call__(self, x: mx.array) -> mx.array:
        return pooling_2d(x, self.stride, self.padding, self.array_mode, mode = 'average')

class MaxPool2D(nn.Module):
    def __init__(self, stride: int = 2, padding: int = 0, array_mode:str = 'BHWC'):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.array_mode = array_mode
        
    def __call__(self, x: mx.array) -> mx.array:
        return pooling_2d(x, self.stride, self.padding, self.array_mode, mode = 'max')
    
class AdaptiveAvgPool2d_toone(nn.Module):
    def __init__(self, flatten = False):
        super().__init__()
        self.flatten = flatten

    def __call__(self, x: mx.array, array_mode:str = 'BHWC') -> mx.array:
        if array_mode == 'BHWC':
            N, H, W, C = x.shape
            x = mx.mean(x, (1,2), keepdims = True)
        else:
            N, C, H, W = x.shape
            x = mx.mean(x, (2,3), keepdims = True)

        if self.flatten:
            x = x.reshape((N, C))

        return x
    
if __name__ == '__main__':
    avgpool = AvgPool2D(4, 0)
    maxpool = MaxPool2D(2, 0)

    sample = mx.random.normal([1, 4, 4, 3])
    result = avgpool(sample)
    print(result.shape)

    result = maxpool(sample)
    print(result.shape)

    adap_avgpooltoone = AdaptiveAvgPool2d_toone(flatten = True)
    result = adap_avgpooltoone(sample)
    print(result.shape)