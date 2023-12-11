import mlx.core as mx
import mlx.nn as nn

# input X, gamma (alpha), beta (bias), moving_mean/moving_var = variable, epsilon = 10^-5, momentum = running_avg_grammar, training
def batchnormalization_2d(X, gamma, beta, moving_mean, moving_var, eps = 1e-5, momentum = 0.1, training = True):
    if not training:
        X_hat = (X - moving_mean) / mx.sqrt(moving_var + eps)
    else:
        # X in shape [B, H, W ,C]
        mean = X.mean((0, 1, 2), keepdims = True)
        var = ((X - mean) ** 2).mean((0, 1, 2), keepdims = True)
    
        X_hat = (X - mean) / mx.sqrt(var + eps)

        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var  = (1.0 - momentum) * moving_var  + momentum * var

    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var

class BatchNorm2d(nn.Module):
    def __init__(self, dims: int, eps:float = 1e-5, affine:bool = True, training:bool = True):
        super().__init__()

        shape = (1, 1, 1, dims)
        if affine:
            self.bias = mx.zeros(shape)
            self.weight = mx.ones(shape)

        self.moving_mean = mx.zeros(shape)
        self.moving_var = mx.ones(shape)
        self.dims = dims
        self.eps = eps
        self.training = training
        self.momentum = 0.1

    def __call__(self, X):
        if not self.training:
            X_hat = (X - self.moving_mean) / mx.sqrt(self.moving_var + self.eps)
        else:
            # X in shape [B, H, W ,C]
            mean = X.mean((0, 1, 2), keepdims = True)
            var = ((X - mean) ** 2).mean((0, 1, 2), keepdims = True)
        
            X_hat = (X - mean) / mx.sqrt(var + self.eps)

            self.moving_mean = (1.0 - self.momentum) * self.moving_mean + self.momentum * mean
            self.moving_var  = (1.0 - self.momentum) * self.moving_var  + self.momentum * var

        return self.weight * X_hat + self.bias if "weight" in self else X_hat
    
if __name__ == '__main__':
    mx.random.seed(0)
    x = mx.random.normal((2, 2, 2, 10))
    batchnorm = BatchNorm2d(10, training = True, affine = False)
    result = batchnorm(x)
    print(batchnorm.moving_mean)
    print(batchnorm.moving_var)