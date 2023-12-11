from model import hopenet
import mlx.core as mx
from mlx.utils import tree_flatten

def preview_structure(model):
    if isinstance(model, dict):
        print(model.keys())
        for k in model.keys():
            print(k)
            preview_structure(model[k])

    elif isinstance(model, list):
        for i in range(len(model)):
            print(i)
            preview_structure(model[i])
    else:
        print(model.shape)

if __name__ == '__main__':
    sample = mx.random.normal((1, 224, 224, 3))
    mobile = hopenet(True)
    sample = mobile(sample)

    checkpoint = dict(tree_flatten(mobile.trainable_parameters()))
    for k, v in checkpoint.items():
        print(f"{ k } => { v.shape }")

    mobile.save_state_dict("test.npz")

    print('load')
    mobile.load_state_dict("test.npz")