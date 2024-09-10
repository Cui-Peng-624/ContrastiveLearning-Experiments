import numpy as np
from PIL import Image
import torch

def tensor_to_image(tensor: torch.Tensor, save_path: str = None) -> Image.Image:
    # 先将 PyTorch Tensor 转换为 NumPy 数组
    tensor = tensor.detach().cpu().numpy()

    # 假设输入 tensor 是二维或三维 numpy 数组
    if tensor.ndim == 2:
        # 如果是二维数组，假设是灰度图像
        mode = 'L'
        tensor = tensor.astype(np.uint8)
    elif tensor.ndim == 3 and tensor.shape[2] == 3:
        # 如果是三维数组，且第三维度大小为3，假设是RGB图像
        mode = 'RGB'
        tensor = tensor.astype(np.uint8)
    else:
        raise ValueError("输入的 tensor 必须是一个二维数组或者是一个形状为 (H, W, 3) 的三维数组")

    # 如果值在 0-1 之间，将其缩放到 0-255 之间
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)

    # 将 numpy 数组转换为 PIL 图像
    image = Image.fromarray(tensor, mode=mode)

    # 如果提供了保存路径，将图像保存
    if save_path:
        image.save(save_path)

    return image

def image_to_tensor(image_path: str) -> np.ndarray:
    # 打开图像文件
    image = Image.open(image_path)

    # 将 PIL 图像转换为 numpy 数组
    tensor = np.array(image)

    return tensor

# test
# test_tensor = image_to_tensor("Images/42dcfb28c5f27723e0f086f158a09310.png")
# print(test_tensor.shape)
# test_image = tensor_to_image(test_tensor)
# test_image