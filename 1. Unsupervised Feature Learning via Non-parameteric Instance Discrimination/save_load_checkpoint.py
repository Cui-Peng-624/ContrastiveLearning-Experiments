import os
import torch

# 确保文件夹存在且可写
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    # 设置文件夹权限为可读写（仅对Unix系统有效，Windows系统无需此步骤）
    os.chmod(path, 0o777)

# 保存模型和优化器状态
def save_checkpoint(model, optimizer, epoch, model_path, optimizer_path):
    ensure_dir(os.path.dirname(model_path))
    ensure_dir(os.path.dirname(optimizer_path))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    print(f'Model and optimizer states saved to {model_path} and {optimizer_path}.')

# 加载模型和优化器状态
def load_checkpoint(model, optimizer, model_path, optimizer_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Loaded model and optimizer states from {model_path} and {optimizer_path}, starting from epoch {epoch}.')
    return model, optimizer, epoch