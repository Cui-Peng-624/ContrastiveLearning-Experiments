import zipfile
import os

def unzip_file(zip_file_path, extract_to_path):
    """
    解压给定的zip文件到指定的目录。

    参数:
    zip_file_path (str): zip文件的路径
    extract_to_path (str): 解压缩文件的目标路径
    """
    # 检查目标路径是否存在，不存在则创建
    if not os.path.exists(extract_to_path):
        os.makedirs(extract_to_path)

    # 解压zip文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

    print(f"文件已解压到 {extract_to_path}")

# 使用示例
# unzip_file('./data/archive.zip', './data/')