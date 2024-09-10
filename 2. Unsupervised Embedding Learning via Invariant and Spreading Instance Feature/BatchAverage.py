import torch
from torch.autograd import Function
from torch import nn
import math
import numpy as np

def top_n_in_rows(matrix: torch.Tensor, n: int) -> torch.Tensor:
    """
    查询二维矩阵中每一行的前 n 个最大的数。
    
    参数:
    matrix (torch.Tensor): 形状为 (m, k) 的二维张量
    n (int): 需要查询的每行中最大的数的数量
    
    返回:
    torch.Tensor: 包含每行前 n 大的数，形状为 (m, n)
    """
    # 使用 torch.topk 函数来查找前 n 大的数
    top_n_values, _ = torch.topk(matrix, n, dim=1)
    
    return top_n_values

class BatchCriterion(nn.Module):
    ''' Compute the loss within each batch  
    '''
    def __init__(self, negM, T, batchSize):
        super(BatchCriterion, self).__init__()
        self.negM = negM
        self.T = T
        self.diag_mat = 1 - torch.eye(batchSize*2).cuda()
        # torch.eye(batchSize*2) 创建了一个单位矩阵，而 1 - torch.eye(batchSize*2) 则将对角线元素设为0，其他元素为1。
        # diag_mat 是一个矩阵，用于在计算相似度时去掉对角线元素，即不考虑样本与自身的相似度。
        # torch.eye(batchSize * 2) 创建了一个单位矩阵，1 - torch.eye(batchSize * 2) 将对角线上的 1 变为 0，其他元素为 1。cuda() 表示将矩阵放在 GPU 上。
        
    def forward(self, x, targets):
        # ### torch.Size([128, 128]) ### torch.Size([64]) ### batch_size设置的是64
        # features 是64原数据+64增强数据的特征，indexes是64原数据对应的index（不是target，label）
        batchSize = x.size(0) # 这里得到的batch_size为128，不是我们初始设置的64
        
        #get positive innerproduct
        reordered_x = torch.cat((x.narrow(0,batchSize//2,batchSize//2), x.narrow(0,0,batchSize//2)), 0) # 这里不就是把x后面的移到前面？有啥用？
        #reordered_x = reordered_x.data
        pos = (x*reordered_x.data).sum(1).div_(self.T).exp_() # 计算每个样本与对应的正样本（增强版本或对比对）的内积，并进行温度缩放（除以 T），然后取指数。pos 包含了正样本对的相似度得分。
        # sum(1) 表示对张量的第二个维度（即行）进行求和。
        # * 一直表示的是逐元素相乘（element-wise multiplication）啊！为什么这里要用 * ？？？？？？？？？
        # 可以这么理解：
        # $f_i^T \cdot \hat{f_i}$ 为(1, 128) * (128, 1) -> (1, 1) 的元素，其实也就是逐元素相乘，而且可以预见的是：x*reordered_x.data 的结果中，前64行与后64行一致
        # pos 就是在计算论文中 Equation 3 的分子

        #get all innerproduct, remove diag
        all_prob = torch.mm(x,x.t().data).div_(self.T).exp_()*self.diag_mat # 乘上diag_mat的作用是将对角线元素变为零，其他位置的元素不变
        # 为什么是x @ x.t()??? Explaination:
        # $f_k^T \cdot \hat{f_i}$ 为(1, 128) * (128, 1)，而x为(128, 128)，第一个维度是2*batch_size，第二个维度是feature_size，所以单独拎出来看x的每一行就是(1, 128)，与(1, 128)意义相同
        # 那么all_prob的含义是：每一张图片与每一张图片之间的相似性，去掉相似性为1的（A对A，B对B，...），可以预见的是 all_prob 还是一个对称矩阵
        if self.negM==1:
            all_div = all_prob.sum(1)
        else:
            #remove pos for neg
            all_div = (all_prob.sum(1) - pos)*self.negM + pos
            # pos 是增强数据集1与增强数据集2的相似度按行求和，(128, 1)
            # all_prob 是所有样本之间的相似性，暂时没有sum，(128, 128)
            # all_prob.sum(1) - pos 咱们拿第一行举例子，all_prob.sum(1)的第一个元素代表增强样本数据集一中的第一个元素与其他所有元素的相似性，pos第一个元素代表强样本数据集一中的第一个元素与强样本数据集二中第一个元素的相似性。所以减掉之后剩余的就是增强样本数据集一中的第一个元素与不是自己和自己的变换的相似度，也就是与所有负样本的相似度！！！！！！！！！
            # 所以 negM 是控制负样本的权重的，negM 越大，越增强与负样本的相似性。

        lnPmt = torch.div(pos, all_div) # pos 代表正样本对的概率，all_div 是所有样本相似度的得分之和，所以 lnPmt 代表样本 x_i 与其正样本 x_i^+ 的相似度在所有样本对中的占比。

        # 测试这个地方 ######################################################################################################################################################
        # n = 5
        # result = top_n_in_rows(all_prob, n)
        # print('\n', result, '\n') # 找到出错原因了，是all_prob中存在一些nan，inf，和e34超大的数（这个maybe没有影响，最主要的是nan和inf）
        #######################################################################################################################################################
        # negative probability
        Pon_div = all_div.repeat(batchSize,1)
        # print("!!!", all_div.shape, Pon_div.shape) # !!! torch.Size([128]) torch.Size([128, 128])
        # print("\n", Pon_div.t(), "\n") # 这里Pon_div.t()中某些行已经nan了 - 那问题出在all_div - 那问题在all_prob
        lnPon = torch.div(all_prob, Pon_div.t()) # all_prob是所有样本之间的相似性，Pon_div.t()举例说明，第一行就是第一个样本与其他所有样本的相似度之和，所以Pon_div.t()一行内的所有元素相同
        lnPon = -lnPon.add(-1) # lnPon = −(lnPon−1) = 1−lnPon
        
        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        # sum(0)，每列求和
        # print(lnPmt.shape, lnPon.shape) # torch.Size([128]) torch.Size([128])
        lnPmtsum = lnPmt.sum(0) # log(正样本对在所有样本对中的占比 之和)
        lnPonsum = lnPon.sum(0) # log(1 - 非正样本对在所有样本对中的占比 之和)
        # 测试这个地方 ######################################################################################################################################################
        # print("\n", lnPmtsum.data, lnPonsum.data, "\n")
        #######################################################################################################################################################
        
        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum)/batchSize
        return loss



"""
这段代码实现了一个名为 `BatchCriterion` 的自定义损失函数类，继承自 `nn.Module`。这个类的作用是计算每个批次内的损失，通常用于对比学习或基于噪声对比估计（NCE, Noise Contrastive Estimation）的任务。下面是对代码的逐行解释。

### 类的初始化部分 (`__init__` 方法)

```python
def __init__(self, negM, T, batchSize):
    super(BatchCriterion, self).__init__()
    self.negM = negM
    self.T = T
    self.diag_mat = 1 - torch.eye(batchSize * 2).cuda()
```

- **`self.negM = negM`**:  
  这里 `negM` 是一个参数，代表负样本的加权系数。它用于控制负样本的权重，在损失计算中会用到。

- **`self.T = T`**:  
  `T` 是温度参数，用于缩放内积值，使得 softmax 函数更加平滑。较小的 `T` 会使得模型对小的差异更加敏感，较大的 `T` 会使得模型的输出更加均匀。

- **`self.diag_mat = 1 - torch.eye(batchSize * 2).cuda()`**:  
  `diag_mat` 是一个矩阵，用于在计算相似度时去掉对角线元素，即不考虑样本与自身的相似度。`torch.eye(batchSize * 2)` 创建了一个单位矩阵，`1 - torch.eye(batchSize * 2)` 将对角线上的 `1` 变为 `0`，其他元素为 `1`。`cuda()` 表示将矩阵放在 GPU 上。

### 前向传播部分 (`forward` 方法)

```python
def forward(self, x, targets):
    batchSize = x.size(0)
```

- **`batchSize = x.size(0)`**:  
  获取当前输入 `x` 的批次大小，即 `batchSize`。通常，`x` 是一个二维张量，形状为 `[batchSize, feature_size]`。

```python
# get positive innerproduct
reordered_x = torch.cat((x.narrow(0, batchSize//2, batchSize//2), x.narrow(0, 0, batchSize//2)), 0)
pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()
```

- **`reordered_x`**:  
  重新排列 `x` 的顺序。`x.narrow(0, batchSize//2, batchSize//2)` 选取 `x` 的后半部分，而 `x.narrow(0, 0, batchSize//2)` 选取前半部分。通过 `torch.cat` 将这两部分连接，得到 `reordered_x`。这通常用于对比学习，其中样本的后半部分可能是它们的增强版本或对比对。

- **`pos`**:  
  计算每个样本与对应的正样本（增强版本或对比对）的内积，并进行温度缩放（除以 `T`），然后取指数。`pos` 包含了正样本对的相似度得分。

```python
# get all innerproduct, remove diag
all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * self.diag_mat
```

- **`all_prob`**:  
  计算 `x` 中所有样本两两之间的内积，得到一个形状为 `[batchSize, batchSize]` 的相似度矩阵。然后进行温度缩放并取指数。`* self.diag_mat` 用于将对角线元素置为 `0`，即去掉样本与自身的相似度。

```python
if self.negM == 1:
    all_div = all_prob.sum(1)
else:
    all_div = (all_prob.sum(1) - pos) * self.negM + pos
```

- **`all_div`**:  
  计算所有负样本相似度的总和。如果 `negM == 1`，直接对 `all_prob` 求和，得到所有负样本的得分总和。如果 `negM != 1`，则减去正样本的得分，然后乘以 `negM`，再加回正样本得分。

```python
lnPmt = torch.div(pos, all_div)
```

- **`lnPmt`**:  
  计算正样本对的概率，即 `pos` 除以 `all_div`，表示正样本得分在所有样本中的占比。

```python
# negative probability
Pon_div = all_div.repeat(batchSize, 1)
lnPon = torch.div(all_prob, Pon_div.t())
lnPon = -lnPon.add(-1)
```

- **`lnPon`**:  
  计算负样本对的概率。`Pon_div` 是 `all_div` 的重复，`lnPon` 是 `all_prob` 除以 `Pon_div.t()`。接着取负值并减去 `1`，得到负样本的概率。

```python
# equation 7 in ref. A (NCE paper)
lnPon.log_()
lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
lnPmt.log_()
```

- **`lnPon.log_()`**:  
  对负样本概率取对数。这部分是基于噪声对比估计（NCE）的思想。

- **`lnPon.sum(1) - (-lnPmt.add(-1)).log_()`**:  
  计算最终的负样本损失。减去正样本对的对数损失。

- **`lnPmt.log_()`**:  
  对正样本概率取对数。

```python
lnPmtsum = lnPmt.sum(0)
lnPonsum = lnPon.sum(0)
```

- **`lnPmtsum` 和 `lnPonsum`**:  
  分别计算所有正样本对和负样本对的对数概率和。

```python
# negative multiply m
lnPonsum = lnPonsum * self.negM
loss = - (lnPmtsum + lnPonsum) / batchSize
return loss
```

- **`lnPonsum = lnPonsum * self.negM`**:  
  负样本部分的损失乘以负样本权重 `negM`。

- **`loss = - (lnPmtsum + lnPonsum) / batchSize`**:  
  计算最终的损失值，将正样本和负样本部分的损失加起来，并除以批次大小取平均。

- **`return loss`**:  
  返回最终的损失值。

### 总结

这个 `BatchCriterion` 类实现了一个自定义的批次内损失函数，用于对比学习或噪声对比估计任务。通过对正负样本的相似度计算，结合温度参数和平滑项，最终计算得到损失值，用于模型训练中的反向传播。
"""