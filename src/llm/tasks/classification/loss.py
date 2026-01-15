import torch

## 计算分类损失的函数
"""
笔记：
先列出三个变量的shape: 
input_batch.shape [B, T] 也就是[batch_size, max_length]
target_batch.shape [B] 也就是[batch_size] 相当于一维数组, 表示每个样本的正确标签, 比如[1, 0, 1, 0]
logits_batch.shape [B, C] 也就是[batch_size, classification_num] 如果是2分类, 那么就代表每个样本都会预测出两个类别的分数

疑惑的点: 为什么target_batch和logits_batch的shape都不一致也可以计算loss? 这里还是陷入了基本的回归问题的loss计算……
这里应该将target_batch中的每个样本看作是一个索引, 比如二分类中, 0和1代表索引。在计算loss的时候我们会用索引去寻找logits_batch中每个样本对应的类别分数
比如logits_batch=[[-0.121, 3.233], [3.325, -0.124]] target_batch=[1, 0]。 这里取loss[3.233, 3.325]去计算
而cross_entropy所要做的就是计算 "你给正确答案分了多大的概率?"
"""
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)     
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy( 
        logits, target_batch   
    )
    return loss 

## 训练时用于根据dataloader来计算train_loss和val_loss, 这里计算的是随机样本的平均loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0     
    if len(data_loader) == 0:         
        return float("nan")      
    elif num_batches is None: 
        num_batches = len(data_loader) 
    else:         
        num_batches = min(num_batches, len(data_loader))     
    for i, (input_batch, target_batch) in enumerate(data_loader):         
        if i < num_batches:             
            loss = calc_loss_batch(                 
                input_batch, target_batch, model, device             
            )              
            total_loss += loss.item()          
        else: 
            break 
    return total_loss / num_batches