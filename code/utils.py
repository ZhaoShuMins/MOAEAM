import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def trans2_data(data):
    # Convert non-numeric values to numeric (replace this with your own preprocessing logic)
    data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Convert to PyTorch Tensor
    data_tensor = torch.Tensor(data.values)
    return data_tensor


def pairwise_row_distances(output):
    output_expanded = output.unsqueeze(1)
    output_transposed = output.unsqueeze(0).transpose(0, 1)

    diffs = output_expanded - output_transposed

    distances_squared = torch.sum(diffs ** 2, dim=2)

    distances = torch.sqrt(distances_squared)

    triu_indices = torch.triu_indices(distances.size(0), distances.size(0))
    distances_vector = distances[triu_indices]

    return distances_vector


def pearson_correlation(distance_vector1, distance_vector2):
    # 计算均值
    mean_d1 = torch.mean(distance_vector1)
    mean_d2 = torch.mean(distance_vector2)
    # 计算标准差
    std_d1 = torch.std(distance_vector1)
    std_d2 = torch.std(distance_vector2)
    # 计算协方差
    cov = (distance_vector1 - mean_d1) * (distance_vector2 - mean_d2).mean()
    # 计算皮尔逊相关系数
    if std_d1 > 0 and std_d2 > 0:
        correlation = cov / (std_d1 * std_d2)
    else:
        correlation = torch.tensor(0.0)  # 如果标准差为零，则相关系数为零
    return correlation


# In[35]:

class Customloss(nn.Module):
    def __init__(self, lambda_or, lambda_reg, lambda_c):
        super(Customloss, self).__init__()
        self.lambda_reg = lambda_reg
        self.lambda_or = lambda_or
        self.lambda_c = lambda_c

    def forward(self, e1,l1, l2, c3,target_y_mlp):
        orgin_loss = torch.mean((l1 - l2) ** 2)
        distance_vector1 = pairwise_row_distances(e1)
        distance_vector2 = pairwise_row_distances(l1)
        distance_correlation = pearson_correlation(distance_vector1, distance_vector2)
        reg_loss = 1 - distance_correlation
        cs_loss = torch.nn.functional.cross_entropy(c3,target_y_mlp)
        return self.lambda_or * orgin_loss + self.lambda_reg * reg_loss+self.lambda_c * cs_loss


def calculate_loss_term(WF):

    WtW = torch.matmul(WF, WF.T)

    L1_WtW = torch.norm(WtW, p=1)

    L2_WF_squared = torch.norm(WF, p=2) ** 2

    loss = L1_WtW - L2_WF_squared

    return loss


class WFLoss(nn.Module):
    def __init__(self, lambda_w):
        super(WFLoss, self).__init__()
        self.lambda_w = lambda_w

    def forward(self, model):
        W1 = model.autoencoder1.get_first_layer_weights()
        W2 = model.autoencoder2.get_first_layer_weights()
        W3 = model.autoencoder3.get_first_layer_weights()
        w1loss = calculate_loss_term(W1)
        w2loss = calculate_loss_term(W2)
        w3loss = calculate_loss_term(W3)
        wloss =  (w1loss+w2loss+w3loss)/3
        return self.lambda_w * wloss


def aemlp_loss(model, x, x_hat, y_mlp, target_y_mlp, l1, l2, c3,x_c, lambda_sa, lambda_or, lambda_reg, lambda_w,lambda_c):
    ae_loss_fn = Customloss(lambda_or, lambda_reg, lambda_c)
    ae_loss1 = ae_loss_fn(x_c[0],l1[0], l2[0], c3[0],target_y_mlp)
    ae_loss2 = ae_loss_fn(x_c[1], l1[1], l2[1], c3[1], target_y_mlp)
    ae_loss3 = ae_loss_fn(x_c[2], l1[2], l2[2], c3[2], target_y_mlp)
    ae_loss = (ae_loss1+ae_loss2+ae_loss3)/3
    wf_loss_fn = WFLoss(lambda_w)
    wf_loss = wf_loss_fn(model)
    # 将 target_y_mlp 转换为一维张量
    target_y_mlp = target_y_mlp.long().view(-1)
    # 计算分类损失（Cross Entropy Loss）
    sa_loss = torch.nn.functional.cross_entropy(y_mlp, target_y_mlp)
    # 计算总损失
    total_loss = ae_loss + lambda_sa * sa_loss + wf_loss

    return total_loss

def load_data(folder_path):
    # 加载训练数据
    omic1_data = pd.read_csv(f'{folder_path}/1_tr.csv', header=None)
    omic2_data = pd.read_csv(f'{folder_path}/2_tr.csv', header=None)
    omic3_data = pd.read_csv(f'{folder_path}/3_tr.csv', header=None)
    omic1t_data = pd.read_csv(f'{folder_path}/1_te.csv', header=None)
    omic2t_data = pd.read_csv(f'{folder_path}/2_te.csv', header=None)
    omic3t_data = pd.read_csv(f'{folder_path}/3_te.csv', header=None)

    # 加载标签数据
    labels_df = pd.read_csv(f'{folder_path}/labels_tr.csv', header=None)
    labels_df_t = pd.read_csv(f'{folder_path}/labels_te.csv', header=None)

    # 假设你有一个trans_data函数来转换数据
    x = torch.cat((trans2_data(omic1_data), trans2_data(omic2_data), trans2_data(omic3_data)), dim=1)
    target_y_sa = labels_df.values.flatten()
    target_y_sa = torch.tensor(target_y_sa).long()

    x_test = torch.cat((trans2_data(omic1t_data), trans2_data(omic2t_data), trans2_data(omic3t_data)), dim=1)
    target_y_sa_t = labels_df_t.values.flatten()
    target_y_sa_t = torch.tensor(target_y_sa_t).long()

    o1 = torch.cat((trans2_data(omic1_data), trans2_data(omic1t_data)), dim=0)
    o2 = torch.cat((trans2_data(omic2_data), trans2_data(omic2t_data)), dim=0)
    o3 = torch.cat((trans2_data(omic3_data), trans2_data(omic3t_data)), dim=0)

    # 合并数据
    x_combined = torch.cat((x, x_test), dim=0)
    y_combined = torch.cat((target_y_sa, target_y_sa_t), dim=0)

    return x_combined, y_combined,o1,o2,o3

def load_data2(folder_path,batch_size):
    # 加载训练数据
    omic1_data = pd.read_csv(f'{folder_path}/1_tr.csv', header=None)
    omic2_data = pd.read_csv(f'{folder_path}/2_tr.csv', header=None)
    omic3_data = pd.read_csv(f'{folder_path}/3_tr.csv', header=None)
    omic1t_data = pd.read_csv(f'{folder_path}/1_te.csv', header=None)
    omic2t_data = pd.read_csv(f'{folder_path}/2_te.csv', header=None)
    omic3t_data = pd.read_csv(f'{folder_path}/3_te.csv', header=None)

    # 加载标签数据
    labels_df = pd.read_csv(f'{folder_path}/labels_tr.csv', header=None)
    labels_df_t = pd.read_csv(f'{folder_path}/labels_te.csv', header=None)

    target_y_sa = labels_df.values.flatten()
    target_y_sa = torch.tensor(target_y_sa).long()
    target_y_sa_t = labels_df_t.values.flatten()
    target_y_sa_t = torch.tensor(target_y_sa_t).long()


    # 将三个特征集组合成训练集和测试集的输入
    X_train = (trans2_data(omic1_data), trans2_data(omic2_data), trans2_data(omic3_data))
    X_test = (trans2_data(omic1t_data), trans2_data(omic2t_data), trans2_data(omic3t_data))

    # 创建TensorDataset
    train_dataset = TensorDataset(*X_train, target_y_sa)  # 使用*来解包元组
    test_dataset = TensorDataset(*X_test, target_y_sa_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    return train_loader,test_loader


def load_data3(folder_path,batch_size):
    # 加载训练数据
    omic1_data = pd.read_csv(f'{folder_path}/1_tr.csv', header=None)
    omic2_data = pd.read_csv(f'{folder_path}/2_tr.csv', header=None)
    omic3_data = pd.read_csv(f'{folder_path}/3_tr.csv', header=None)
    omic1t_data = pd.read_csv(f'{folder_path}/1_te.csv', header=None)
    omic2t_data = pd.read_csv(f'{folder_path}/2_te.csv', header=None)
    omic3t_data = pd.read_csv(f'{folder_path}/3_te.csv', header=None)
    omic1v_data = pd.read_csv(f'{folder_path}/1_val.csv', header=None)
    omic2v_data = pd.read_csv(f'{folder_path}/2_val.csv', header=None)
    omic3v_data = pd.read_csv(f'{folder_path}/3_val.csv', header=None)

    # 加载标签数据
    labels_df = pd.read_csv(f'{folder_path}/labels_tr.csv', header=None)
    labels_df_t = pd.read_csv(f'{folder_path}/labels_te.csv', header=None)
    labels_df_v = pd.read_csv(f'{folder_path}/labels_val.csv', header=None)

    target_y_sa = labels_df.values.flatten()
    target_y_sa = torch.tensor(target_y_sa).long()
    target_y_sa_t = labels_df_t.values.flatten()
    target_y_sa_t = torch.tensor(target_y_sa_t).long()
    target_y_sa_v = labels_df_v.values.flatten()
    target_y_sa_v = torch.tensor(target_y_sa_v).long()


    # 将三个特征集组合成训练集和测试集的输入
    X_train = (trans2_data(omic1_data), trans2_data(omic2_data), trans2_data(omic3_data))
    X_test = (trans2_data(omic1t_data), trans2_data(omic2t_data), trans2_data(omic3t_data))
    X_val = (trans2_data(omic1v_data), trans2_data(omic2v_data), trans2_data(omic3v_data))

    # 创建TensorDataset
    train_dataset = TensorDataset(*X_train, target_y_sa)  # 使用*来解包元组
    test_dataset = TensorDataset(*X_test, target_y_sa_t)
    val_dataset = TensorDataset(*X_val, target_y_sa_v)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



    return train_loader,test_loader,val_loader



