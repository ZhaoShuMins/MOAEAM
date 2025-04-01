import torch.nn as nn
import torch
import torch.autograd
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn / abs(attn.min())
        attn = self.dropout(F.softmax(F.normalize(attn, dim=-1), dim=-1))
        output = torch.matmul(attn, v)

        return output, attn, v


class FeedForwardLayer(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class VariLengthInputLayer(nn.Module):
    def __init__(self, input_data_dims, d_k, d_v, n_head, dropout):
        super(VariLengthInputLayer, self).__init__()
        self.n_head = n_head
        self.dims = input_data_dims
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = []
        self.w_ks = []
        self.w_vs = []
        for i, dim in enumerate(self.dims):
            self.w_q = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_k = nn.Linear(dim, n_head * d_k, bias=False)
            self.w_v = nn.Linear(dim, n_head * d_v, bias=False)
            self.w_qs.append(self.w_q)
            self.w_ks.append(self.w_k)
            self.w_vs.append(self.w_v)
            self.add_module('linear_q_%d_%d' % (dim, i), self.w_q)
            self.add_module('linear_k_%d_%d' % (dim, i), self.w_k)
            self.add_module('linear_v_%d_%d' % (dim, i), self.w_v)

        self.attention = Attention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.fc = nn.Linear(n_head * d_v, n_head * d_v)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(n_head * d_v, eps=1e-6)

    def forward(self, input_data, mask=None):

        temp_dim = 0
        bs = input_data.size(0)
        modal_num = len(self.dims)
        q = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        k = torch.zeros(bs, modal_num, self.n_head * self.d_k).to(device)
        v = torch.zeros(bs, modal_num, self.n_head * self.d_v).to(device)

        for i in range(modal_num):
            w_q = self.w_qs[i]
            w_k = self.w_ks[i]
            w_v = self.w_vs[i]

            data = input_data[:, temp_dim: temp_dim + self.dims[i]]
            temp_dim += self.dims[i]
            q[:, i, :] = w_q(data)
            k[:, i, :] = w_k(data)
            v[:, i, :] = w_v(data)

        q = q.view(bs, modal_num, self.n_head, self.d_k)
        k = k.view(bs, modal_num, self.n_head, self.d_k)
        v = v.view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, residual = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        residual = residual.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, dropout):
        super(EncodeLayer, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Attention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, modal_num, mask=None):
        bs = q.size(0)
        residual = q
        q = self.w_q(q).view(bs, modal_num, self.n_head, self.d_k)
        k = self.w_k(k).view(bs, modal_num, self.n_head, self.d_k)
        v = self.w_v(v).view(bs, modal_num, self.n_head, self.d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn, _ = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(bs, modal_num, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class OutputLayer(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, modal_num, dropout=0.5):
        super(OutputLayer, self).__init__()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Sequential(nn.Linear(d_hidden + modal_num ** 2, n_classes))

    def forward(self, x, attn_embedding):

        x = self.mlp_head(x)
        combined_x = torch.cat((x, attn_embedding), dim=-1)
        output = self.classifier(combined_x)

        return output


class Autoencoder1(nn.Module):
    def __init__(self, fan_in, encodim,subtypes):
        super(Autoencoder1, self).__init__()
        # 编码器层单独定义以便于访问
        self.encoder_layer1 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 3000),
            nn.BatchNorm1d(3000),
            nn.ELU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(3000, encodim),
            nn.ELU()
        )

        # 解码器
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(encodim, 3000),
            nn.ELU()
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(3000, fan_in)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(encodim, 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )

    def get_first_layer_weights(self):
        return self.encoder_layer1[1].weight

    def forward(self, x):
        # 编码过程
        x = self.encoder_layer1(x)
        encoder_output1 = x
        x = self.encoder_layer2(x)
        encoded_output = x

        # 分类过程
        pre = self.classifier(encoded_output)

        # 解码过程
        x = self.decoder_layer1(encoded_output)
        decoder_layer1_output = x
        x = self.decoder_layer2(x)
        decoded_output = x

        return encoded_output,decoded_output,encoder_output1, decoder_layer1_output, pre

class Autoencoder2(nn.Module):
    def __init__(self, fan_in, encodim,subtypes):
        super(Autoencoder2, self).__init__()
        # 编码器层单独定义以便于访问
        self.encoder_layer1 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 3000),
            nn.BatchNorm1d(3000),
            nn.ELU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(3000, encodim),
            nn.ELU()
        )

        # 解码器
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(encodim, 3000),
            nn.ELU()
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(3000, fan_in)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(encodim, 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )

    def get_first_layer_weights(self):
        return self.encoder_layer1[1].weight

    def forward(self, x):
        # 编码过程
        x = self.encoder_layer1(x)
        encoder_output1 = x
        x = self.encoder_layer2(x)
        encoded_output = x

        # 分类过程
        pre = self.classifier(encoded_output)

        # 解码过程
        x = self.decoder_layer1(encoded_output)
        decoder_layer1_output = x
        x = self.decoder_layer2(x)
        decoded_output = x

        return encoded_output,decoded_output,encoder_output1, decoder_layer1_output, pre


class Autoencoder3(nn.Module):
    def __init__(self, fan_in, encodim,subtypes):
        super(Autoencoder3, self).__init__()
        # 编码器层单独定义以便于访问
        self.encoder_layer1 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(fan_in, 3000),
            nn.BatchNorm1d(3000),
            nn.ELU()
        )
        self.encoder_layer2 = nn.Sequential(
            nn.Linear(3000, encodim),
            nn.ELU()
        )

        # 解码器
        self.decoder_layer1 = nn.Sequential(
            nn.Linear(encodim, 3000),
            nn.ELU()
        )
        self.decoder_layer2 = nn.Sequential(
            nn.Linear(3000, fan_in)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(encodim, 1024),
            nn.ELU(),
            nn.Linear(1024, subtypes)
        )

    def get_first_layer_weights(self):
        return self.encoder_layer1[1].weight

    def forward(self, x):
        # 编码过程
        x = self.encoder_layer1(x)
        encoder_output1 = x
        x = self.encoder_layer2(x)
        encoded_output = x

        # 分类过程
        pre = self.classifier(encoded_output)

        # 解码过程
        x = self.decoder_layer1(encoded_output)
        decoder_layer1_output = x
        x = self.decoder_layer2(x)
        decoded_output = x

        return encoded_output,decoded_output,encoder_output1, decoder_layer1_output, pre

# In[4]:

class Multiomics_Attention_mechanism(nn.Module):
    def __init__(self):
        super().__init__()

        self.hiddim = 3
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_x1 = nn.Linear(in_features=3, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=3)
        self.sigmoidx = nn.Sigmoid()

    def forward(self,input_list):
        new_input_list1 = input_list[0].reshape(1, 1, input_list[0].shape[0], -1)
        new_input_list2 = input_list[1].reshape(1, 1, input_list[1].shape[0], -1)
        new_input_list3 = input_list[2].reshape(1, 1, input_list[2].shape[0], -1)
        XM = torch.cat((new_input_list1, new_input_list2, new_input_list3), 1)

        x_channel_attenttion = self.globalAvgPool(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]



class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm, num_class):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = num_class
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)

        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output = self.Outputlayer(x, attn_embedding)
        return output



class AemlpModel(nn.Module):
    def __init__(self, ae_input_dim, ae_class_dim,input_data_dims, num_classes, lambda_sa, lambda_or,
                 lambda_reg,
                 lambda_w, lambda_c, hyperpm):
        super(AemlpModel, self).__init__()
        self.autoencoder1 = Autoencoder1(ae_input_dim[0],input_data_dims[0], ae_class_dim[0])
        self.autoencoder2 = Autoencoder2(ae_input_dim[1],input_data_dims[1], ae_class_dim[1])
        self.autoencoder3 = Autoencoder3(ae_input_dim[2],input_data_dims[2], ae_class_dim[2])


        self.transformer_encoder = TransformerEncoder(input_data_dims, hyperpm, num_classes)
        self.attention_mechanism = Multiomics_Attention_mechanism()

        self.lambda_sa = lambda_sa
        self.lambda_or = lambda_or
        self.lambda_reg = lambda_reg
        self.lambda_w = lambda_w
        self.lambda_c = lambda_c

    def forward(self, x):
        x_hat1, z_ae1, l1_1, l2_1, c3_1 = self.autoencoder1(x[0])
        x_hat2, z_ae2, l1_2, l2_2, c3_2 = self.autoencoder2(x[1])
        x_hat3, z_ae3, l1_3, l2_3, c3_3 = self.autoencoder3(x[2])

        x_hat = self.attention_mechanism([x_hat1, x_hat2, x_hat3])
        new_data = torch.cat([x_hat[0], x_hat[1], x_hat[2]], dim=1)
        y_transformer = self.transformer_encoder(new_data)



        return new_data, y_transformer, [l1_1, l1_2, l1_3], [l2_1, l2_2, l2_3], [c3_1, c3_2, c3_3], [x_hat1,x_hat2,x_hat3]