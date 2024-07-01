import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import copy
from utils import one_hot_tensor, cal_sample_weight, gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

        
def restore_input_from_output(output_tensor, x_dim_set):
    X_set = {}
    #adj_list = []
    idx = 0
    for i in range(len(x_dim_set)):
        X_set[i] = output_tensor[:, idx:idx+x_dim_set[i]]
        idx += x_dim_set[i]
        
    M = output_tensor[:, -len(x_dim_set):]
    
    return X_set, M

class GCFNN(nn.Module):
    
    def __init__(
        self,
        input_dims, 
        network_settings, 
        cuda = False
    ):
        super(GCFNN, self).__init__()
        self.device = torch.device("cuda" if cuda == True else "cpu")
        self.M                = len(input_dims['x_dim_set'])
        self.x_dim_set = {}
        for m in range(self.M):
            self.x_dim_set[m] = input_dims['x_dim_set'][m]
        
        self.dim_enc          = network_settings['dim_enc']        #encoder hidden nodes
        self.num_layers_enc   = network_settings['num_layers_enc'] #encoder layers

        self.z_dim            = input_dims['z_dim']           
        #self.steps_per_batch  = input_dims['steps_per_batch']
        
        self.dim_specificpre  = network_settings['dim_specificpre']      #predictor hidden nodes
        self.num_layers_specificpre      = network_settings['num_layers_specificpre'] #predictor layers
        
        self.dim_jointpre    = network_settings['dim_joint_pre']      #predictor hidden nodes
        self.num_layers_jointpre    = network_settings['num_layers_jointpre'] #predictor layers
        self.y_dim            = input_dims['y_dim']
        self.dropout   = network_settings['dropout'] 
        #self.reg_scale        = network_settings['reg_scale']   #regularization
        
        self.edge_per_node = network_settings['edge_per_node']
        self.ITERATION = network_settings['ITERATION']
        
        # module
        self.specific_encoder = {}
        self.specific_predictor = {}
        for m in range(self.M):
            self.specific_encoder[m] = specific_encoder(input_dim = self.x_dim_set[m], output_dim = 2*self.z_dim, num_layers = self.num_layers_enc, 
                                                        hidden_dim = self.dim_enc, dropout = self.dropout)
            self.add_module(f"specific_encoder_{m}", self.specific_encoder[m])
            self.specific_predictor[m] = predictor(input_dim = self.z_dim, output_dim = self.y_dim, num_layers = self.num_layers_specificpre, 
                                                        hidden_dim = self.dim_specificpre, dropout = self.dropout)
            self.add_module(f"specific_predictor{m}", self.specific_predictor[m])
            
        self.joint_encoder = joint_encoder()
        self.joint_predictor = predictor(input_dim = self.z_dim, output_dim = self.y_dim, num_layers = self.num_layers_jointpre, 
                                                        hidden_dim = self.dim_jointpre, dropout = self.dropout)
       
    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def forward(self, test_data):
        if torch.equal(test_data, self.train_x):
            print('you are test train_data')
            y_set, y, mu_set, logvar_set, mu_z, logvar_z = self.train_forward(self.train_x)
            return y
        #只构建测试集和训练集之间关联的图
        tr_x_set, tr_mask = restore_input_from_output(self.train_x, self.x_dim_set)
        trte_data = torch.cat([self.train_x, test_data], dim = 0) 
        x_set, mask = restore_input_from_output(trte_data, self.x_dim_set)
        adj_list = {}
        mu_set, logvar_set, u_set, z_set, y_set = {},{},{},{},{}
        idx_dict = {}
        idx_dict["tr"] = list(range(self.train_x.shape[0]))
        idx_dict["te"] = list(range(self.train_x.shape[0], (self.train_x.shape[0]+test_data.shape[0])))
        for m in range(self.M):
            adj_parameter_adaptive = cal_adj_mat_parameter(self.edge_per_node, tr_x_set[m], "cosine")
            adj = gen_test_adj_mat_tensor(x_set[m], idx_dict, adj_parameter_adaptive, "cosine")
            adj_list[m] = adj.float()
            #u_set[m] = self.feature_selective_layer[m])
            x_set[m] = x_set[m].to(torch.float32)
            mu_set[m], logvar_set[m] = self.specific_encoder[m](x_set[m], adj_list[m])
            z_set[m] = self.reparametrize(mu_set[m], logvar_set[m])
        mu_z, logvar_z = self.joint_encoder(mask, mu_set, logvar_set)
        z = self.reparametrize(mu_z, logvar_z)
    
        # specific omics predict
        for m in range(self.M):
            y_set[m] = self.specific_predictor[m](z_set[m])
    
        # joint omics predict
        y = self.joint_predictor(z)              
        y = y[-test_data.shape[0]:]   
        
        return y
    
    def train_forward(self, train_data):
        tr_x_set, mask = restore_input_from_output(train_data, self.x_dim_set)
        adj_list = {}
        mu_set, logvar_set, u_set, z_set, y_set = {},{},{},{},{}
        for m in range(self.M):
            adj_parameter_adaptive = cal_adj_mat_parameter(self.edge_per_node, tr_x_set[m], "cosine")
            adj = gen_adj_mat_tensor(tr_x_set[m], adj_parameter_adaptive, "cosine")
            adj_list[m] = adj.float()
            
            tr_x_set[m] = tr_x_set[m].to(torch.float32)
            mu_set[m], logvar_set[m] = self.specific_encoder[m](tr_x_set[m], adj_list[m])
            z_set[m] = self.reparametrize(mu_set[m], logvar_set[m])
        mu_z, logvar_z = self.joint_encoder(mask, mu_set, logvar_set)
        z = self.reparametrize(mu_z, logvar_z)
    
        # specific omics predict
        for m in range(self.M):
            y_set[m] = self.specific_predictor[m](z_set[m])
    
        # joint omics predict
        y = self.joint_predictor(z)        
            
        return y_set, y, mu_set, logvar_set, mu_z, logvar_z
    
    def div(self, x, y):
        return torch.div(x, y + 1e-8)
        
    def log(self, x):
        return torch.log(x + 1e-8)
        
    def loss_y(self, y_true, y_pre):
        tmp_loss = -torch.sum(y_true * self.log(y_pre), dim=-1)
        return tmp_loss
        
    def loss_function(self, mask, y_true, y_set, y, mu_set, logvar_set, mu_z, logvar_z, alpha, beta):
        ds = torch.distributions
        qz = ds.Normal(mu_z, torch.sqrt(torch.exp(logvar_z)))
        prior_z  = ds.Normal(0.0, 1.0)
        LOSS_KL = torch.mean(torch.sum(ds.kl_divergence(qz, prior_z), dim=-1))
        LOSS_PRE = torch.mean(self.loss_y(y_true, y))
            
        LOSS_JOINT = LOSS_PRE + beta*LOSS_KL
            
        LOSS_PRE_set  = []
        LOSS_KL_set = []
        for m in range(self.M):
            qz_set, prior_z_set = {},{}
            qz_set[m] = ds.Normal(mu_set[m], torch.sqrt(torch.exp(logvar_set[m])))
            prior_z_set[m] = ds.Normal(0.0, 1.0)
            tmp_pre = self.loss_y(y_true, y_set[m])
            tmp_kl = torch.sum(ds.kl_divergence(qz_set[m], prior_z_set[m]), dim=-1)
            
            LOSS_PRE_set += [self.div(torch.sum(mask[:,m]*tmp_pre), torch.sum(mask[:,m]))]
            LOSS_KL_set += [self.div(torch.sum(mask[:,m]*tmp_kl), torch.sum(mask[:,m]))]
            
        LOSS_PRE_set  = torch.stack(LOSS_PRE_set, dim=0)
        LOSS_KL_set = torch.stack(LOSS_KL_set, dim=0)
            
        LOSS_PRE_set_all = torch.sum(LOSS_PRE_set)
        LOSS_KL_set_all = torch.sum(LOSS_KL_set)
            
        LOSS_MARGINAL = LOSS_PRE_set_all + beta*LOSS_KL_set_all
            
        LOSS_TOTAL = LOSS_JOINT\
                    + alpha*(LOSS_MARGINAL)
                    
        return LOSS_TOTAL, LOSS_PRE, LOSS_KL, LOSS_PRE_set_all, LOSS_KL_set_all, LOSS_PRE_set, LOSS_KL_set
        
    def train_model(self, train_input, test_input, alpha, beta, l_rate, tr_Y_onehot, te_Y_onehot):
        
        self.train_x = train_input
        optimizer = optim.Adam(self.parameters(),lr=l_rate)
        #maxperform = 0
        maxf1 = 0
        for epoch in range(self.ITERATION):
            self.train()
            epoch_LOSS_TOTAL = 0
            epoch_LOSS_PRE = 0
            epoch_LOSS_KL = 0
            epoch_LOSS_PRE_set_all = 0
            epoch_LOSS_KL_set_all = 0
                
            optimizer.zero_grad()
            tr_mask = train_input[:, -self.M:]
            te_mask = test_input[:, -self.M:]
        
            y_set, y, mu_set, logvar_set, mu_z, logvar_z = self.train_forward(train_input)
            epoch_LOSS_TOTAL, epoch_LOSS_PRE, epoch_LOSS_KL, epoch_LOSS_PRE_set_all, epoch_LOSS_KL_set_all, LOSS_PRE_set, LOSS_KL_set = self.loss_function(tr_mask, tr_Y_onehot, y_set, y, mu_set, logvar_set, mu_z, logvar_z, alpha, beta)
            epoch_LOSS_TOTAL.backward()
            optimizer.step()
                
            if (epoch+1)%50 == 0:
                print("Train F1: {:.4f}".format(f1_score(tr_Y_onehot.cpu().argmax(1), self.train_forward(train_input)[1].cpu().argmax(1))))     
                print( "{:05d}: TRAIN| LT={:.3f} LP={:.3f} LKL={:.3f} LPS={:.3f} LKLS={:.3f} | ".format(
            epoch+1, epoch_LOSS_TOTAL, epoch_LOSS_PRE, epoch_LOSS_KL, epoch_LOSS_PRE_set_all, epoch_LOSS_KL_set_all))
                
                f1, acc, auc = self.predict_test(test_input, te_Y_onehot)
                #perform = acc + f1 + auc
                if f1 > maxf1 or (f1 == maxf1 and acc + auc > maxacc + maxauc) : 
                    maxf1 = f1
                    maxacc = acc
                    maxauc = auc
                    model_save = copy.deepcopy(self)
                    
        print('f1:',maxf1)
        print('acc:',maxacc)
        print('auc:',maxauc)
        return model_save, maxf1, maxacc, maxauc
    
    def predict_test(self, test_data, test_y):
        self.eval()
        with torch.no_grad():
            
            y = self(test_data)
    
            f1 = f1_score(test_y.cpu().argmax(1), y.cpu().argmax(1))
            acc = accuracy_score(test_y.cpu().argmax(1), y.cpu().argmax(1))
            auc = roc_auc_score(test_y.cpu(), y.cpu().numpy())
            print("Test F1: {:.4f}".format(f1))
            print("Test ACC: {:.4f}".format(acc))
            print("Test AUC: {:.4f}".format(auc))
            #print('single_pre_y:',torch.mean((torch.argmax(test_y, dim=1) == torch.argmax(y, dim=1)).float()))
        return f1, acc, auc

class specific_encoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        super(specific_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout
   
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gat3 = GraphAttentionLayer(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        x_1 = self.gc1(x, adj)
        x_1 = F.leaky_relu(x_1, 0.25)
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        
        x_2 = self.gc2(x_1, adj)
        x_2 = F.leaky_relu(x_2, 0.25)
        x_2 = F.dropout(x_2, self.dropout, training=self.training)
    
        x_3 = self.gat3(x_2, adj)
        self.output = F.leaky_relu(x_3, 0.25)
        self.mu = self.output[:, :self.output_dim//2] 
        self.logvar = self.output[:, self.output_dim//2:]
        return self.mu, self.logvar
        
class predictor(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, dropout):
        super(predictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.Softmax())
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.25))
            layers.append(nn.Dropout(p = dropout))
            
            for i in range(num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.25))
                layers.append(nn.Dropout(p = dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            layers.append(nn.Softmax())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.layers(x)
        return output
    
class joint_encoder(nn.Module):
    
    def __init__(self):
        super(joint_encoder, self).__init__()
        self.epsilon = 1e-8

    def div(self, x, y):
        return torch.div(x, y + self.epsilon)
    
    def log(self, x):
        return torch.log(x + self.epsilon)

    def forward(self, mask, mu_set, logvar_set):
        tmp = 1.
        for m in range(len(mu_set)):
            tmp += mask[:, m].reshape(-1, 1) * self.div(1., torch.exp(logvar_set[m]))
        joint_var = self.div(1., tmp)
        joint_logvar = self.log(joint_var)

        tmp = 0.
        for m in range(len(mu_set)):
            tmp += mask[:, m].reshape(-1, 1) * self.div(1., torch.exp(logvar_set[m])) * mu_set[m]
        joint_mu = joint_var * tmp

        return joint_mu.float(), joint_logvar.float()
    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
    
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        #print(adj.shape)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer 
    """
    def __init__(self, in_features, out_features, concat=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features   # 节点表示向量的输入特征维度
        self.out_features = out_features   # 节点表示向量的输出特征维度
        self.concat = concat   # 如果为true, 再进行elu激活
        
        # 定义可训练参数，即论文中的W和a
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))  
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # xavier初始化
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)   # xavier初始化
    
    def forward(self, inp, adj):
        h = torch.mm(inp, self.W)   # [N, out_features]
        N = h.size()[0]    # N 图的节点数
        
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 2*out_features]
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2), 0.25)
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）
        
        zero_vec = -1e12 * torch.ones_like(e)    # 将没有连接的边置为负无穷
        attention = torch.where(adj.to_dense() >0, e, zero_vec)   # [N, N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)    # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime 
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'        