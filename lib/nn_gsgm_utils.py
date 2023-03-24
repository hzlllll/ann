import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SGConv



# 1.定义SAGEConv层
class SAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SAGEConv, self).__init__()
        # 线性层
        self.w1 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        self.w2 = pyg_nn.dense.linear.Linear(in_channels, out_channels, weight_initializer='glorot', bias=bias)
        
    def forward(self, x, edge_index):
        # 对自身节点进行特征映射
        wh_1 = self.w1(x)
        
        # 获取邻居特征
        x_j = x[edge_index[0]]
        
        # 对邻居节点进行特征映射
        wh_2 = self.w2(x_j)
        
        # 对邻居节点进行聚合
        wh_2 = scatter(src=wh_2, index=edge_index[1], dim=0, reduce='mean') # max聚合操作 [num_nodes, feature_size]
        
        return wh_1 + wh_2
        

# 2.定义GraphSAGE网络
class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, num_hidden_layer ,num_classes, is_class=False,residual=False):
        """
        num_node_features: feature数目的矩阵
        num_classes: 最后类别数目
        """
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels=num_node_features,
                              out_channels=num_hidden_layer)
        self.conv2 = SAGEConv(in_channels=num_hidden_layer,
                              out_channels=num_classes)
        self.is_class = is_class
        self.residual = residual

    def forward(self, inp, edge_index):
        """
        inp:  Node数目*feature数目的矩阵
        edge_index: 2*edge数的矩阵  每一列代表边的编号
        """
        
        x = self.conv1(inp, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if self.is_class:
            return F.log_softmax(x, dim=1)
        else:
            if self.residual:
                x += inp
            return x

class SGC(nn.Module):
    def __init__(self, in_feats, out_feats, k, is_class=False, residual=False):
        super(SGC, self).__init__()
        self.conv = SGConv(in_feats, out_feats, K=k, cached=True)
        self.residual = residual
        self.is_class = is_class

    def forward(self, inp, edge_index):
        x = self.conv(inp, edge_index)
        if self.is_class:
            return F.log_softmax(x, dim=1)
        else:
            if self.residual:
                x += inp
            return x

if __name__=="__main__":
    # 3.加载Cora数据集
    dataset = Planetoid(root='/home/huzhilong/learning-to-route-online/lib/data/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
    epochs = 200 # 学习轮数
    lr = 0.0003 # 学习率
    num_node_features = dataset.num_node_features # 每个节点的特征数
    num_classes = dataset.num_classes # 每个节点的类别数
    data = dataset[0].to(device) # Cora的一张图

    # 4.定义模型
    model = GraphSAGE(num_node_features=num_node_features,num_hidden_layer=16 ,num_classes=num_classes,is_class=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器
    loss_function = nn.NLLLoss() # 损失函数

    # 训练模式
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x,data.edge_index)
        
        loss = loss_function(pred[data.train_mask], data.y[data.train_mask]) # 损失
        correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item() # epoch正确分类数目
        acc_train = correct_count_train / data.train_mask.sum().item() # epoch训练精度
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print("【EPOCH: 】%s" % str(epoch + 1))
            print('训练损失为：{:.4f}'.format(loss.item()), '训练精度为：{:.4f}'.format(acc_train))

    print('【Finished Training！】')

    # 模型验证
    model.eval()
    pred = model(data)

    # 训练集（使用了掩码）
    correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item()
    acc_train = correct_count_train / data.train_mask.sum().item()
    loss_train = loss_function(pred[data.train_mask], data.y[data.train_mask]).item()

    # 测试集
    correct_count_test = pred.argmax(axis=1)[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc_test = correct_count_test / data.test_mask.sum().item()
    loss_test = loss_function(pred[data.test_mask], data.y[data.test_mask]).item()

    print('Train Accuracy: {:.4f}'.format(acc_train), 'Train Loss: {:.4f}'.format(loss_train))
    print('Test  Accuracy: {:.4f}'.format(acc_test), 'Test  Loss: {:.4f}'.format(loss_test))
