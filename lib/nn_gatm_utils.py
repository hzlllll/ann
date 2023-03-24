import torch
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F


class GATM(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_layer, nhead, num_classes, is_class=False, residual=False):
        super(GATM,self).__init__()
        self.gat1 = GATConv(num_node_features,num_hidden_layer,nhead,dropout=0.6)
        self.gat2 = GATConv(num_hidden_layer*nhead,num_classes,1,dropout=0.6)
        self.is_class = is_class
        self.residual = residual

    def forward(self,inp,edge_index):
        x = self.gat1(inp,edge_index)
        x = self.gat2(x,edge_index)
        if self.is_class:
            return F.log_softmax(x, dim=1)
        else:
            if self.residual:
                x += inp
            return x
        
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_layer, nhead, num_classes, is_class=False, residual=False):
        super(GAT,self).__init__()
        self.gat1 = GATConv(num_node_features,num_hidden_layer,nhead,dropout=0.6)
        self.gat2 = GATConv(num_hidden_layer*nhead,num_classes,1,dropout=0.6)
        self.is_class = is_class
        self.residual = residual

    def forward(self,inp,edge_index):
        x = self.gat1(inp,edge_index)
        x = self.gat2(x,edge_index)
        if self.is_class:
            return F.log_softmax(x, dim=1)
        else:
            if self.residual:
                x += inp
            return x

if __name__=="__main__":
    dataset = Planetoid(root='Cora', name='Cora')
    x=dataset[0].x
    edge_index=dataset[0].edge_index

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATM().to(device)
    data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct/int(data.test_mask.sum())
    print('Accuracy:{:.4f}'.format(acc))

