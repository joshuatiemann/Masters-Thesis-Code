import torch
from tensordict import NonTensorData
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GraphConv
from torch_geometric.data import Batch
    
def mlp_actor(num_cells=64, action_dim=8, hidden_layer=2, device="cpu"):
    return nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * action_dim, device=device),
        NormalParamExtractor(),
    )
    
class joint_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_lin = nn.Sequential(
            Linear(2, 64)
        )
        self.torso_lin = nn.Sequential(
            Linear(11, 64)
        )
        self.propagation_model = Sequential("x, edge_index", [
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            #(GraphConv(64, 64), "x, edge_index -> x"), used for T=3
            #nn.Tanh(),
        ])
        self.output_layer = nn.ModuleList([
            Linear(64, 2) for _ in range(8)
        ])

    def forward(self, data):
        if isinstance(data, NonTensorData):
            data = data.data
        
        # Handle NonTensorStack
        if hasattr(data, "tolist") and not isinstance(data, (list, tuple, torch.Tensor)):
            try:
                data_list = data.tolist()
                if isinstance(data_list, list):
                    data = data_list
            except Exception:
                pass

        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
            
        x = batch.x
        edge_index = batch.edge_index
        
        num_nodes = x.size(0)
        # Assuming 9 nodes per graph (1 torso, 8 joints)
        is_torso = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        is_torso[::9] = True
        is_joint = ~is_torso
        
        h_torso = self.torso_lin(x[is_torso])
        # Joint features are first 2 columns
        h_joint = self.joint_lin(x[is_joint][:, :2])
        
        h = torch.empty((num_nodes, 64), dtype=h_torso.dtype, device=x.device)
        h[is_torso] = h_torso
        h[is_joint] = h_joint
        
        h = self.propagation_model(h, edge_index)
        
        joints = h[is_joint]
        batch_size = num_nodes // 9
        
        joints = joints.view(batch_size, 8, -1)
        
        outputs = []
        layer_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(8):
            outputs.append(self.output_layer[layer_indices[i]](joints[:, i, :]))
            
        out = torch.stack(outputs, dim=1)
        
        loc, scale = NormalParamExtractor()(out)
        
        if batch_size == 1:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)
        
        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        return loc, scale
    
class leg_actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_lin = nn.Sequential(
            Linear(4, 64)
        )
        self.torso_lin = nn.Sequential(
            Linear(11, 64)
        )
        self.propagation_model = Sequential("x, edge_index", [
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            #(GraphConv(64, 64), "x, edge_index -> x"), used for T=3
            #nn.Tanh(),
        ])

        self.output_layer = nn.ModuleList([
            Linear(64, 4) for _ in range(4) 
        ])

    def forward(self, data):
        if isinstance(data, NonTensorData):
            data = data.data

        # Handle NonTensorStack
        if hasattr(data, "tolist") and not isinstance(data, (list, tuple, torch.Tensor)):
            try:
                data_list = data.tolist()
                if isinstance(data_list, list):
                    data = data_list
            except Exception:
                pass

        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
            
        x = batch.x
        edge_index = batch.edge_index
        
        num_nodes = x.size(0)
        # Assuming 5 nodes per graph (1 torso, 4 legs)
        is_torso = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        is_torso[::5] = True
        is_joint = ~is_torso
        
        h_torso = self.torso_lin(x[is_torso])
        # leg features are first 4 columns
        h_joint = self.joint_lin(x[is_joint][:, :4])
        
        h = torch.empty((num_nodes, 64), dtype=h_torso.dtype, device=x.device)
        h[is_torso] = h_torso
        h[is_joint] = h_joint
        
        h = self.propagation_model(h, edge_index)
        
        joints = h[is_joint]
        batch_size = num_nodes // 5
        
        joints = joints.view(batch_size, 4, -1)
        
        outputs = []
        layer_indices = [0, 1, 2, 3]
        for i in range(4):
            outputs.append(self.output_layer[layer_indices[i]](joints[:, i, :]))
            
        out = torch.stack(outputs, dim=1)
        
        loc, scale = NormalParamExtractor()(out)
        
        # Flatten: (Batch, 4 legs, 2 actions) -> (Batch, 8 actions)
        loc = loc.reshape(batch_size, -1)
        scale = scale.reshape(batch_size, -1)
        
        if batch_size == 1:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)
        return loc, scale
