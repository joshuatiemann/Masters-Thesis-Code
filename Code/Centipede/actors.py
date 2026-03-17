import torch
from tensordict import NonTensorData
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torch_geometric.nn import Linear, Sequential, GCNConv, GraphConv, MessagePassing, global_mean_pool, HGTConv, HeteroConv, to_hetero, GATv2Conv, GatedGraphConv
from torch_geometric.data import Data, Batch
    
def mlp_actor(num_cells=128, action_dim=16, device="cpu"):
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
        self.joint_lin = Linear(2, 64)
        self.torso_lin = nn.LazyLinear(64)
        self.propagation_model = Sequential("x, edge_index", [
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
        ])
        # Joint output heads: 1 action per joint → Linear(64, 2) → NormalParamExtractor → 1 loc + 1 scale
        # Support up to 20 joints (5 segments × 4 joints = tenpede)
        self.joint_output_layer = nn.ModuleList([
            Linear(64, 2) for _ in range(20)
        ])
        # Non-root torso output heads: 2 actions per torso (connector_rl + connector_ud)
        # Linear(64, 4) → NormalParamExtractor → 2 loc + 2 scale
        # Support up to 4 non-root torsos (tenpede = 5 segments)
        self.torso_output_layer = nn.ModuleList([
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
        
        # Determine batch size
        if hasattr(batch, 'num_graphs'):
            batch_size = batch.num_graphs
        else:
            batch_size = 1

        num_nodes = x.size(0)
        nodes_per_graph = num_nodes // batch_size
        nodes_per_segment = 5  # 1 torso + 4 joints (L_hip, L_ankle, R_hip, R_ankle)
        num_segments = nodes_per_graph // nodes_per_segment
        
        # --- Identify node types ---
        # Within each graph, torso nodes are at indices 0, 5, 10, ... (every nodes_per_segment)
        is_torso = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        for s in range(num_segments):
            is_torso[s * nodes_per_segment::nodes_per_graph] = True
        is_joint = ~is_torso
        
        # --- Encode ---
        # All torsos (root + non-root) share torso_lin; root is padded to 17 with zeros
        h_torso = self.torso_lin(x[is_torso])
        # Joint features are first 2 columns of the padded 17-dim vector
        h_joint = self.joint_lin(x[is_joint][:, :2])
        
        h = torch.empty((num_nodes, 64), dtype=h_torso.dtype, device=x.device)
        h[is_torso] = h_torso
        h[is_joint] = h_joint
        
        # --- Message passing ---
        h = self.propagation_model(h, edge_index)
        
        # --- Action output (match XML actuator order) ---
        # XML actuator order per segment:
        #   Root:     hip_left, ankle_left, hip_right, ankle_right
        #   Non-root: conn_rl, conn_ud, hip_left, ankle_left, hip_right, ankle_right
        # Graph node order per segment already stores joints as
        # [Torso, L_Hip, L_Ankle, R_Hip, R_Ankle], so we simply emit
        # connectors first (non-root) and then iterate joints left→right.
        
        h_per_graph = h.view(batch_size, nodes_per_graph, -1)
        
        all_actions = []
        joint_counter = 0
        torso_counter = 0
        
        for s in range(num_segments):
            base = s * nodes_per_segment
            
            # Non-root torso: emit connector actions BEFORE legs (XML order)
            if s > 0:
                torso_feat = h_per_graph[:, base, :]
                torso_out = self.torso_output_layer[torso_counter](torso_feat)
                all_actions.append(torso_out.view(batch_size, 2, 2))
                torso_counter += 1

            # Joint actions in XML order: L_Hip, L_Ankle, R_Hip, R_Ankle
            # Graph node layout per segment: [Torso(+0), L_Hip(+1), L_Ankle(+2), R_Hip(+3), R_Ankle(+4)]
            for offset in [1, 2, 3, 4]:
                j_feat = h_per_graph[:, base + offset, :]
                j_out = self.joint_output_layer[joint_counter](j_feat)
                all_actions.append(j_out.unsqueeze(1))
                joint_counter += 1

        # [batch, total_actions, 2] where total_actions = 16 for sixpede
        out = torch.cat(all_actions, dim=1)
        
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
        # Leg node input: 4 dims (hip_qpos, hip_qvel, ankle_qpos, ankle_qvel), padded to 17
        # Torso node input: up to 17 dims (body_state + connector)
        # Both are padded to max_dim=17 in the transform, but leg nodes only use first 4 dims
        self.leg_lin = nn.Sequential(Linear(4, 64),    # leg nodes (slice to 4 dims)
                                     nn.Tanh(),
        )
        self.torso_lin = nn.Sequential(nn.LazyLinear(64),  # torso nodes (lazy: handles root=13pad vs non-root=17)
                                       nn.Tanh(),
        )

        self.propagation_model = Sequential("x, edge_index", [
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
            (GraphConv(64, 64), "x, edge_index -> x"),
            nn.Tanh(),
        ])

        # Leg output head: 2 actions (hip, ankle) -> 4 values (2 loc + 2 scale)
        # Support up to 10 legs (5 segments × 2 legs)
        self.leg_output_layer = nn.ModuleList([
            Linear(64, 4) for _ in range(10)
        ])

        # Non-root torso output head: 2 connector actions -> 4 values (2 loc + 2 scale)
        # Support up to 4 non-root torsos
        self.torso_output_layer = nn.ModuleList([
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

        if hasattr(batch, 'num_graphs'):
            batch_size = batch.num_graphs
        else:
            batch_size = 1

        num_nodes = x.size(0)
        nodes_per_graph = num_nodes // batch_size
        nodes_per_segment = 3      # 1 torso + 2 legs (L, R)
        num_segments = nodes_per_graph // nodes_per_segment

        # --- Identify node types ---
        # Torso nodes: indices 0, 3, 6, ... (every nodes_per_segment) within each graph
        is_torso = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        for s in range(num_segments):
            # global index for torso of segment s in EACH graph
            is_torso[s * nodes_per_segment::nodes_per_graph] = True
        is_leg = ~is_torso

        # --- Encode ---
        h_torso = self.torso_lin(x[is_torso])
        h_leg = self.leg_lin(x[is_leg][:, :4])

        h = torch.empty((num_nodes, 64), dtype=h_torso.dtype, device=x.device)
        h[is_torso] = h_torso
        h[is_leg] = h_leg

        # --- Message passing ---
        h = self.propagation_model(h, edge_index)

        # --- Action output ---
        h_per_graph = h.view(batch_size, nodes_per_graph, -1)

        all_actions = []
        leg_counter = 0
        torso_counter = 0

        for s in range(num_segments):
            base = s * nodes_per_segment  # offset within the nodes_per_graph dimension

            # Non-root connectors must precede leg joints in the action vector.
            if s > 0:
                torso_feat = h_per_graph[:, base, :]
                torso_out = self.torso_output_layer[torso_counter](torso_feat)
                all_actions.append(torso_out.view(batch_size, 2, 2))
                torso_counter += 1

            # L_Leg actions: hip_left, ankle_left  (offsets [1] = L_Leg node)
            l_leg_feat = h_per_graph[:, base + 1, :]           # [batch, 64]
            l_leg_out = self.leg_output_layer[leg_counter](l_leg_feat)   # [batch, 4]
            all_actions.append(l_leg_out.view(batch_size, 2, 2))
            leg_counter += 1

            # R_Leg actions: hip_right, ankle_right  (offset [2] = R_Leg node)
            r_leg_feat = h_per_graph[:, base + 2, :]           # [batch, 64]
            r_leg_out = self.leg_output_layer[leg_counter](r_leg_feat)   # [batch, 4]
            all_actions.append(r_leg_out.view(batch_size, 2, 2))
            leg_counter += 1

        # [batch, total_actions, 2]
        out = torch.cat(all_actions, dim=1)

        loc, scale = NormalParamExtractor()(out)

        if batch_size == 1:
            loc = loc.squeeze(0)
            scale = scale.squeeze(0)

        loc = loc.squeeze(-1)
        scale = scale.squeeze(-1)
        return loc, scale
