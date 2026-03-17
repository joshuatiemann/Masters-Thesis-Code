from torchrl.envs.transforms import ObservationTransform
import torch
from torch_geometric.data import Data, HeteroData
from tensordict import NonTensorData, TensorDictBase
from torchrl.data.tensor_specs import Bounded
from torchrl.envs.transforms.transforms import _apply_to_composite
from torch_geometric.utils import dense_to_sparse
from torch_geometric import transforms as T


class Notransform(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        return obs

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec


class joint_centipede_graph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        data = Data()
        
        node_features = []
        edges = []
        
        # Observation layout from _get_obs():
        #   Segment 0 (root): root.qpos(7) + root.qvel(6) = 13, then 4 joints × 2 = 8  → 21 dims
        #   Segment i>0:      xpos(3)+xquat(4)+linvel(3)+angvel(3) + conn_rl(2)+conn_ud(2) = 17, then 4 joints × 2 = 8  → 25 dims
        # Connector features are merged into non-root torso nodes (no separate connector node).
        #
        # Graph structure per segment: [Torso, L_Hip, L_Ankle, R_Hip, R_Ankle] = 5 nodes
        # Sixpede example: 3 segments → 15 nodes, 16 actuators
        
        dim_root_torso = 13       # qpos(7) + qvel(6)
        dim_body_state = 13       # xpos(3) + xquat(4) + linvel(3) + angvel(3)
        dim_connector = 4         # conn_rl_qpos(1) + conn_rl_qvel(1) + conn_ud_qpos(1) + conn_ud_qvel(1)
        dim_joint = 2             # qpos(1) + qvel(1)
        max_dim = dim_body_state + dim_connector  # 17 (pad all nodes to this)
        
        ptr = 0
        node_idx = 0
        prev_torso_idx = None
        total_len = obs.shape[0]
        segment = 0

        def pad_to(feat, target_len):
            if feat.shape[0] < target_len:
                padding = torch.zeros(target_len - feat.shape[0], device=obs.device, dtype=obs.dtype)
                return torch.cat([feat, padding], dim=0)
            return feat

        while ptr < total_len:
            # --- Torso node ---
            if segment == 0:
                # Root torso: 13 dims (qpos 7 + qvel 6), padded to 17
                if ptr + dim_root_torso > total_len:
                    break
                torso_feat = pad_to(obs[ptr:ptr + dim_root_torso], max_dim)
                ptr += dim_root_torso
            else:
                # Non-root torso: 13 body state + 4 connector = 17 dims
                torso_dim = dim_body_state + dim_connector
                if ptr + torso_dim > total_len:
                    break
                torso_feat = obs[ptr:ptr + torso_dim]
                ptr += torso_dim
            
            torso_idx = node_idx
            node_features.append(torso_feat)
            node_idx += 1
            
            # Edge to previous torso (inter-segment connection)
            if prev_torso_idx is not None:
                edges.append([prev_torso_idx, torso_idx])
                edges.append([torso_idx, prev_torso_idx])
            
            # --- 4 joint nodes: L_Hip, L_Ankle, R_Hip, R_Ankle ---
            # Obs order from _get_obs(): left_hip, left_ankle, right_hip, right_ankle
            joint_idxs = []
            for _ in range(4):
                if ptr + dim_joint > total_len:
                    break
                j_feat = pad_to(obs[ptr:ptr + dim_joint], max_dim)
                ptr += dim_joint
                j_idx = node_idx
                node_features.append(j_feat)
                joint_idxs.append(j_idx)
                node_idx += 1
            
            if len(joint_idxs) == 4:
                l_hip, l_ankle, r_hip, r_ankle = joint_idxs
                # Torso <-> Hips
                edges.extend([[torso_idx, l_hip], [l_hip, torso_idx]])
                edges.extend([[torso_idx, r_hip], [r_hip, torso_idx]])
                # Hip <-> Ankle
                edges.extend([[l_hip, l_ankle], [l_ankle, l_hip]])
                edges.extend([[r_hip, r_ankle], [r_ankle, r_hip]])
            
            prev_torso_idx = torso_idx
            segment += 1
        
        # Construct PyG Data
        data.x = torch.stack(node_features)
        
        if len(edges) > 0:
            data.edge_index = torch.tensor(edges, dtype=torch.long, device=obs.device).t().contiguous()
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long, device=obs.device)

        return NonTensorData(data)
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec
    

class leg_centipede_graph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        data = Data()

        # Observation layout from _get_obs():
        #   Segment 0 (root): root.qpos(7) + root.qvel(6) = 13,
        #                     then left_hip(2) + left_ankle(2) + right_hip(2) + right_ankle(2) = 8 → 21 dims
        #   Segment i>0:      xpos(3)+xquat(4)+linvel(3)+angvel(3) = 13,
        #                     conn_rl(2)+conn_ud(2) = 4, → 17 body+conn dims
        #                     then left_hip(2) + left_ankle(2) + right_hip(2) + right_ankle(2) = 8 → 25 dims
        #
        # Graph structure per segment: [Torso, L_Leg, R_Leg] = 3 nodes
        # Leg node features: [hip_qpos, hip_qvel, ankle_qpos, ankle_qvel] = 4 dims, padded to max_dim
        # Torso node features: body state (+ connector for non-root), padded to max_dim

        dim_root_torso = 13       # qpos(7) + qvel(6)
        dim_body_state = 13       # xpos(3) + xquat(4) + linvel(3) + angvel(3)
        dim_connector = 4         # conn_rl_qpos + conn_rl_qvel + conn_ud_qpos + conn_ud_qvel
        dim_joint = 2             # qpos + qvel per joint
        dim_leg = dim_joint * 2   # hip(2) + ankle(2) = 4
        max_dim = dim_body_state + dim_connector  # 17: pad all nodes to this

        def pad_to(feat, target_len):
            if feat.shape[0] < target_len:
                padding = torch.zeros(target_len - feat.shape[0], device=obs.device, dtype=obs.dtype)
                return torch.cat([feat, padding], dim=0)
            return feat

        node_features = []
        edges = []
        ptr = 0
        node_idx = 0
        prev_torso_idx = None
        segment = 0
        total_len = obs.shape[0]

        while ptr < total_len:
            # --- Torso node ---
            if segment == 0:
                if ptr + dim_root_torso > total_len:
                    break
                torso_feat = pad_to(obs[ptr:ptr + dim_root_torso], max_dim)
                ptr += dim_root_torso
            else:
                torso_dim = dim_body_state + dim_connector  # 17
                if ptr + torso_dim > total_len:
                    break
                torso_feat = obs[ptr:ptr + torso_dim]
                ptr += torso_dim

            torso_idx = node_idx
            node_features.append(torso_feat)
            node_idx += 1

            # Inter-segment edge (torso chain)
            if prev_torso_idx is not None:
                edges.append([prev_torso_idx, torso_idx])
                edges.append([torso_idx, prev_torso_idx])

            # --- 2 leg nodes: L_Leg, R_Leg ---
            # Obs order from _get_obs(): left_hip(2), left_ankle(2), right_hip(2), right_ankle(2)
            # So L_Leg = obs[ptr:ptr+4], R_Leg = obs[ptr+4:ptr+8]
            if ptr + 2 * dim_leg > total_len:
                break

            l_leg_feat = pad_to(obs[ptr:ptr + dim_leg], max_dim)           # left_hip + left_ankle
            ptr += dim_leg
            r_leg_feat = pad_to(obs[ptr:ptr + dim_leg], max_dim)           # right_hip + right_ankle
            ptr += dim_leg

            l_leg_idx = node_idx
            node_features.append(l_leg_feat)
            node_idx += 1

            r_leg_idx = node_idx
            node_features.append(r_leg_feat)
            node_idx += 1

            # Torso <-> Legs (bidirectional)
            edges.extend([[torso_idx, l_leg_idx], [l_leg_idx, torso_idx]])
            edges.extend([[torso_idx, r_leg_idx], [r_leg_idx, torso_idx]])

            prev_torso_idx = torso_idx
            segment += 1

        data.x = torch.stack(node_features)

        if len(edges) > 0:
            data.edge_index = torch.tensor(edges, dtype=torch.long, device=obs.device).t().contiguous()
        else:
            data.edge_index = torch.empty((2, 0), dtype=torch.long, device=obs.device)

        return NonTensorData(data)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec