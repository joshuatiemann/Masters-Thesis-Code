from torchrl.envs.transforms import ObservationTransform
import torch
from torch_geometric.data import Data
from tensordict import NonTensorData, TensorDictBase
try: from torchrl.data.tensor_specs import Bounded
except ImportError:
    from torchrl.data.tensor_specs import BoundedTensorSpec as Bounded
from torchrl.envs.transforms.transforms import _apply_to_composite


class Notransform(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        return obs

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec


#corresponds to the baseline graph, where each actuator is represented by a single node + torso node, edges resulting from body connectivity
#joint nodes, padded with 0 only for data construction, will be ignored in forward pass
class joint_graph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        data = Data()
        data.x = torch.tensor([[obs[0], obs[1], obs[2], obs[3], obs[4], obs[13], obs[14], obs[15], obs[16], obs[17], obs[18]],
                                [obs[5], obs[19],0,0,0,0,0,0,0,0,0],
                                [obs[6], obs[20],0,0,0,0,0,0,0,0,0],
                                [obs[7], obs[21],0,0,0,0,0,0,0,0,0],
                                [obs[8], obs[22],0,0,0,0,0,0,0,0,0],
                                [obs[9], obs[23],0,0,0,0,0,0,0,0,0],
                                [obs[10], obs[24],0,0,0,0,0,0,0,0,0],
                                [obs[11], obs[25],0,0,0,0,0,0,0,0,0],
                                [obs[12], obs[26],0,0,0,0,0,0,0,0,0],], dtype=torch.float)
        
        data.edge_index = torch.tensor([[0, 0, 0, 0, 1, 3, 5, 7, 1, 3, 5, 7, 2, 4, 6, 8], [1, 3, 5, 7, 2, 4, 6, 8, 0, 0, 0, 0, 1, 3, 5, 7]], dtype=torch.long)
        return NonTensorData(data)
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec
    
class leg_graph(ObservationTransform):
    def _apply_transform(self, obs: torch.Tensor) -> NonTensorData:
        data = Data()
        data.x = torch.tensor([
            [obs[0], obs[1], obs[2], obs[3], obs[4], obs[13], obs[14], obs[15], obs[16], obs[17], obs[18]],
            [obs[5], obs[19], obs[6], obs[20], 0,0,0,0,0,0,0],
            [obs[7], obs[21], obs[8], obs[22], 0,0,0,0,0,0,0],
            [obs[9], obs[23], obs[10], obs[24], 0,0,0,0,0,0,0],
            [obs[11], obs[25], obs[12], obs[26], 0,0,0,0,0,0,0],],
            dtype=torch.float)
        data.edge_index = torch.tensor([[0, 0, 0, 0, 1, 2, 3, 4],[1, 2, 3, 4, 0, 0, 0, 0]],
                                       dtype=torch.long)
        return NonTensorData(data)

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return observation_spec
