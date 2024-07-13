from torch import nn
import torch
from egnn_pytorch import EGNN, EGNN_Network
from einops import rearrange
from openfold.utils.rigid_utils import Rigid


class EGNNScore(nn.Module):
    def __init__(self, model_conf, diffuser, residual=True, attention=False, normalize=False, tanh=False):
        super(EGNNScore, self).__init__()
        self._model_conf = model_conf

        self.diffuser = diffuser

        self.scale_pos = lambda x: x * model_conf.ipa.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / model_conf.ipa.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        # layers
        self.egnn = nn.ModuleList()
        for i in range(model_conf.n_layers):
            self.egnn.append(
                EGNN(
                    dim=model_conf.node_embed_size,
                    edge_dim=model_conf.edge_embed_size,
                    m_dim=model_conf.hidden_dim,
                    fourier_features=0,
                    num_nearest_neighbors=0,
                    dropout=model_conf.dropout,
                    norm_feats=True,
                    norm_coors=True,
                    update_feats=True,
                    update_coors=False,
                    valid_radius=model_conf.max_radius,
                    m_pool_method='sum',
                    soft_edge=False,
                    coor_weights_clamp_value=None
                )
            )
        self.final_layer = nn.Linear(model_conf.node_embed_size, 3)

    def forward(self, h, x, edge_attr, input_feats):
        batch_size, num_atoms = input_feats['atom_types'].shape

        # obtain node embeddings 
        x = self.scale_pos(x)
        for i in range(len(self.egnn)):
            h, x = self.egnn[i](feats=h, coors=x, edges=edge_attr)          # [B, N, D]

        # output layer
        num_bb_atoms = input_feats['num_bb_atoms'][0]
        start_idx = 0
        rot_pred = [] 
        for i, num_atom in enumerate(num_bb_atoms):
            rot_pred.append(h[:, start_idx:start_idx+num_atom, :].mean(dim=1))
            start_idx += num_atom
        rot_pred = torch.stack(rot_pred, dim=1)
        rot_pred = self.final_layer(rot_pred)                               # [B, M, 3]

        #  Compute scores
        rot_score = self.diffuser._so3_diffuser.torch_score(rot_pred, input_feats['t'])

        model_out = {
            'rot_pred': rot_pred,
            'rot_score': rot_score,
        }
        
        return model_out
