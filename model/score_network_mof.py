"""Score network module."""
import torch
import math
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from data import utils as du
from data import all_atom
from model import egnn
import functools as fn

from openfold.utils.rigid_utils import Rigid

Tensor = torch.Tensor


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
    

class Embedder(nn.Module):

    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size 
        edge_in = (t_embed_size) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=self._embed_conf.index_embed_size
        )
        self.atom_embedder = nn.Embedding(model_conf.max_atoms, index_embed_size)

        # RBF for embedding edge distances
        self.edge_distance_embedder = GaussianSmearing(0.0, model_conf.rbf_max, index_embed_size)

    def _cross_concat(self, feats_1d, num_batch, num_atoms):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_atoms, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_atoms, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_atoms**2, -1])

    
    def _pairwise_distances(self, x):
        """        
        Args:
            x (torch.Tensor): Tensor of shape [B, N, 3], representing the coordinates.
        """

        dists = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # Shape: [B, N, N]
        return dists

    def forward(
            self,
            *,
            atom_types,
            t,
            x_t
        ):
        """Embeds a set of inputs

        Args:
            atom_types: [B, N] Atomic number of each atom
            t: [B,] Sampled t in [0, 1].
            x_t: [B, N, 3] Noised coordinates of all atoms 

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_atoms = atom_types.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        time_emb = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_atoms, 1)) # [B, N, D]
        node_feats = [time_emb]
        pair_feats = [self._cross_concat(time_emb, num_batch, num_atoms)] # [B, N^2, 2*D]

        # Atom type embeddings
        node_feats.append(self.atom_embedder(atom_types - 1))

        # Edge radius embeddings
        pairwise_dist = self._pairwise_distances(x_t) # [B, N, N]
        edge_feats = self.edge_distance_embedder(pairwise_dist) # [B*N*N, D]
        pair_feats.append(rearrange(edge_feats, '(B N1 N2) D -> B (N1 N2) D', B=num_batch, N1=num_atoms, N2=num_atoms))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_atoms, num_atoms, -1])
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = egnn.EGNNScore(
            model_conf=model_conf,
            diffuser=diffuser,
        )

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)
                - t: [B,]
                - x_t: [B, N, 3]
                - rot_score: [B, M, 3]
                - rot_score_scaling: [B,]
                - atom_types: [B, N]
                - num_bb_atoms: [B, M]

        Returns:
            model_out: dictionary of model outputs.
        """
        batch_size, num_atoms = input_feats['atom_types'].shape

        # Initial node and edge embeddings
        node_embed, edge_embed = self.embedding_layer(
            atom_types=input_feats['atom_types'],
            t=input_feats['t'],
            x_t=input_feats['x_t'],
        )
        
        # Run main network 
        model_out = self.score_model(node_embed, input_feats['x_t'], edge_embed, input_feats)

        pred_out = {
            'rot_pred': model_out['rot_pred'],
            'rot_score': model_out['rot_score'],
            'trans_pred': None
            'trans_score': None,
        }

        return pred_out
    
    def compute_coords(self, input_feats, pred_rigid):
        """
        Predicts coordinates by applying rotation and translation to backbone coordinates.

        Args:
            x_bb (torch.Tensor): [N, 3], backbone atom coordinates
            num_bb_atoms (torch.Tensor): [M],  containing the number of backbone atoms in each segment.
            pred_rigid (Rigid): [B, M], predicted rototranslations for each bb and batch

        Returns:
            torch.Tensor: [B, N, 3], predicted atom coordinates
        """
        x_bb = input_feats['x_bb'][0]  # [N, 3]
        num_bb_atoms = input_feats['num_bb_atoms'][0]

        start_idx = 0
        x_pred = []

        for i, num_bb_atom in enumerate(num_bb_atoms):
            bb_coord = x_bb[start_idx:start_idx + num_bb_atom, :]  # [num_bb_atom, 3]
            rot_mats = pred_rigid[:, i]._rots.get_rot_mats()  # [batch_size, 3, 3]
            trans = pred_rigid[:, i]._trans[:, None, :]  # [batch_size, 1, 3]
            
            # Apply rotation and translation
            pred_coord = torch.einsum('bij,nj->bni', rot_mats, bb_coord) + trans  # [batch_size, num_bb_atom, 3]
            
            x_pred.append(pred_coord)
            start_idx += num_bb_atom

        x_pred = torch.cat(x_pred, dim=1)  

        return x_pred
