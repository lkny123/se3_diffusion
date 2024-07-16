"""Score network module."""
import torch
import math
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from data import utils as du
from data import all_atom
from model import ipa_pytorch
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

        # Embedding dimensions 
        t_embed_dim = self._embed_conf.time_embed_dim
        bb_embed_dim = self._embed_conf.bb_embed_dim

        # Compute input dimensions
        node_embed_in = (t_embed_dim + 1) + bb_embed_dim
        edge_embed_in = 2 * (t_embed_dim + 1)
        if self._embed_conf.embed_self_conditioning:
            edge_embed_in += self._embed_conf.num_bins

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_in, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_embed_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding,
            embedding_dim=t_embed_dim
        )
        # self.fingerprint_embedder = nn.Linear(self._embed_conf.fingerprint_dim, bb_embed_dim, bias=False)


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
            bb_emb,
            t,
            fixed_mask,
            self_conditioning_ca,
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
        num_batch, num_bbs, _ = bb_emb.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        fixed_mask = fixed_mask[..., None]
        time_emb = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_bbs, 1))     # [B, N, D]
        time_emb = torch.cat([time_emb, fixed_mask], dim=-1)            # [B, N, D + 1]
        node_feats = [time_emb]
        pair_feats = [self._cross_concat(time_emb, num_batch, num_bbs)] # [B, N^2, 2*D]

        # Building block embeddings
        node_feats.append(bb_emb)                                       # [B, N, D]

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_bbs**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_bbs, num_bbs, -1])
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):

    def __init__(self, model_conf, diffuser):
        super(ScoreNetwork, self).__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats):
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats['res_mask'].type(torch.float32)  # [B, N]
        fixed_mask = input_feats['fixed_mask'].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        # Initial node and edge embeddings
        init_node_embed, init_edge_embed = self.embedding_layer(
            bb_emb=input_feats['bb_emb'],
            t=input_feats['t'],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats['sc_ca_t']
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]
        
        # Run main network 
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        pred_out = {
            'psi': None,
            'rot_score': model_out['rot_score'],
            'trans_score': model_out['trans_score'],
        }
        rigids_pred = model_out['final_rigids']
        pred_out['rigids'] = rigids_pred.to_tensor_7()

        return pred_out
    