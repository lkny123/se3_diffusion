from torch import nn
import torch
from egnn_pytorch import EGNN_Sparse
from einops import rearrange
from torch_cluster import radius_graph


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord += agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNNScore(nn.Module):
    def __init__(self, model_conf, diffuser):
        super(EGNNScore, self).__init__()
        self._model_conf = model_conf

        self.diffuser = diffuser

        self.scale_pos = lambda x: x * model_conf.ipa.coordinate_scaling
        self.scale_rigids = lambda x: x.apply_trans_fn(self.scale_pos)

        self.unscale_pos = lambda x: x / model_conf.ipa.coordinate_scaling
        self.unscale_rigids = lambda x: x.apply_trans_fn(self.unscale_pos)

        # layers
        self.intra_bb = nn.ModuleList()
        self.inter_bb = nn.ModuleList()

        for i in range(model_conf.n_layers):
            self.intra_bb.append(
                E_GCL(
                    input_nf=model_conf.node_embed_size,
                    output_nf=model_conf.node_embed_size,
                    hidden_nf=model_conf.hidden_dim,
                    edges_in_d=model_conf.edge_embed_size,
                    act_fn=nn.SiLU(),
                    residual=True,
                    attention=False,
                    normalize=False,
                    coords_agg='mean',
                    tanh=False
                )
            )

            if i < model_conf.n_layers - 1:
                self.inter_bb.append(
                    E_GCL(
                        input_nf=model_conf.node_embed_size,
                        output_nf=model_conf.node_embed_size,
                        hidden_nf=model_conf.hidden_dim,
                        edges_in_d=model_conf.edge_embed_size,
                        act_fn=nn.SiLU(),
                        residual=True,
                        attention=False,
                        normalize=False,
                        coords_agg='mean',
                        tanh=False
                    )
                )

        self.final_layer = nn.Linear(model_conf.node_embed_size, 3)
        nn.init.zeros_(self.final_layer.weight)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, h, x, edge_attr, input_feats):
        batch_size, num_atoms = input_feats['atom_types'].shape
        num_bb_atoms = input_feats['num_bb_atoms'][0]

        h = rearrange(h, 'b n d -> (b n) d')
        x = rearrange(x, 'b n d -> (b n) d')
        bb_node_attr = h
        coord = self.scale_pos(x)

        # Compute intra edge_index, edge_attr
        bb_vector = [i for i, count in enumerate(num_bb_atoms) for _ in range(count)]
        bb_vector = torch.tensor([i + b*len(num_bb_atoms) for b in range(batch_size) for i in bb_vector]).to(x.device)
        intra_edge_index = radius_graph(x, r=self._model_conf.max_intra_radius, batch=bb_vector, loop=False)
        
        row, col = intra_edge_index
        batch_vec = torch.arange(batch_size, device=x.device).repeat_interleave(num_atoms)
        batch_indices = batch_vec[row]
        intra_edge_attr = edge_attr[batch_indices, row % num_atoms, col % num_atoms]

        # Compute inter edge_index, edge_attr
        inter_edge_index = radius_graph(x, r=self._model_conf.max_inter_radius, batch=batch_vec, loop=False)
        mask = bb_vector[inter_edge_index[0]] != bb_vector[inter_edge_index[1]]
        inter_edge_index = inter_edge_index[:, mask]                                                                                    # Remove edges within same building blocks

        row, col = inter_edge_index
        batch_indices = batch_vec[row]
        inter_edge_attr = edge_attr[batch_indices, row % num_atoms, col % num_atoms]

        for i in range(len(self.intra_bb)):
            # Message passing within building blocks
            bb_intra_update, x_intra_update, _ = self.intra_bb[i](
                h=bb_node_attr,
                coord=coord,
                edge_index=intra_edge_index,
                edge_attr=intra_edge_attr,
            )          

            # Message passing between building blocks
            if i < len(self.intra_bb) - 1:
                bb_inter_update, x_inter_update, _ = self.inter_bb[i](
                    h=bb_node_attr,
                    coord=coord,
                    edge_index=inter_edge_index,
                    edge_attr=inter_edge_attr,
                )      
            
            # Update node embeddings
            bb_node_attr = bb_node_attr + bb_intra_update + bb_inter_update if i < len(self.intra_bb) - 1 else bb_intra_update + bb_node_attr
            coord = coord + x_intra_update + x_inter_update if i < len(self.intra_bb) - 1 else coord + x_intra_update

        # Reshape
        bb_node_attr = rearrange(bb_node_attr, '(b n) d -> b n d', b=batch_size)

        # output layer
        start_idx = 0
        rot_pred = [] 
        for i, num_atom in enumerate(num_bb_atoms):
            rot_pred.append(bb_node_attr[:, start_idx:start_idx+num_atom, :].mean(dim=1))
            start_idx += num_atom
        rot_pred = torch.stack(rot_pred, dim=1)
        rot_pred = self.final_layer(rot_pred)                                                       # [B, M, 3]

        #  Compute scores
        rot_score = self.diffuser._so3_diffuser.torch_score(rot_pred, input_feats['t'])

        model_out = {
            'rot_pred': rot_pred,
            'rot_score': rot_score,
        }
        
        return model_out
