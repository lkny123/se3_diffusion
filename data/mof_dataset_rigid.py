"""MOF dataset loader."""
import math
from typing import Optional

import torch
import torch.distributed as dist

import tree
import numpy as np
import torch
import pandas as pd
import logging
import random
import functools as fn

from torch.utils import data
from torch_geometric.data import Data

from data import utils as du

from openfold.utils import rigid_utils
from ase.data import chemical_symbols

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from mofdiff.common.data_utils import frac_to_cart_coords


CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)

class MOFDataset(data.Dataset):
    def __init__(
            self,
            *,
            cache_path,
            data_conf,
            diffuser,
            is_training,
        ):

        self.cached_data = torch.load(cache_path)

        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._diffuser = diffuser

    @property
    def is_training(self):
        return self._is_training

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def data_conf(self):
        return self._data_conf

    def get_pca_axes(self, data):
        """
        Args:
            data: np.array of shape (n_samples, 3)
        
        Returns:
            eigenvalues: np.array of shape (3,)
            eigenvectors: np.array of shape (3, 3)
        """

        # Center the data
        data_mean = np.mean(data, axis=0)
        centered_data = data - data_mean

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)
        if covariance_matrix.ndim == 0:
            return np.zeros(3), np.eye(3)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors
    
    def get_rot_0(self, mof):
        """
        Returns:
            rots: rigid.utils.Rotation of shape (n_bbs, 3, 3)
        """
        rots = []

        for bb in mof.pyg_mols:
            x_bb = frac_to_cart_coords(bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms)
            _, bb_rot_0 = self.get_pca_axes(x_bb.numpy())

            rots.append(bb_rot_0)

        rots = torch.tensor(np.stack(rots, axis=0))
        rots = rigid_utils.Rotation(rot_mats=rots)
        return rots 
    
    def get_trans_0(self, mof):
        """
        Returns:
            trans: torch.Tensor of shape (n_bbs, 3)
        """
        num_bbs = mof.num_components
        return torch.zeros((num_bbs, 3))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):

        feats = {}

        data = self.cached_data[idx]
        rot_0 = self.get_rot_0(data)
        trans_0 = self.get_trans_0(data)
        rigids_0 = rigid_utils.Rigid(rots=rot_0, trans=trans_0)

        num_bb_atoms = torch.tensor([bb.num_atoms for bb in data.pyg_mols])
        atom_types = torch.cat([bb.atom_types for bb in data.pyg_mols])

        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self._data_conf.min_t, 1.0)
            diff_feats_t = self._diffuser.forward_marginal(
                rigids_0=rigids_0,
                t=t,
                diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=rigids_0.shape[0],
                impute=rigids_0,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        
        feats['t'] = t
        feats['rigids_0'] = rigids_0.to_tensor_7()
        feats.update(diff_feats_t)

        feats['atom_types'] = atom_types
        feats['num_atoms'] = data.num_atoms
        feats['num_bb_atoms'] = num_bb_atoms
        feats['lattice'] = torch.cat([data.lengths, data.angles], dim=1)
        feats['fingerprint'] = torch.stack([bb.fp for bb in data.bbs], dim=0).float()       # (n_bbs, 4096)
        feats['rigids_0'] = rigids_0.to_tensor_7()
        feats['sc_ca_t'] = torch.zeros_like(rigids_0.get_trans())

        feats['res_mask'] = torch.ones(data.num_components)
        feats['fixed_mask'] = torch.zeros(data.num_components)

        feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), feats)

        return feats


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        # self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._dataset)))
        # self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len



# modified from torch.utils.data.distributed.DistributedSampler
# key points: shuffle of each __iter__ is determined by epoch num to ensure the same shuffle result for each proccessor
class DistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, 
                *,
                data_conf,
                dataset,
                batch_size,
                num_replicas: Optional[int] = None,
                rank: Optional[int] = None, shuffle: bool = True,
                seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) :
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate((indices, np.repeat(indices, math.ceil(padding_size / len(indices)))[:padding_size]), axis=0)

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        # 
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


if __name__ == "__main__":
    dataset = MOFDataset(
        cache_path='data/mof/cif_inf_100/train_mini.pt',
        data_conf=None,
        diffuser=None,
        is_training=True,
    )

    data = dataset[0]
    print(data)