"""Script for running inference and sampling.

Sample command:
> python scripts/run_inference.py

"""

import os
import time
import tree
import numpy as np
import hydra
import torch
import subprocess
import logging
import pandas as pd
import shutil
from datetime import datetime
from tqdm import tqdm
from biotite.sequence.io import fasta
from hydra.core.hydra_config import HydraConfig
import GPUtil
from typing import Optional

from analysis import utils as au
from analysis import metrics
from data import utils as du
from data import residue_constants
from data import mof_dataset
from typing import Dict
from experiments import train_se3_diffusion_mof
from common.utils import PROJECT_ROOT
from openfold.utils.rigid_utils import Rigid

from omegaconf import DictConfig, OmegaConf
import esm

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifWriter

def get_rigids_impute(num_bbs):
    # return Rigid.from_tensor_4x4(torch.zeros((num_bbs, 4, 4)))
    return Rigid.identity((num_bbs,), device=num_bbs.device, requires_grad=False)

class Sampler:

    def __init__(
            self,
            conf: DictConfig,
            conf_overrides: Dict=None
        ):
        """Initialize sampler.

        Args:
            conf: inference config.
            gpu_id: GPU device ID.
            conf_overrides: Dict of fields to override with new values.
        """
        self._log = logging.getLogger(__name__)

        # Remove static type checking.
        OmegaConf.set_struct(conf, False)

        # Prepare configs.
        self._conf = conf
        self._infer_conf = conf.inference
        self._diff_conf = self._infer_conf.diffusion
        self._sample_conf = self._infer_conf.samples

        self._rng = np.random.default_rng(self._infer_conf.seed)

        # Set-up accelerator
        if torch.cuda.is_available():
            if self._infer_conf.gpu_id is None:
                available_gpus = ''.join(
                    [str(x) for x in GPUtil.getAvailable(
                        order='memory', limit = 8)])
                self.device = f'cuda:{available_gpus[0]}'
            else:
                self.device = f'cuda:{self._infer_conf.gpu_id}'
        else:
            self.device = 'cpu'
        self._log.info(f'Using device: {self.device}')

        # Set-up directories
        hydra_dir = HydraConfig.get().run.dir
        self._weights_path = os.path.join(
            hydra_dir,
            self._infer_conf.weights_path
        )
        output_dir = os.path.join(
            hydra_dir,
            self._infer_conf.output_dir
        )
        if self._infer_conf.name is None:
            dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        else:
            dt_string = self._infer_conf.name
        self._output_dir = os.path.join(output_dir, dt_string)
        os.makedirs(self._output_dir, exist_ok=True)
        self._log.info(f'Saving results to {self._output_dir}')

        config_path = os.path.join(self._output_dir, 'inference_conf.yaml')
        with open(config_path, 'w') as f:
            OmegaConf.save(config=self._conf, f=f)
        self._log.info(f'Saving inference config to {config_path}')

        # Load models and experiment
        self._load_ckpt(conf_overrides)

    def _load_ckpt(self, conf_overrides):
        """Loads in model checkpoint."""
        self._log.info(f'Loading weights from {self._weights_path}')

        # Read checkpoint and create experiment.
        weights_pkl = du.read_pkl(
            self._weights_path, use_torch=True,
            map_location=self.device)

        # Merge base experiment config with checkpoint config.
        self._conf.model = OmegaConf.merge(
            self._conf.model, weights_pkl['conf'].model)
        if conf_overrides is not None:
            self._conf = OmegaConf.merge(self._conf, conf_overrides)

        # Prepare model
        self._conf.experiment.ckpt_dir = None
        self._conf.experiment.warm_start = None
        self.exp = train_se3_diffusion_mof.Experiment(
            conf=self._conf)
        self.model = self.exp.model

        # Remove module prefix if it exists.
        model_weights = weights_pkl['model']
        model_weights = {
            k.replace('module.', ''):v for k,v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.diffuser = self.exp.diffuser

    def init_data(
            self,
            *,
            rigids_impute,
            torsion_impute,
            fixed_mask,
            res_mask,
        ):
        num_res = res_mask.shape[0]
        diffuse_mask = (1 - fixed_mask) * res_mask
        fixed_mask = fixed_mask * res_mask

        ref_sample = self.diffuser.sample_ref(
            n_samples=num_res,
            rigids_impute=rigids_impute,
            diffuse_mask=diffuse_mask,
            as_tensor_7=True,
        )
        res_idx = torch.arange(1, num_res+1)
        init_feats = {
            'res_mask': res_mask,
            'seq_idx': res_idx * res_mask,
            'fixed_mask': fixed_mask,
            'torsion_angles_sin_cos': torsion_impute,
            'sc_ca_t': torch.zeros_like(rigids_impute.get_trans()),
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)
        return init_feats

    def create_dataset(self):

        # Datasets
        test_dataset = mof_dataset.MOFDataset(
            cache_path=os.path.join(self._conf.data.cache_dir, 'val_dev.pt'),
            data_conf=self._conf.data,
            diffuser=self.diffuser,
            is_training=False,
            is_testing=True
        )

        return test_dataset
    
    def run_sampling(self):
        """Sets up inference run.

        All outputs are written to 
            {output_dir}/{date_time}
        where {output_dir} is created at initialization.
        """
        # Create dataset 
        test_dataset = self.create_dataset()

        for data in test_dataset:
            data_dir = os.path.join(
                self._output_dir, f"{data['name']}")
            os.makedirs(data_dir, exist_ok=True)
            for sample_i in range(self._sample_conf.samples_per_data):
                sample_dir = os.path.join(data_dir, f'sample_{sample_i}')
                if os.path.isdir(sample_dir):
                    continue
                os.makedirs(sample_dir, exist_ok=True)
                sample_output = self.sample(data)
                traj_paths = self.save_traj(
                    data,
                    sample_output['mof_traj'],
                    sample_output['rigid_0_traj'],
                    np.ones(data['num_bbs']),
                    output_dir=sample_dir
                )

                # Logging
                mof_path = traj_paths['sample_path']
                self._log.info(f'Done sample {sample_i}: {mof_path}')

    def save_traj(
            self,
            data,
            mof_traj: np.ndarray,
            x0_traj: np.ndarray,
            diffuse_mask: np.ndarray,
            output_dir: str
        ):
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            aatype: [T, N, 21] amino acid probability vector trajectory.
            res_mask: [N] residue mask.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.

        Returns:
            Dictionary with paths to saved samples.
                'sample_path': PDB file of final state of reverse trajectory.
                'traj_path': PDB file os all intermediate diffused states.
                'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
            residues if there are any.
        """

        # Write sample.
        sample_path = os.path.join(output_dir, 'sample')
        mof_traj_path = os.path.join(output_dir, 'mof_traj')
        x0_traj_path = os.path.join(output_dir, 'x0_traj')

        sample_coords = mof_traj[0]
        atom_types = data['atom_types'].numpy()
        lattice = data['lattice'].numpy().flatten()

        structure = Structure(
            lattice=Lattice.from_parameters(*lattice), 
            species=atom_types, 
            coords=sample_coords,
            coords_are_cartesian=True,
        )

        cif_path = os.path.join(output_dir, 'sample.cif')
        cif_writer = CifWriter(structure)
        cif_writer.write_file(cif_path)

        assert False
        
        prot_traj_path = None
        x0_traj_path = None
        return {
            'sample_path': sample_path,
            'traj_path': mof_traj_path,
            'x0_traj_path': x0_traj_path,
        }

    def sample(self, data):
        """Sample based on length.

        Args:
            data: mof sample from dataset.

        Returns:
            Sample outputs. See train_se3_diffusion_mof.inference_fn.
        """
        # Process motif features.
        res_mask = np.ones(data['num_bbs'])
        fixed_mask = np.zeros_like(data['num_bbs'])

        # Initialize data
        if not self.diffuser._diffuse_rot:
            # no rotation diffusion
            rigids_impute = get_rigids_impute(data['num_bbs'])
            ref_sample = self.diffuser.sample_ref(
                n_samples=data['num_bbs'],
                impute=rigids_impute,
                as_tensor_7=True,
            )
        else:
            # both rotation and translation diffusion
            ref_sample = self.diffuser.sample_ref(
                n_samples=data['num_bbs'],
                as_tensor_7=True,
            )
        init_feats = {
            'res_mask': res_mask,
            'x_bb': data['x_bb'],
            'atom_types': data['atom_types'],
            'num_bb_atoms': data['num_bb_atoms'],
            'fixed_mask': fixed_mask,
            **ref_sample,
        }
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats)
        init_feats = tree.map_structure(
            lambda x: x[None].to(self.device), init_feats)

        # Run inference
        sample_out = self.exp.inference_fn(
            init_feats,
            num_t=self._diff_conf.num_t,
            min_t=self._diff_conf.min_t, 
            aux_traj=True,
            noise_scale=self._diff_conf.noise_scale,
        )
        return tree.map_structure(lambda x: x[:, 0], sample_out)

@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "config"), config_name="inference_mof")
def run(conf: DictConfig) -> None:

    # Read model checkpoint.
    print('Starting inference')
    start_time = time.time()
    sampler = Sampler(conf)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    print(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()
