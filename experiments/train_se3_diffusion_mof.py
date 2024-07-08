"""Pytorch script for training SE(3) protein diffusion.

To run:

> python experiments/train_se3_diffusion.py

Without Wandb,

> python experiments/train_se3_diffusion.py experiment.use_wandb=False

To modify config options with the command line,

> python experiments/train_se3_diffusion.py experiment.batch_size=32

"""
import os
import torch
import GPUtil
import time
import tree
import numpy as np
import wandb
import copy
import hydra
import logging
import copy
import random
import pandas as pd
from tqdm import tqdm

from pathlib import Path
from collections import defaultdict
from collections import deque
from datetime import datetime
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from openfold.utils import rigid_utils as ru
from hydra.core.hydra_config import HydraConfig

from analysis import utils as au
from analysis import metrics
from data import mof_dataset
from data import se3_diffuser
from data import utils as du
from data import all_atom
from model import score_network_mof
from experiments import utils as eu
from common.utils import PROJECT_ROOT


class Experiment:

    def __init__(
            self,
            *,
            conf: DictConfig,
        ):
        """Initialize experiment.

        Args:
            exp_cfg: Experiment configuration.
        """
        self._log = logging.getLogger(__name__)
        self._available_gpus = ''.join(
            [str(x) for x in GPUtil.getAvailable(
                order='memory', limit = 8)])

        # Configs
        self._conf = conf
        self._exp_conf = conf.experiment
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            self._exp_conf.name = (
                f'{self._exp_conf.name}_{HydraConfig.get().job.num}')
        self._diff_conf = conf.diffuser
        self._model_conf = conf.model
        self._data_conf = conf.data
        self._use_wandb = self._exp_conf.use_wandb
        self._use_ddp = self._exp_conf.use_ddp
        # 1. initialize ddp info if in ddp mode
        # 2. silent rest of logger when use ddp mode
        # 3. silent wandb logger
        # 4. unset checkpoint path if rank is not 0 to avoid saving checkpoints and evaluation
        if self._use_ddp :
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(backend='nccl')
            self.ddp_info = eu.get_ddp_info()
            if self.ddp_info['rank'] not in [0,-1]:
                self._log.addHandler(logging.NullHandler())
                self._log.setLevel("ERROR")
                self._use_wandb = False
                self._exp_conf.ckpt_dir = None
        # Warm starting
        ckpt_model = None
        ckpt_opt = None
        self.trained_epochs = 0
        self.trained_steps = 0
        if conf.experiment.warm_start:
            ckpt_dir = conf.experiment.warm_start
            self._log.info(f'Warm starting from: {ckpt_dir}')
            ckpt_files = [
                x for x in os.listdir(ckpt_dir)
                if 'pkl' in x or '.pth' in x
            ]
            if len(ckpt_files) != 1:
                raise ValueError(f'Ambiguous ckpt in {ckpt_dir}')
            ckpt_name = ckpt_files[0]
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            self._log.info(f'Loading checkpoint from {ckpt_path}')
            ckpt_pkl = du.read_pkl(ckpt_path, use_torch=True)
            ckpt_model = ckpt_pkl['model']

            if conf.experiment.use_warm_start_conf:
                OmegaConf.set_struct(conf, False)
                conf = OmegaConf.merge(conf, ckpt_pkl['conf'])
                OmegaConf.set_struct(conf, True)
            conf.experiment.warm_start = ckpt_dir

            # For compatibility with older checkpoints.
            if 'optimizer' in ckpt_pkl:
                ckpt_opt = ckpt_pkl['optimizer']
            if 'epoch' in ckpt_pkl:
                self.trained_epochs = ckpt_pkl['epoch']
            if 'step' in ckpt_pkl:
                self.trained_steps = ckpt_pkl['step']


        # Initialize experiment objects
        self._diffuser = se3_diffuser.SE3Diffuser(self._diff_conf)
        self._model = score_network_mof.ScoreNetwork(
            self._model_conf, self.diffuser)

        if ckpt_model is not None:
            ckpt_model = {k.replace('module.', ''):v for k,v in ckpt_model.items()}
            self._model.load_state_dict(ckpt_model, strict=True)

        num_parameters = sum(p.numel() for p in self._model.parameters())
        self._exp_conf.num_parameters = num_parameters
        self._log.info(f'Number of model parameters {num_parameters}')
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._exp_conf.learning_rate)
        if ckpt_opt is not None:
            self._optimizer.load_state_dict(ckpt_opt)

        dt_string = datetime.now().strftime("%dD_%mM_%YY_%Hh_%Mm_%Ss")
        hydra_dir = HydraConfig.get().run.dir
        if self._exp_conf.ckpt_dir is not None:
            # Set-up checkpoint location
            ckpt_dir = os.path.join(
                hydra_dir,
                self._exp_conf.ckpt_dir,
            )
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir, exist_ok=True)
            self._exp_conf.ckpt_dir = ckpt_dir
            self._log.info(f'Checkpoints saved to: {ckpt_dir}')
        else:  
            self._log.info('Checkpoint not being saved.')
        if self._exp_conf.eval_dir is not None :
            eval_dir = os.path.join(
                hydra_dir,
                self._exp_conf.eval_dir,
            )
            self._exp_conf.eval_dir = eval_dir
            self._log.info(f'Evaluation saved to: {eval_dir}')
        else:
            self._exp_conf.eval_dir = os.devnull
            self._log.info(f'Evaluation will not be saved.')
        self._aux_data_history = deque(maxlen=100)

    @property
    def diffuser(self):
        return self._diffuser

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

    def create_dataset(self):

        # Datasets
        train_dataset = mof_dataset.MOFDataset(
            cache_path=os.path.join(self._data_conf.cache_dir, 'train_dev.pt'),
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=True
        )

        valid_dataset = mof_dataset.MOFDataset(
            cache_path=os.path.join(self._data_conf.cache_dir, 'val_dev.pt'),
            data_conf=self._data_conf,
            diffuser=self._diffuser,
            is_training=False
        )
        if not self._use_ddp:
            train_sampler = mof_dataset.TrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
                sample_mode=self._exp_conf.sample_mode,
            )
        else:
            train_sampler = mof_dataset.DistributedTrainSampler(
                data_conf=self._data_conf,
                dataset=train_dataset,
                batch_size=self._exp_conf.batch_size,
            )
        valid_sampler = None

        # Loaders
        num_workers = self._exp_conf.num_loader_workers
        train_loader = du.create_data_loader(
            train_dataset,
            sampler=train_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=self._exp_conf.batch_size if not self._exp_conf.use_ddp else self._exp_conf.batch_size // self.ddp_info['world_size'],
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            max_squared_res=self._exp_conf.max_squared_res,
        )
        valid_loader = du.create_data_loader(
            valid_dataset,
            sampler=valid_sampler,
            np_collate=False,
            length_batch=True,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, valid_loader, train_sampler, valid_sampler

    def init_wandb(self):
        self._log.info('Initializing Wandb.')
        conf_dict = OmegaConf.to_container(self._conf, resolve=True)
        wandb.init(
            project=self._exp_conf.project,
            name=self._exp_conf.name,
            config=dict(eu.flatten_dict(conf_dict)),
            dir=self._exp_conf.wandb_dir,
        )
        wandb.watch(self.model, log='all', log_freq=100)
        self._exp_conf.run_id = wandb.util.generate_id()
        self._exp_conf.wandb_dir = wandb.run.dir
        self._log.info(
            f'Wandb: run_id={self._exp_conf.run_id}, run_dir={self._exp_conf.wandb_dir}')

    def start_training(self, return_logs=False):
        # Set environment variables for which GPUs to use.
        if HydraConfig.initialized() and 'num' in HydraConfig.get().job:
            replica_id = int(HydraConfig.get().job.num)
        else:
            replica_id = 0
        if self._use_wandb and replica_id == 0:
                self.init_wandb()
        assert(not self._exp_conf.use_ddp or self._exp_conf.use_gpu)

        # GPU mode
        if torch.cuda.is_available() and self._exp_conf.use_gpu:
            # single GPU mode
            if self._exp_conf.num_gpus==1 :
                gpu_id = self._available_gpus[replica_id]
                device = f"cuda:{gpu_id}"
                self._model = self.model.to(device)
                self._log.info(f"Using device: {device}")
            #muti gpu mode
            elif self._exp_conf.num_gpus > 1:
                device_ids = [
                f"cuda:{i}" for i in self._available_gpus[:self._exp_conf.num_gpus]
                ]
                #DDP mode
                if self._use_ddp :
                    device = torch.device("cuda",self.ddp_info['local_rank'])
                    model = self.model.to(device)
                    self._model = DDP(model, device_ids=[self.ddp_info['local_rank']], output_device=self.ddp_info['local_rank'],find_unused_parameters=True)
                    self._log.info(f"Multi-GPU training on GPUs in DDP mode, node_id : {self.ddp_info['node_id']}, devices: {device_ids}")
                #DP mode
                else:
                    if len(self._available_gpus) < self._exp_conf.num_gpus:
                        raise ValueError(f"require {self._exp_conf.num_gpus} GPUs, but only {len(self._available_gpus)} GPUs available ")
                    self._log.info(f"Multi-GPU training on GPUs in DP mode: {device_ids}")
                    gpu_id = self._available_gpus[replica_id]
                    device = f"cuda:{gpu_id}"
                    self._model = DP(self._model, device_ids=device_ids)
                    self._model = self.model.to(device)
        else:
            device = 'cpu'
            self._model = self.model.to(device)
            self._log.info(f"Using device: {device}")

        self._model.train()

        (
            train_loader,
            valid_loader,
            train_sampler,
            valid_sampler
        ) = self.create_dataset()

        logs = []
        for epoch in range(self.trained_epochs, self._exp_conf.num_epoch):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)
            self.trained_epochs = epoch
            epoch_log = self.train_epoch(
                train_loader,
                valid_loader,
                device,
                return_logs=return_logs
            )
            if return_logs:
                logs.append(epoch_log)

        self._log.info('Done')
        return logs

    def update_fn(self, data):
        """Updates the state using some data and returns metrics."""
        self._optimizer.zero_grad()
        loss, aux_data = self.loss_fn(data)
        loss.backward()
        self._optimizer.step()
        return loss, aux_data

    def train_epoch(
            self, train_loader, valid_loader, device, return_logs=False):
        log_lossses = defaultdict(list)
        global_logs = []
        log_time = time.time()
        step_time = time.time()
        for train_feats in tqdm(train_loader, desc=f"Epoch {self.trained_epochs}"):
            train_feats = tree.map_structure(
                lambda x: x.to(device), train_feats)
            loss, aux_data = self.update_fn(train_feats)
            if return_logs:
                global_logs.append(loss)
            for k,v in aux_data.items():
                log_lossses[k].append(du.move_to_np(v))
            self.trained_steps += 1

            # Logging to terminal
            if self.trained_steps == 1 or self.trained_steps % self._exp_conf.log_freq == 0:
                elapsed_time = time.time() - log_time
                log_time = time.time()
                step_per_sec = self._exp_conf.log_freq / elapsed_time
                rolling_losses = tree.map_structure(np.mean, log_lossses)
                loss_log = ' '.join([
                    f'{k}={v[0]:.4f}'
                    for k,v in rolling_losses.items() if 'batch' not in k
                ])
                self._log.info(
                    f'[{self.trained_steps}]: {loss_log}, steps/sec={step_per_sec:.5f}')
                log_lossses = defaultdict(list)

            # Take checkpoint
            if self._exp_conf.ckpt_dir is not None and (
                    (self.trained_steps % self._exp_conf.ckpt_freq) == 0
                    or (self._exp_conf.early_ckpt and self.trained_steps == 100)
                ):
                ckpt_path = os.path.join(
                    self._exp_conf.ckpt_dir, f'step_{self.trained_steps}.pth')
                du.write_checkpoint(
                    ckpt_path,
                    copy.deepcopy(self.model.state_dict()),
                    self._conf,
                    copy.deepcopy(self._optimizer.state_dict()),
                    self.trained_epochs,
                    self.trained_steps,
                    logger=self._log,
                    use_torch=True
                )

                # Run evaluation
                self._log.info(f'Running evaluation of {ckpt_path}')
                start_time = time.time()
                eval_dir = os.path.join(
                    self._exp_conf.eval_dir, f'step_{self.trained_steps}')
                os.makedirs(eval_dir, exist_ok=True)
                ckpt_metrics = self.eval_fn(valid_loader, device)
                eval_time = time.time() - start_time
                self._log.info(f'Finished evaluation in {eval_time:.2f}s')
            else:
                ckpt_metrics = None
                eval_time = None

            # Remote log to Wandb.
            if self._use_wandb:
                step_time = time.time() - step_time
                example_per_sec = self._exp_conf.batch_size / step_time
                step_time = time.time()
                wandb_logs = {
                    'loss': loss,
                    'rotation_loss': aux_data['rot_loss'],
                    'batch_size': aux_data['examples_per_step'],
                    'num_bbs': aux_data['num_bbs'],
                    'examples_per_sec': example_per_sec,
                    'num_epochs': self.trained_epochs,
                }

                # Stratified losses
                wandb_logs.update(eu.t_stratified_loss(
                    du.move_to_np(train_feats['t']),
                    du.move_to_np(aux_data['batch_rot_loss']),
                    loss_name='rot_loss',
                ))

                if ckpt_metrics is not None:
                    wandb_logs['eval_time'] = eval_time
                    wandb_logs.update(ckpt_metrics)

                wandb.log(wandb_logs, step=self.trained_steps)

            if torch.isnan(loss):
                if self._use_wandb:
                    wandb.alert(
                        title="Encountered NaN loss",
                        text=f"Loss NaN after {self.trained_epochs} epochs, {self.trained_steps} steps"
                    )
                raise Exception(f'NaN encountered')

        if return_logs:
            return global_logs

    def eval_fn(self, valid_loader, device):
        log_losses = defaultdict(list)
        for valid_feats in tqdm(valid_loader, desc=f"Epoch {self.trained_epochs} Eval"):
            try:
                valid_feats = tree.map_structure(
                    lambda x: x.to(device), valid_feats
                )
                _, val_aux_data = self.loss_fn(valid_feats)
                for k,v in val_aux_data.items():
                    log_losses[k].append(du.move_to_np(v))
            except RuntimeError as e:
                self._log.warning(f'Failed evaluation: {e}')
        
        log_losses = tree.map_structure(np.mean, log_losses)

        ckpt_metrics = {
            'val_loss': np.mean(log_losses['total_loss']),
            'val_rot_loss': np.mean(log_losses['rot_loss']),
        }

        return ckpt_metrics

    def loss_fn(self, batch):
        """Computes loss and auxiliary data.

        Args:
            batch: Batched data.
            model_out: Output of model ran on batch.

        Returns:
            loss: Final training loss scalar.
            aux_data: Additional logging data.
        """
        # if self._model_conf.embed.embed_self_conditioning and random.random() > 0.5:
        #     with torch.no_grad():
        #         batch = self._self_conditioning(batch)
        model_out = self.model(batch)
        bb_mask = batch['res_mask']
        diffuse_mask = 1 - batch['fixed_mask']
        loss_mask = bb_mask * diffuse_mask
        batch_size, num_res = bb_mask.shape

        gt_rot_score = batch['rot_score']
        rot_score_scaling = batch['rot_score_scaling']
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        pred_rot_score = model_out['rot_score'] * diffuse_mask[..., None]

        # Rotation loss
        if self._exp_conf.separate_rot_loss:
            gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
            gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)

            pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
            pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)

            # Separate loss on the axis
            axis_loss = (gt_rot_axis - pred_rot_axis)**2 * loss_mask[..., None]
            axis_loss = torch.sum(
                axis_loss, dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)

            # Separate loss on the angle
            angle_loss = (gt_rot_angle - pred_rot_angle)**2 * loss_mask[..., None]
            angle_loss = torch.sum(
                angle_loss / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            angle_loss *= self._exp_conf.rot_loss_weight
            angle_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
            rot_loss = angle_loss + axis_loss
        else:
            rot_mse = (gt_rot_score - pred_rot_score)**2 * loss_mask[..., None]
            rot_loss = torch.sum(
                rot_mse / rot_score_scaling[:, None, None]**2,
                dim=(-1, -2)
            ) / (loss_mask.sum(dim=-1) + 1e-10)
            rot_loss *= self._exp_conf.rot_loss_weight
            rot_loss *= batch['t'] > self._exp_conf.rot_loss_t_threshold
        rot_loss *= int(self._diff_conf.diffuse_rot)

        final_loss = rot_loss

        def normalize_loss(x):
            return x.sum() /  (batch_loss_mask.sum() + 1e-10)

        aux_data = {
            'batch_train_loss': final_loss,
            'batch_rot_loss': rot_loss,
            'total_loss': normalize_loss(final_loss),
            'rot_loss': normalize_loss(rot_loss),
            'examples_per_step': torch.tensor(batch_size),
            'num_bbs': torch.mean(torch.sum(bb_mask, dim=-1)),
        }

        # Maintain a history of the past N number of steps.
        # Helpful for debugging.
        self._aux_data_history.append({
            'aux_data': aux_data,
            'model_out': model_out,
            'batch': batch
        })

        assert final_loss.shape == (batch_size,)
        assert batch_loss_mask.shape == (batch_size,)
        return normalize_loss(final_loss), aux_data

    def _calc_trans_0(self, trans_score, trans_t, t):
        beta_t = self._diffuser._se3_diffuser._r3_diffuser.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        cond_var = 1 - torch.exp(-beta_t)
        return (trans_score * cond_var + trans_t) / torch.exp(-1/2*beta_t)

    def compute_coords(self, input_feats, rigid=None):

        x_bb = input_feats['x_bb'][0]  # [N, 3]
        num_bb_atoms = input_feats['num_bb_atoms'][0]

        start_idx = 0
        x_pred = []

        # Deal with different types of rigid
        if rigid is not None:
            pred_rigid = ru.Rigid.from_tensor_7(rigid) if isinstance(rigid, torch.Tensor) else rigid # [1, M, 7]
            pred_rigid._rots = pred_rigid._rots.to(x_bb.device, dtype=x_bb.dtype)
            pred_rigid._trans = pred_rigid._trans.to(x_bb.device)
        else:
            pred_rigid = ru.Rigid.from_tensor_7(input_feats['rigids_t']) # [1, M, 7]

        # Rototranslate each building block 
        for i, num_bb_atom in enumerate(num_bb_atoms):
            bb_coord = x_bb[start_idx:start_idx + num_bb_atom, :]  # [num_bb_atom, 3]
            rot_mats = pred_rigid[:, i]._rots.get_rot_mats()  # [1, 3, 3]
            trans = pred_rigid[:, i]._trans[:, None, :]  # [1, 1, 3]
            
            # Apply rotation and translation
            pred_coord = torch.einsum('bij,nj->bni', rot_mats, bb_coord) + trans  # [1, num_bb_atom, 3]
            
            x_pred.append(pred_coord)
            start_idx += num_bb_atom

        x_pred = torch.cat(x_pred, dim=1)  # [1, N, 3]

        return x_pred # [1, N, 3]
    
    def _set_t_feats(self, feats, t, t_placeholder):
        feats['t'] = t * t_placeholder
        rot_score_scaling, trans_score_scaling = self.diffuser.score_scaling(t)
        feats['rot_score_scaling'] = rot_score_scaling * t_placeholder
        feats['trans_score_scaling'] = trans_score_scaling * t_placeholder
        feats['x_t'] = self.compute_coords(feats)
        return feats

    def forward_traj(self, x_0, min_t, num_t):
        forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
        x_traj = [x_0]
        for t in forward_steps:
            x_t = self.diffuser.se3_diffuser._r3_diffuser.forward(
                x_traj[-1], t, num_t)
            x_traj.append(x_t)
        x_traj = torch.stack(x_traj, axis=0)
        return x_traj

    def inference_fn(
            self,
            data_init,
            num_t=None,
            min_t=None,
            center=True,
            aux_traj=False,
            noise_scale=1.0,
        ):
        """Inference function.

        Args:
            data_init: Initial data values for sampling.
        """

        # Run reverse process.
        sample_feats = copy.deepcopy(data_init)
        device = sample_feats['rigids_t'].device
        if sample_feats['rigids_t'].ndim == 2:
            t_placeholder = torch.ones((1,)).to(device)
        else:
            t_placeholder = torch.ones(
                (sample_feats['rigids_t'].shape[0],)).to(device)
        if num_t is None:
            num_t = self._data_conf.num_t
        if min_t is None:
            min_t = self._data_conf.min_t
        reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
        dt = 1/num_t
        all_rigids = [du.move_to_np(copy.deepcopy(sample_feats['rigids_t']))]
        all_bb_prots = []
        all_trans_0_pred = []
        all_bb_0_pred = []
        with torch.no_grad():
            for t in reverse_steps:
                if t > min_t:
                    sample_feats = self._set_t_feats(sample_feats, t, t_placeholder)
                    model_out = self.model(sample_feats)
                    rot_score = model_out['rot_score']
                    trans_score = model_out['trans_score']
                    rigid_pred = model_out['rigids']
                    fixed_mask = sample_feats['fixed_mask'] * sample_feats['res_mask']
                    diffuse_mask = (1 - sample_feats['fixed_mask']) * sample_feats['res_mask']
                    rigids_t = self.diffuser.reverse(
                        rigid_t=ru.Rigid.from_tensor_7(sample_feats['rigids_t']),
                        rot_score=du.move_to_np(rot_score),
                        trans_score=du.move_to_np(trans_score),
                        diffuse_mask=du.move_to_np(diffuse_mask),
                        t=t,
                        dt=dt,
                        center=center,
                        noise_scale=noise_scale,
                    )
                else:
                    model_out = self.model(sample_feats)
                    rigids_t = ru.Rigid.from_tensor_7(model_out['rigids'])
                sample_feats['rigids_t'] = rigids_t.to_tensor_7().to(device)
                if aux_traj:
                    all_rigids.append(du.move_to_np(rigids_t.to_tensor_7()))

                # Calculate x0 prediction derived from score predictions.
                gt_trans_0 = sample_feats['rigids_t'][..., 4:]
                pred_trans_0 = rigid_pred[..., 4:]
                trans_pred_0 = diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
                if aux_traj:
                    mof_0 = self.compute_coords(sample_feats, rigid_pred)
                    all_bb_0_pred.append(du.move_to_np(mof_0))
                    all_trans_0_pred.append(du.move_to_np(trans_pred_0))
                mof_t = self.compute_coords(sample_feats, rigids_t)
                all_bb_prots.append(du.move_to_np(mof_t))

        # Flip trajectory so that it starts from t=0.
        # This helps visualization.
        flip = lambda x: np.flip(np.stack(x), (0,))
        all_bb_prots = flip(all_bb_prots)
        if aux_traj:
            all_rigids = flip(all_rigids)
            all_trans_0_pred = flip(all_trans_0_pred)
            all_bb_0_pred = flip(all_bb_0_pred)
        ret = {
            'mof_traj': all_bb_prots,
        }
        if aux_traj:
            ret['rigid_traj'] = all_rigids
            ret['trans_traj'] = all_trans_0_pred
            ret['rigid_0_traj'] = all_bb_0_pred
        return ret


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "config"), config_name="mof_translation")
def run(conf: DictConfig) -> None:

    # Fixes bug in https://github.com/wandb/wandb/issues/1525
    os.environ["WANDB_START_METHOD"] = "thread"

    exp = Experiment(conf=conf)
    exp.start_training()


if __name__ == '__main__':
    run()
