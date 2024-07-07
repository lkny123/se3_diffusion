### Environment installation

```bash
mamba env create -f env.yml
pip install -e .
pip install git+https://lkny123:github.com/lkny123/MOFDiff.git --upgrade
```

### Multi-GPU training

```bash
torchrun experiments/train_se3_diffusion_mof.py experiment.num_gpus=4 experiment.use_wandb=true
```