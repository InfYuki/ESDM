# ESDM

### ESDM: Efficient Spectrum-Conditioned Molecular Diffusion Models for molecular structure elucidation

![image](others/m3_2.jpg)

# Environment Settings
This implementation is based on Python3. To run the code, you need the following dependencies:

- torch==2.3.1+cu118
- torch-geometric==2.3.1
- scipy==1.13.1
- numpy==1.23.0
- tqdm==4.67.1
- scikit-learn==1.6.1
- pytorch-lightning==2.0.4
- pandas==1.4.0
- omegaconf==2.3.0
- rdkit==2024.9.4
- wandb==0.24.0

Detailed environment configuration is in the [requirements.txt](requirements.txt) file.

# Usage
Case commands and parameters are in the [cmd_case.txt](others/cmd_case.txt) 

## Prepare data
Download dataset by run the [00_download_fp2mol_data.sh](data_processing/00_download_fp2mol_data.sh), [01_download_canopus_data.sh](data_processing/01_download_canopus_data.sh), [02_download_msg_data.sh](data_processing/02_download_msg_data.sh), and then run [03_preprocess_fp2mol.sh](data_processing/03_preprocess_fp2mol.sh), [build_fp2mol_datasets.sh](data_processing/build_fp2mol_datasets.sh) to get preprocessed data for trainning.

## Run node classification experiment (train model):

    PYTHONPATH=. python src/spec2mol_main.py \
      general.name=canopus_fs150_TT_C \
      dataset=canopus \
      general.test_only=checkpoints/canopus_fs150_TT_C/last-v1.ckpt \
      general.resume=null \
      general.load_weights=null \
      hydra.job.chdir=false \
      hydra.run.dir=. \
      dataset.datadir=/root/autodl-tmp/DMS/data/canopus \
      dataset.split_file=/root/autodl-tmp/DMS/data/canopus/splits/canopus_hplus_100_0.tsv \
      dataset.subform_folder=/root/autodl-tmp/DMS/data/canopus/subformulae/subformulae_default \
      dataset.labels_file=/root/autodl-tmp/DMS/data/canopus/labels.tsv \
      dataset.spec_folder=/root/autodl-tmp/DMS/data/canopus/spec_files\
      dataset.spec_features=peakformula \
      model.encoder_type=mist \
      general.encoder_finetune_strategy=null \
      model.use_ion_bias=false \
      model.ion_bias_alpha_init=0.00 \
      model.use_heavy_atom_bias=false \
      model.heavy_atom_alpha_init=0.00 \
      model.sampling_steps=100 \
      model.sampling_schedule=quadratic \
      model.use_sampling_corrector=true \
      model.use_per_sample_early_stop=false \
      model.use_multitraj_rerank=false \
      model.use_conditional_timestep_tuner=false \
      general.append_resume_suffix=false




## Result:
![image](others/res.png)

## Ablation Study experiment:

You could run the command (if you are interesting in our work, you could also reset some parameters im the [train_AS.py](train_AS.py) to change the rate of two modules for more ablation study experiment)：

    python train_AS.py --seed 42 --cuda 0 --runs 10 --dataset cora --epoch 2000 --k 1 --nheads 1 --dim 32 --hidden_dim 128 --nlayer 1 --tran_dropout 0.7 --feat_dropout 0.5 --prop_dropout 0.6 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 2 --num_freq 16 --Omega 45.0 --delta_min 0.25 --weight_penalty 1e-4

## Visualization of Filter Responses:

    python train_vis.py --seed 42 --cuda 0 --runs 2 --dataset pubmed --epoch 2000 --k 3 --nheads 2 --dim 16 --hidden_dim 128 --nlayer 3 --tran_dropout 0.4 --feat_dropout 0.3 --prop_dropout 0.0 --lr 0.01 --weight_decay 5e-4 --norm 'none' --patience 300 --num_layers 1 --num_freq 8 --Omega 50.0 --delta_min 0.25 --weight_penalty 1e-4

        
# Baselines links
* [H2GCN](https://github.com/GitEventhandler/H2GCN-PyTorch)
* [HopGNN](https://github.com/JC-202/HopGNN)
* [GPRGNN](https://github.com/jianhao2016/GPRGNN)
* [BernNet](https://github.com/ivam-he/BernNet)
* [JacobiConv](https://github.com/GraphPKU/JacobiConv)
* [HiGCN](https://github.com/Yiminghh/HiGCN)
* [NodeFormer](https://github.com/qitianwu/NodeFormer)
* [SGFormer](https://github.com/qitianwu/SGFormer)
* [NAGphormer](https://github.com/JHL-HUST/NAGphormer)
* [PolyFormer](https://github.com/air029/PolyFormer)
* [Specformer](https://github.com/DSL-Lab/Specformer)
* [GrokFormer](https://github.com/GGA23/GrokFormer/tree/main)
* The implementations of others are taken from the Pytorch Geometric library

# Acknowledgements
The code and filter learning code are implemented based on [GrokFormer: Graph Fourier Kolmogorov-Arnold Transformer](https://github.com/GGA23/GrokFormer/tree/main). We gratefully acknowledge the authors of GrokFormer for their helpful guidance on experimental reproduction and parameter configuration.


# 📖 Citation

If you find this work useful, please cite our paper:

```bibtex

