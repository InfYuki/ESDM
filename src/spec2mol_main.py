import os
import sys
import pathlib
import warnings
import logging

import torch
torch.cuda.empty_cache()
try:
    torch.set_float32_matmul_precision('medium')
    logging.info("Enabled float32 matmul precision - medium")
except:
    logging.info("Could not enable float32 matmul precision - medium")
import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from src.diffusion_model_spec2mol import Spec2MolDenoisingDiffusion
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.analysis.visualization import MolecularVisualization
from src.datasets import spec2mol_dataset


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def copy_runtime_overrides(cfg, saved_cfg):
    """Keep checkpoint architecture, but allow test-time/data/sampling overrides."""
    override_keys = {
        "general": [
            "test_only",
            "resume",
            "load_weights",
            "name",
            "wandb",
            "wandb_name",
            "gpus",
            "val_samples_to_generate",
            "test_samples_to_generate",
            "encoder_finetune_strategy",
            "decoder_finetune_strategy",
            "evaluate_all_checkpoints",
            "checkpoint_strategy",
            "test_resume_enabled",
            "test_resume_strict_pair",
            "test_resume_overwrite_broken",
            "test_resume_skip_loss_on_hit",
            "append_resume_suffix",
        ],
        "dataset": [
            "datadir",
            "split_file",
            "subform_folder",
            "labels_file",
            "spec_folder",
            "spec_features",
            "dreams_cache_dir",
            "max_count",
            "max_peaks",
        ],
        "train": [
            "num_workers",
            "pin_memory",
        ],
        "model": [
            "use_sampling_acceleration",
            "sampling_steps",
            "sampling_schedule",
            "sampling_tuned_steps",
            "use_sampling_corrector",
            "corrector_every_n",
            "corrector_temperature",
            "corrector_edge_prior_strength",
            "corrector_apply_until_t",
            "corrector_logit_clip",
            "use_per_sample_early_stop",
            "early_stop_change_threshold",
            "early_stop_patience",
            "early_stop_min_steps",
            "use_multitraj_rerank",
            "multitraj_candidates",
            "rerank_w_valid",
            "rerank_w_conf",
            "rerank_w_prior",
            "use_conditional_timestep_tuner",
            "conditional_tuner_mode",
            "ctt_min_steps",
            "ctt_max_steps",
            "ctt_default_power",
            "ctt_entropy_eps",
            "use_ion_bias",
            "use_heavy_atom_bias",
        ],
    }

    for section, keys in override_keys.items():
        if section not in cfg or section not in saved_cfg:
            continue
        with open_dict(cfg[section]):
            for key in keys:
                if key in saved_cfg[section]:
                    cfg[section][key] = saved_cfg[section][key]
    return cfg


# TODO: refactor how configs are resumed (need old cfg.model and cfg.train but probably not general)
def get_resume(cfg, model_kwargs):
    """Resumes a run. It loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()

    base_name = cfg.general.name
    append_suffix = bool(getattr(cfg.general, "append_resume_suffix", True))
    name = f"{base_name}_resume" if append_suffix else base_name

    resume = cfg.general.test_only
    val_samples_to_generate = cfg.general.val_samples_to_generate
    test_samples_to_generate = cfg.general.test_samples_to_generate
    gpus = cfg.general.gpus

    try:
        model = Spec2MolDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    except KeyError as e:
        if str(e).strip("'\"") != "pytorch-lightning_version":
            raise
        logging.warning(
            "Checkpoint %s is not a full Lightning checkpoint. "
            "Using the current config and loading weights only.",
            resume,
        )
        cfg = copy_runtime_overrides(saved_cfg, saved_cfg)
        return cfg, None

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.val_samples_to_generate = val_samples_to_generate
    cfg.general.test_samples_to_generate = test_samples_to_generate
    cfg.general.gpus = gpus
    #cfg.general.append_resume_suffix = append_suffix

    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    cfg = copy_runtime_overrides(cfg, saved_cfg)
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """Resumes a run. It loads previous config but allows to make some changes (used for resuming training)."""
    saved_cfg = cfg.copy()
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    resume_path = os.path.join(root_dir, cfg.general.resume)
    model = Spec2MolDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)

    new_cfg = model.cfg
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    append_suffix = bool(getattr(cfg.general, "append_resume_suffix", True))
    new_cfg.general.resume = resume_path
    new_cfg.general.name = f"{new_cfg.general.name}_resume" if append_suffix else new_cfg.general.name
    #new_cfg.general.append_resume_suffix = append_suffix

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)
    return new_cfg, model

def apply_encoder_finetuning(model, strategy):    
    if strategy is None:
        pass
    elif strategy == 'freeze':
        for param in model.encoder.parameters():
            param.requires_grad = False
    elif strategy == 'ft-unfold':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer != '2':
                param[1].requires_grad = False
    elif strategy == 'freeze-unfold':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer == '2':
                param[1].requires_grad = False
    elif strategy == 'ft-transformer':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer != '0':
                param[1].requires_grad = False
    elif strategy == 'freeze-transformer':
        for param in model.encoder.named_parameters():
            layer = param[0].split('.')[1]
            if layer == '0':
                param[1].requires_grad = False
    else:
        raise NotImplementedError(f'Unknown Finetune Strategy: {strategy}')
    
def apply_decoder_finetuning(model, strategy):
    if strategy is None:
        pass
    elif strategy == 'freeze':
        for param in model.decoder.parameters():
            param.requires_grad = False
    elif strategy == 'ft-input':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name not in ['mlp_in_X', 'mlp_in_E', 'mlp_in_y']:
                p[1].requires_grad = False
    elif strategy == 'freeze-input':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name in ['mlp_in_X', 'mlp_in_E', 'mlp_in_y']:
                p[1].requires_grad = False
    elif strategy == 'ft-transformer':
        for param in model.decoder.parameters():
            param.requires_grad = False
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = True
    elif strategy == 'freeze-transformer':
        for param in model.decoder.tf_layers.parameters():
            param.requires_grad = False
    elif strategy == 'ft-output':
        for p in model.decoder.named_parameters():
            layer_name = p[0].split('.')[0]
            if layer_name not in ['mlp_out_X', 'mlp_out_E', 'mlp_out_y']:
                p[1].requires_grad = False
    else:
        raise NotImplementedError(f'Unknown Finetune Strategy: {strategy}')

def load_weights(model, path):
    """
    Loads only the weights from a checkpoint file into the model without loading the full Lightning module.
    
    Args:
        model: The model to load weights into
        path: Path to the checkpoint file
        
    Returns:
        The model with loaded weights
    """
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Filter out keys that don't match the model (for partial loading)
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    
    # Load the weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    logging.info(f"Loaded weights from {path}")
    logging.info(f"Missing keys: {missing_keys}")
    logging.info(f"Unexpected keys: {unexpected_keys}")
    
    return model

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    logger = logging.getLogger("msms_main")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    path = os.path.join("msms_main.log")
    fh = logging.FileHandler(path)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logging.info(cfg)

    dataset_config = cfg["dataset"]

    if dataset_config["name"] not in ("canopus", "msg"):
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    datamodule = spec2mol_dataset.Spec2MolDataModule(cfg) # TODO: Add hyper for n_bits
    dataset_infos = spec2mol_dataset.Spec2MolDatasetInfos(datamodule, cfg)

    domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features, domain_features=domain_features)

    logging.info("Dataset infos:", dataset_infos.output_dims)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)

    # We do not evaluate novelty during training
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features}

    if cfg.general.test_only:
        # When testing, previous configuration is fully loaded
        cfg, _ = get_resume(cfg, model_kwargs)
        #os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # When resuming, we can override some parts of previous configuration
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        #os.chdir(cfg.general.resume.split('checkpoints')[0])

    os.makedirs('preds/', exist_ok=True)
    os.makedirs('logs/', exist_ok=True)
    os.makedirs('logs/' + cfg.general.name, exist_ok=True)

    model = Spec2MolDenoisingDiffusion(cfg=cfg, **model_kwargs)

    ##################
    if cfg.general.test_only:
        logging.info(f"Loading weights (no hyperparameter restore) from {cfg.general.test_only}")
        model = load_weights(model, cfg.general.test_only)  # 只加载 state_dict
        ckpt_path = None  # 后面 test 不再让 Lightning 还原超参
    else:
        ckpt_path = cfg.general.resume  # 训练/验证用原逻辑

    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    if cfg.train.save_model: # TODO: More advanced checkpointing
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", # best (top-5) checkpoints
                                              filename='{epoch}',
                                              monitor='val/NLL',
                                              save_top_k=5,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1) # most recent checkpoint
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    name = cfg.general.name
    if name == 'debug':
        logging.warning("Run is called 'debug' -- it will run with fast_dev_run. ")

    loggers = [
        CSVLogger(save_dir=f"logs/{name}", name=name),
        WandbLogger(name=name, save_dir=f"logs/{name}", project=cfg.general.wandb_name, log_model=False, config=utils.cfg_to_dict(cfg))
    ]

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints   ddp_find_unused_parameters_true
                      accelerator='gpu' if use_gpu else 'cpu',
                      devices=cfg.general.gpus if use_gpu else 1,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      callbacks=callbacks,
                      log_every_n_steps=50 if name != 'debug' else 1,
                      logger=loggers)

    apply_encoder_finetuning(model, cfg.general.encoder_finetune_strategy)
    apply_decoder_finetuning(model, cfg.general.decoder_finetune_strategy)

    if cfg.general.load_weights is not None:
        logging.info(f"Loading weights from {cfg.general.load_weights}")
        model = load_weights(model, cfg.general.load_weights)

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.checkpoint_strategy)

    else:
        trainer.test(model, datamodule=datamodule, ckpt_path=None)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            for file in os.listdir(directory):
                if file.endswith(".ckpt"):
                    ckpt_path_iter = os.path.join(directory, file)
                    logging.info(f"Loading weights {ckpt_path_iter}")
                    model = load_weights(model, ckpt_path_iter)
                    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == '__main__':
    main()
