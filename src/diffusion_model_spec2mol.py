import os
import time
import logging
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL, CrossEntropyMetric
from src.metrics.diffms_metrics import K_ACC_Collection, K_SimilarityCollection, Validity
from src import utils
from src.mist.models.spectra_encoder import SpectraEncoderGrowing

from src.mist.models.dreams_encoder import DreaMSEncoder

from src.mist.utils.chem_utils import ION_LST
import numpy as np


class Spec2MolDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.decoder_dtype = torch.float32
        self.T = cfg.model.diffusion_steps


        #断点重启
        self.test_resume_enabled = bool(getattr(cfg.general, "test_resume_enabled", True))
        self.test_resume_strict_pair = bool(getattr(cfg.general, "test_resume_strict_pair", True))
        self.test_resume_overwrite_broken = bool(getattr(cfg.general, "test_resume_overwrite_broken", True))

        self._test_resume_hit = 0  # 命中缓存（跳过采样）的batch数
        self._test_resume_miss = 0  # 未命中缓存（需要重跑）的batch数
        self._test_resume_broken = 0  # 缓存损坏回退重跑的batch数

        self._test_metric_skip = 0  # 新增这里

        self.test_resume_skip_loss_on_hit = bool(
            getattr(cfg.general, "test_resume_skip_loss_on_hit", True)
        )


        self.use_sampling_acceleration = bool(
            getattr(cfg.model, "use_sampling_acceleration", True)
        )
        self.sampling_steps = int(getattr(cfg.model, "sampling_steps", self.T))
        self.sampling_schedule = getattr(cfg.model, "sampling_schedule", "uniform")
        self.sampling_tuned_steps = getattr(cfg.model, "sampling_tuned_steps", None)

        #########################################################################################
        # ---- Sampling acceleration (no encoder/decoder changes) ----
        self.sampling_steps = int(getattr(cfg.model, "sampling_steps", self.T))
        self.sampling_schedule = getattr(cfg.model, "sampling_schedule", "uniform")
        # Optional manually tuned timetable, e.g. [500, 420, 350, ..., 0]
        self.sampling_tuned_steps = getattr(cfg.model, "sampling_tuned_steps", None)

        self.val_num_samples = cfg.general.val_samples_to_generate
        self.test_num_samples = cfg.general.test_samples_to_generate


        # ---- Lightweight sampling corrector (edge logits) ----
        self.use_sampling_corrector = bool(getattr(cfg.model, "use_sampling_corrector", False))
        self.corrector_every_n = int(getattr(cfg.model, "corrector_every_n", 3))
        self.corrector_temperature = float(getattr(cfg.model, "corrector_temperature", 1.0))
        self.corrector_edge_prior_strength = float(getattr(cfg.model, "corrector_edge_prior_strength", 0.0))
        self.corrector_apply_until_t = float(getattr(cfg.model, "corrector_apply_until_t", 0.35))  # only late steps
        self.corrector_logit_clip = float(getattr(cfg.model, "corrector_logit_clip", 20.0))





        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        ###############################
        # precompute edge prior log-prob once (only if enabled)
        if self.use_sampling_corrector:
            edge_prior = self.dataset_info.edge_types.float()
            edge_prior = edge_prior / torch.clamp(edge_prior.sum(), min=1e-12)
            edge_prior_log = torch.log(torch.clamp(edge_prior, min=1e-12))
            self.register_buffer("corrector_edge_prior_log", edge_prior_log)


        # ---- Per-sample early stop during sampling ----
        self.use_per_sample_early_stop = bool(getattr(cfg.model, "use_per_sample_early_stop", False))
        self.early_stop_change_threshold = float(getattr(cfg.model, "early_stop_change_threshold", 0.002))
        self.early_stop_patience = int(getattr(cfg.model, "early_stop_patience", 3))
        self.early_stop_min_steps = int(getattr(cfg.model, "early_stop_min_steps", 5))



        # ---- Multi-trajectory + lightweight rerank ----
        self.use_multitraj_rerank = bool(getattr(cfg.model, "use_multitraj_rerank", False))
        self.multitraj_candidates = int(getattr(cfg.model, "multitraj_candidates", 4))

        # rerank weights
        self.rerank_w_valid = float(getattr(cfg.model, "rerank_w_valid", 1.0))
        self.rerank_w_conf = float(getattr(cfg.model, "rerank_w_conf", 0.25))
        self.rerank_w_prior = float(getattr(cfg.model, "rerank_w_prior", 0.05))

        # edge prior for score proxy（只在启用时初始化）
        if self.use_multitraj_rerank:
            edge_prior = dataset_infos.edge_types.float()
            edge_prior = edge_prior / torch.clamp(edge_prior.sum(), min=1e-12)
            self.register_buffer("rerank_edge_prior_log", torch.log(torch.clamp(edge_prior, min=1e-12)))


        # ---- Conditional timestep tuner ----
        self.use_conditional_timestep_tuner = bool(getattr(cfg.model, "use_conditional_timestep_tuner", False))
        self.conditional_tuner_mode = getattr(cfg.model, "conditional_tuner_mode", "rule")  # "rule" | "mlp"

        self.ctt_min_steps = int(getattr(cfg.model, "ctt_min_steps", max(10, self.sampling_steps // 2)))
        self.ctt_max_steps = int(getattr(cfg.model, "ctt_max_steps", self.sampling_steps))
        self.ctt_default_power = float(getattr(cfg.model, "ctt_default_power", 1.0))
        self.ctt_entropy_eps = float(getattr(cfg.model, "ctt_entropy_eps", 1e-8))

        if self.use_conditional_timestep_tuner and self.conditional_tuner_mode == "mlp":
            # in: [num_peaks_norm, entropy_norm, precursor_norm]
            self.ctt_mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.SiLU(),
                nn.Linear(32, 2),
            )
            # 初始化为“接近全局默认”，避免刚启用就发散
            with torch.no_grad():
                self.ctt_mlp[-1].weight.zero_()
                self.ctt_mlp[-1].bias.zero_()



        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_k_acc = K_ACC_Collection(list(range(1, self.val_num_samples + 1)))
        self.val_sim_metrics = K_SimilarityCollection(list(range(1, self.val_num_samples + 1)))
        self.val_validity = Validity()
        self.val_CE = CrossEntropyMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_k_acc = K_ACC_Collection(list(range(1, self.test_num_samples + 1)))
        self.test_sim_metrics = K_SimilarityCollection(list(range(1, self.test_num_samples + 1)))
        self.test_validity = Validity()
        self.test_CE = CrossEntropyMetric()

        self.train_metrics = train_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.ion_alpha_max = float(getattr(cfg.model, "ion_alpha_max", 0.1))
        self.heavy_alpha_max = float(getattr(cfg.model, "heavy_alpha_max", 0.1))
        ##################


        self.decoder = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        try:
            if cfg.general.decoder is not None:
                state_dict = torch.load(cfg.general.decoder, map_location='cpu')
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        k = k[6:]
                        cleaned_state_dict[k] = v

                self.decoder.load_state_dict(cleaned_state_dict)
        except Exception as e:
            logging.info(f"Could not load decoder: {e}")

        hidden_size = 256
        try:
            hidden_size = cfg.model.encoder_hidden_dim
        except:
            print("No hidden size specified, using default value of 256")

        magma_modulo = 512
        try:
            magma_modulo = cfg.model.encoder_magma_modulo
        except:
            print("No magma modulo specified, using default value of 512")
        
        encoder_type = getattr(cfg.model, "encoder_type", "mist")

        if encoder_type == "dreams":
            self.encoder = DreaMSEncoder(
                output_size=4096,
                hidden_size=cfg.model.dreams_hidden_dim,
                num_layers=cfg.model.dreams_num_layers,
                num_heads=cfg.model.dreams_num_heads,
                num_freq=cfg.model.dreams_num_freq,
                dropout=cfg.model.dreams_dropout,
                max_peaks=getattr(cfg.dataset, "max_peaks", 100),
                use_int_preds=cfg.model.dreams_use_int_preds,
            )
        else:
            self.encoder = SpectraEncoderGrowing(
                inten_transform='float',
                inten_prob=0.1,
                remove_prob=0.5,
                peak_attn_layers=2,
                num_heads=8,
                pairwise_featurization=True,
                embed_instrument=False,
                cls_type='ms1',
                set_pooling='cls',
                spec_features='peakformula',
                mol_features='fingerprint',
                form_embedder='pos-cos',
                output_size=4096,
                hidden_size=hidden_size,
                spectra_dropout=0.1,
                top_layers=1,
                refine_layers=4,
                magma_modulo=magma_modulo,
                use_diff_attn=False,
            )
        
        try:
            if cfg.general.encoder is not None:
                self.encoder.load_state_dict(torch.load(cfg.general.encoder), strict=True)
        except Exception as e:
            logging.info(f"Could not load encoder: {e}")

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)
        self.denoise_nodes = getattr(cfg.dataset, 'denoise_nodes', False)
        self.merge = getattr(cfg.dataset, 'merge', 'none')

        if self.merge == 'merge-encoder_output-linear':
            self.merge_function = nn.Linear(hidden_size, cfg.dataset.morgan_nbits)
        elif self.merge == 'merge-encoder_output-mlp':
            self.merge_function = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.SiLU(),
                nn.Linear(1024, cfg.dataset.morgan_nbits)
            )
        elif self.merge == 'downproject_4096':
            self.merge_function = nn.Linear(4096, cfg.dataset.morgan_nbits)


        ####################################
        # ===== MIST-only ion bias adapter (does not change decoder structure) =====
        self.use_ion_bias = bool(getattr(cfg.model, "use_ion_bias", False))
        self.ion_bias_hidden = int(getattr(cfg.model, "ion_bias_hidden", 128))
        self.ion_emb_dim = int(getattr(cfg.model, "ion_emb_dim", 32))
        self.ion_bias_dropout = float(getattr(cfg.model, "ion_bias_dropout", 0.1))

        if self.use_ion_bias:
            # ion index comes from PeakFormula featurizer's ion_vec
            self.num_ion_types = len(ION_LST)
            self.ion_embed = nn.Embedding(self.num_ion_types, self.ion_emb_dim)

            # precursor embedding + pooled fragment embedding + 2 meta stats
            in_dim = self.ion_emb_dim * 2 + 2
            self.ion_bias_mlp = nn.Sequential(
                nn.Linear(in_dim, self.ion_bias_hidden),
                nn.SiLU(),
                nn.Dropout(self.ion_bias_dropout),
                nn.Linear(self.ion_bias_hidden, self.ydim_output),
            )

            # zero init => initially identical behavior to original framework
            #self.ion_bias_alpha = nn.Parameter(torch.zeros(1))

            alpha_init = float(getattr(cfg.model, "ion_bias_alpha_init", 0.0))
            self.ion_bias_alpha = nn.Parameter(torch.tensor([alpha_init], dtype=torch.float32))
        ##############################################

        # ===== 在 __init__ 里（建议放在 ion_bias 初始化后）=====

        # No-leak heavy-atom bias: use encoder hidden state aux["h0"] only
        self.use_heavy_atom_bias = bool(getattr(cfg.model, "use_heavy_atom_bias", False))
        self.heavy_atom_hidden = int(getattr(cfg.model, "heavy_atom_hidden", 128))
        self.heavy_atom_bias_hidden = int(getattr(cfg.model, "heavy_atom_bias_hidden", 128))
        self.heavy_atom_max = float(getattr(cfg.model, "heavy_atom_max", 64.0))  # normalization scale

        if self.use_heavy_atom_bias:
            # predict normalized heavy-atom count from h0 (no GT graph used)
            self.heavy_atom_predictor = nn.Sequential(
                nn.Linear(hidden_size, self.heavy_atom_hidden),
                nn.SiLU(),
                nn.Dropout(float(getattr(cfg.model, "heavy_atom_dropout", 0.1))),
                nn.Linear(self.heavy_atom_hidden, 1),
            )

            # map heavy-atom features -> y-dim bias
            self.heavy_atom_to_y = nn.Sequential(
                nn.Linear(2, self.heavy_atom_bias_hidden),
                nn.SiLU(),
                nn.Dropout(float(getattr(cfg.model, "heavy_atom_bias_dropout", 0.1))),
                nn.Linear(self.heavy_atom_bias_hidden, self.ydim_output),
            )

            # learnable weight (same style as ion_bias_alpha)
            heavy_alpha_init = float(getattr(cfg.model, "heavy_atom_alpha_init", 0.0))
            self.heavy_atom_alpha = nn.Parameter(
                torch.tensor([heavy_alpha_init], dtype=torch.float32)
            )


        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            logging.info(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.best_val_nll = 1e8
        self.val_counter = 1

    # 放在类里，统一封装
    def _safe_update_test_metrics_one_sample(self, pred_mols, true_mol):
        try:
            self.test_k_acc.update(pred_mols, true_mol)
            self.test_sim_metrics.update(pred_mols, true_mol)
            self.test_validity.update(pred_mols)
            return True
        except Exception as e:
            self._test_metric_skip += 1
            logging.warning(f"[TEST-METRIC-SKIP] skip one sample due to metric error: {e}")
            return False

    def training_step(self, batch, i):
        output, aux = self.encoder(batch)

        data = batch["graph"]
        '''
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)
        '''
        self._apply_merge_and_ion_bias(batch, output, aux, data)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=False)
 
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=False)

        return {'loss': loss}

    def configure_optimizers(self):
        if self.cfg.train.scheduler == 'const':
            return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.scheduler == 'one_cycle':
            opt = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True, weight_decay=self.cfg.train.weight_decay)
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.cfg.train.lr, total_steps=stepping_batches, pct_start=self.cfg.train.pct_start)
            lr_scheduler = {
                'scheduler': scheduler,
                'name': 'learning_rate',
                'interval':'step',
                'frequency': 1,
            }

            return [opt], [lr_scheduler]
        else:
            raise ValueError('Unknown Scheduler')

    def on_fit_start(self) -> None:
        if self.global_rank == 0:
            logging.info(f"Size of the input features: X-{self.Xdim}, E-{self.Edim}, y-{self.ydim}")
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        
    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        to_log["train_epoch/epoch"] = float(self.current_epoch)
        to_log["train_epoch/time"] = time.time() - self.start_epoch_time

        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        for key, value in epoch_at_metrics.items():
            to_log[f"train_epoch/{key}"] = value
        for key, value in epoch_bond_metrics.items():
            to_log[f"train_epoch/{key}"] = value

        self.log_dict(to_log, sync_dist=True)
        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE']:.2f} -- E_CE: {to_log['train_epoch/E_CE']:.2f} -- time: {to_log['train_epoch/time']:.2f}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_k_acc.reset()
        self.val_sim_metrics.reset()
        self.val_validity.reset()
        self.val_CE.reset()
        if self.global_rank == 0:
            self.val_counter += 1



    def validation_step(self, batch, i):
        if self.global_rank == 0 and i == 0:
            print(batch.keys())

        output, aux = self.encoder(batch)

        data = batch["graph"]
        '''
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)'''
        self._apply_merge_and_ion_bias(batch, output, aux, data)


        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.Y = data.y

        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)

        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))  # (bs * n * n, de)
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))   # (bs * n * n, de)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        self.val_CE(flat_pred_E, flat_true_E)

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))] # Is this correct?
            predicted_mols = [list() for _ in range(len(data))]
            for _ in range(self.val_num_samples):
                for idx, mol in enumerate(self.sample_batch(data)):
                    predicted_mols[idx].append(mol)
        
            for idx in range(len(data)):
                self.val_k_acc.update(predicted_mols[idx], true_mols[idx])
                self.val_sim_metrics.update(predicted_mols[idx], true_mols[idx])
                self.val_validity.update(predicted_mols[idx])

        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [
            self.val_nll.compute(), 
            self.val_X_kl.compute(), 
            self.val_E_kl.compute(),
            self.val_X_logp.compute(), 
            self.val_E_logp.compute(),
            self.val_CE.compute()
        ]

        log_dict = {
            "val/NLL": metrics[0],
            "val/X_KL": metrics[1],
            "val/E_KL": metrics[2],
            "val/X_logp": metrics[3],
            "val/E_logp": metrics[4],
            "val/E_CE": metrics[5]
        }

        if self.val_counter % self.cfg.general.sample_every_val == 0:
            for key, value in self.val_k_acc.compute().items():
                log_dict[f"val/{key}"] = value
            for key, value in self.val_sim_metrics.compute().items():
                log_dict[f"val/{key}"] = value
            log_dict["val/validity"] = self.val_validity.compute()

        self.log_dict(log_dict, sync_dist=True)

        if self.global_rank == 0:
            logging.info(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL: {metrics[1] :.2f} -- Val Edge type KL: {metrics[2] :.2f} -- Val Edge type logp: {metrics[4] :.2f} -- Val Edge type CE: {metrics[5] :.2f}")

            val_nll = metrics[0]
            if val_nll < self.best_val_nll:
                self.best_val_nll = val_nll
            logging.info(f"Val NLL: {val_nll :.4f} \t Best Val NLL:  {self.best_val_nll}")

    
    def on_test_epoch_start(self) -> None:
        if self.global_rank == 0:
            logging.info("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_k_acc.reset()
        self.test_sim_metrics.reset()
        self.test_validity.reset()
        self.test_CE.reset()

        self._test_resume_hit = 0
        self._test_resume_miss = 0
        self._test_resume_broken = 0

        logging.info(
            f"[TEST-RESUME] enabled={self.test_resume_enabled}, strict_pair={self.test_resume_strict_pair}, "
            f"overwrite_broken={self.test_resume_overwrite_broken}"
        )

        logging.info(
            f"[TEST] test_num_samples={self.test_num_samples}, "
            f"diffusion_steps={self.T}, "
            f"merge={self.merge}, "
            f"encoder_finetune={self.cfg.general.encoder_finetune_strategy}, "
            f"decoder_finetune={self.cfg.general.decoder_finetune_strategy}"
        )



    def test_step(self, batch, i):
        # ===== Fast path: 命中缓存直接返回（跳过前向和loss）=====
        cached = None
        if self.test_resume_enabled:
            cached = self._try_load_cached_test_batch(i)

        if cached is not None:
            predicted_mols, true_mols = cached
            self._test_resume_hit += 1

            #for idx in range(len(true_mols)):
            #    self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
            #    self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
            #    self.test_validity.update(predicted_mols[idx])

            for idx in range(len(true_mols)):
                self._safe_update_test_metrics_one_sample(predicted_mols[idx], true_mols[idx])

            if self.test_resume_skip_loss_on_hit:
                # Lightning test_step 返回值可选；这里返回0张量避免潜在聚合报错
                return {"loss": torch.tensor(0.0, device=self.device)}

        # ===== Slow path: 未命中缓存，按原逻辑计算并写缓存=====
        self._test_resume_miss += 1

        output, aux = self.encoder(batch)
        data = batch["graph"]
        self._apply_merge_and_ion_bias(batch, output, aux, data)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        pred = self.forward(noisy_data, extra_data, node_mask)
        pred.X = dense_data.X
        pred.Y = data.y

        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)

        true_E = torch.reshape(dense_data.E, (-1, dense_data.E.size(-1)))
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        self.test_CE(flat_pred_E, flat_true_E)

        true_mols = [Chem.inchi.MolFromInchi(data.get_example(idx).inchi) for idx in range(len(data))]
        predicted_mols = [list() for _ in range(len(data))]

        for _ in range(self.test_num_samples):
            for idx, mol in enumerate(self.sample_batch(data)):
                predicted_mols[idx].append(mol)

        self._atomic_pickle_dump(predicted_mols, self._test_pred_path(i))
        self._atomic_pickle_dump(true_mols, self._test_true_path(i))

        #for idx in range(len(data)):
        #    self.test_k_acc.update(predicted_mols[idx], true_mols[idx])
        #    self.test_sim_metrics.update(predicted_mols[idx], true_mols[idx])
        #    self.test_validity.update(predicted_mols[idx])

        for idx in range(len(data)):
            self._safe_update_test_metrics_one_sample(predicted_mols[idx], true_mols[idx])

        return {"loss": nll}

    def on_test_epoch_end(self) -> None:
        # 先打resume统计
        logging.info(
            f"[TEST-RESUME] hit={self._test_resume_hit}, miss={self._test_resume_miss}, broken={self._test_resume_broken}"
        )

        logging.info(f"[TEST-METRIC-SKIP] skipped_samples={self._test_metric_skip}")

        # 生成质量指标（一定有）
        log_dict = {}
        for key, value in self.test_k_acc.compute().items():
            log_dict[f"test/{key}"] = value
        for key, value in self.test_sim_metrics.compute().items():
            log_dict[f"test/{key}"] = value
        log_dict["test/validity"] = self.test_validity.compute()
        self.log_dict(log_dict, sync_dist=True)

        # 如果启用“命中即跳过loss”且存在hit，则loss类指标不再可信，直接跳过
        if self.test_resume_skip_loss_on_hit and self._test_resume_hit > 0:
            logging.info("[TEST-RESUME] loss/NLL metrics skipped because cached batches bypassed forward.")
            return
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [
            self.test_nll.compute(), 
            self.test_X_kl.compute(), 
            self.test_E_kl.compute(),
            self.test_X_logp.compute(), 
            self.test_E_logp.compute(),
            self.test_CE.compute()
        ]

        log_dict = {
            "test/NLL": metrics[0],
            "test/X_KL": metrics[1],
            "test/E_KL": metrics[2],
            "test/X_logp": metrics[3],
            "test/E_logp": metrics[4],
            "test/E_CE": metrics[5]
        }

        self.log_dict(log_dict, sync_dist=True)

        
        
    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, y, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = X
        if self.denoise_nodes:
            X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
        """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, y, node_mask)

        #loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())
        metric_X = self.test_X_logp if test else self.val_X_logp
        metric_E = self.test_E_logp if test else self.val_E_logp
        loss_term_0 = metric_X(X * prob0.X.log()) + metric_E(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        if test:
            nll = self.test_nll(nlls)
        else:
            nll = self.val_nll(nlls)

        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.decoder(X, E, y, node_mask)

    '''
    @torch.no_grad()
    def sample_batch(self, data: Batch) -> Batch:
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = dense_data.X, z_T.E, data.y

        assert (E == torch.transpose(E, 1, 2)).all()

        timesteps = self._build_sampling_timesteps()  # e.g. [500, 420, ..., 0]
        sampled_s = None

        # Iteratively sample p(z_s | z_t) on tuned timetable
        for i in range(len(timesteps) - 1):
            t_val = timesteps[i]
            s_val = timesteps[i + 1]

            t_array = torch.full((len(data), 1), float(t_val), dtype=torch.float32, device=self.device)
            s_array = torch.full((len(data), 1), float(s_val), dtype=torch.float32, device=self.device)

            #sampled_s, __ = self.sample_p_zs_given_zt(s_array, t_array, X, E, y, node_mask)
            sampled_s, __ = self.sample_p_zs_given_zt(s_array, t_array, X, E, y, node_mask, step_idx=i)

            # Keep original behavior (node type usually not denoised in this setup)
            _, E, y = sampled_s.X, sampled_s.E, data.y

        # Final decode
        sampled_s.X = X
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, data.y

        mols = []
        for nodes, adj_mat in zip(X, E):
            mols.append(self.visualization_tools.mol_from_graphs(nodes, adj_mat))

        return mols
    '''

    @torch.no_grad()
    def _sample_batch_single(self, data: Batch, return_score=False):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)

        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = dense_data.X, z_T.E, data.y
        timesteps = self._build_sampling_timesteps()
        B, n = E.shape[0], E.shape[1]

        last_conf = torch.zeros(B, device=self.device)

        if not self.use_per_sample_early_stop:
            for i in range(len(timesteps) - 1):
                t_val, s_val = timesteps[i], timesteps[i + 1]
                t_array = torch.full((B, 1), float(t_val), dtype=torch.float32, device=self.device)
                s_array = torch.full((B, 1), float(s_val), dtype=torch.float32, device=self.device)

                if return_score:
                    sampled_s, __, edge_conf = self.sample_p_zs_given_zt(
                        s_array, t_array, X, E, y, node_mask, step_idx=i, return_edge_conf=True
                    )
                    last_conf = edge_conf
                else:
                    sampled_s, __ = self.sample_p_zs_given_zt(
                        s_array, t_array, X, E, y, node_mask, step_idx=i, return_edge_conf=False
                    )
                _, E, y = sampled_s.X, sampled_s.E, data.y
        else:
            active = torch.ones(B, dtype=torch.bool, device=self.device)
            stable_counts = torch.zeros(B, dtype=torch.long, device=self.device)

            for i in range(len(timesteps) - 1):
                if not torch.any(active):
                    break

                t_val, s_val = timesteps[i], timesteps[i + 1]
                active_idx = torch.where(active)[0]
                b_active = active_idx.numel()
                t_array = torch.full((b_active, 1), float(t_val), dtype=torch.float32, device=self.device)
                s_array = torch.full((b_active, 1), float(s_val), dtype=torch.float32, device=self.device)
                E_prev_active = E[active_idx]

                if return_score:
                    sampled_s_active, __, edge_conf_active = self.sample_p_zs_given_zt(
                        s_array, t_array, X[active_idx], E_prev_active, y[active_idx], node_mask[active_idx],
                        step_idx=i, return_edge_conf=True
                    )
                    last_conf[active_idx] = edge_conf_active
                else:
                    sampled_s_active, __ = self.sample_p_zs_given_zt(
                        s_array, t_array, X[active_idx], E_prev_active, y[active_idx], node_mask[active_idx],
                        step_idx=i, return_edge_conf=False
                    )

                E_new_active = sampled_s_active.E
                E[active_idx] = E_new_active

                if i < self.early_stop_min_steps:
                    continue

                prev_cls = E_prev_active.argmax(dim=-1)
                new_cls = E_new_active.argmax(dim=-1)
                changed = (prev_cls != new_cls)

                nm = node_mask[active_idx].bool()
                valid = nm.unsqueeze(1) & nm.unsqueeze(2)
                diag = torch.eye(n, dtype=torch.bool, device=self.device).unsqueeze(0)
                valid = valid & (~diag)

                changed_ratio = ((changed & valid).float().sum(dim=(1, 2)) /
                                 torch.clamp(valid.float().sum(dim=(1, 2)), min=1.0))
                is_stable = changed_ratio <= self.early_stop_change_threshold
                cnt = stable_counts[active_idx]
                cnt = torch.where(is_stable, cnt + 1, torch.zeros_like(cnt))
                stable_counts[active_idx] = cnt
                done_local = cnt >= self.early_stop_patience
                if torch.any(done_local):
                    active[active_idx[done_local]] = False

        sampled_final = utils.PlaceHolder(
            X=X,
            E=E,
            y=torch.zeros((B, 0), device=X.device, dtype=X.dtype),
        ).mask(node_mask, collapse=True)

        Xc, Ec = sampled_final.X, sampled_final.E
        mols, valid_flags = [], []
        for nodes, adj_mat in zip(Xc, Ec):
            mol = self.visualization_tools.mol_from_graphs(nodes, adj_mat)
            mols.append(mol)
            valid_flags.append(1.0 if mol is not None else 0.0)

        if not return_score:
            return mols

        valid_score = torch.tensor(valid_flags, device=self.device, dtype=torch.float32)
        prior_score = self._edge_prior_score_from_E(E, node_mask)
        total_score = self.rerank_w_valid * valid_score + self.rerank_w_conf * last_conf + self.rerank_w_prior * prior_score
        return mols, total_score

    @torch.no_grad()
    def sample_batch(self, data: Batch) -> Batch:
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)

        X, E, y = dense_data.X, z_T.E, data.y
        B = E.shape[0]

        # fast path: 关闭时完全走你原来的批量采样逻辑（零额外开销）
        if not self.use_conditional_timestep_tuner:
            # === 这里保留你当前已有的 sample_batch 主体 ===
            # 例如你的固定timesteps / early-stop / corrector逻辑
            ...

            # Fast path: disabled => zero extra multi-trajectory overhead
            if not self.use_multitraj_rerank:
                return self._sample_batch_single(data, return_score=False)

            K = max(1, self.multitraj_candidates)
            all_mols = []
            all_scores = []

            for _ in range(K):
                mols_k, scores_k = self._sample_batch_single(data, return_score=True)
                all_mols.append(mols_k)
                all_scores.append(scores_k)

            # pick best trajectory per sample
            score_mat = torch.stack(all_scores, dim=0)  # [K,B]
            best_k = score_mat.argmax(dim=0)  # [B]
            B = best_k.numel()

            best_mols = []
            for b in range(B):
                kk = int(best_k[b].item())
                best_mols.append(all_mols[kk][b])

            return best_mols
        else:
            # 条件化路径：逐样本使用不同时间表
            # 仅在启用时才有这部分额外计算
            for b in range(B):
                X_b = X[b:b + 1]
                E_b = E[b:b + 1]
                y_b = y[b:b + 1]
                nm_b = node_mask[b:b + 1]

                timesteps_b = self._get_conditional_timesteps_for_sample(data, b)

                for i in range(len(timesteps_b) - 1):
                    t_val = timesteps_b[i]
                    s_val = timesteps_b[i + 1]
                    t_array = torch.full((1, 1), float(t_val), dtype=torch.float32, device=self.device)
                    s_array = torch.full((1, 1), float(s_val), dtype=torch.float32, device=self.device)

                    sampled_s, __ = self.sample_p_zs_given_zt(
                        s_array, t_array, X_b, E_b, y_b, nm_b, step_idx=i
                    )
                    _, E_b, y_b = sampled_s.X, sampled_s.E, y_b

                E[b:b + 1] = E_b

            sampled_final = utils.PlaceHolder(
                X=X,
                E=E,
                y=torch.zeros((B, 0), device=X.device, dtype=X.dtype),
            ).mask(node_mask, collapse=True)

            X, E = sampled_final.X, sampled_final.E
            mols = [self.visualization_tools.mol_from_graphs(nodes, adj_mat) for nodes, adj_mat in zip(X, E)]
            return mols


    def sample_p_zs_given_zt(self, s_int, t_int, X_t, E_t, y_t, node_mask, step_idx=0, return_edge_conf=False):
        """Samples zs ~ p(zs | zt) with possibly skipped timesteps (s < t)."""
        bs, n, _ = X_t.shape

        # normalized times for model conditioning
        s = s_int / self.T
        t = t_int / self.T

        # alpha_bar at s,t
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_int=s_int.long())
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=t_int.long())

        # Transition matrices
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)   # q(z_t | z_0)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)   # q(z_s | z_0)
        Qt = self._get_Qt_between(alpha_s_bar, alpha_t_bar)                # q(z_t | z_s), jump-aware

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0

        ##############################################
        # edge logits corrector (optional, gated)
        pred_E_logits = pred.E
        if self.use_sampling_corrector:
            pred_E_logits = self._apply_sampling_corrector(pred_E_logits, t, step_idx)

        #pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0
        pred_E = F.softmax(pred_E_logits, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
        )

        # X
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        # E
        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        ################
        edge_conf = None
        if return_edge_conf:
            # confidence proxy: mean max prob over valid edges
            maxp = prob_E.max(dim=-1).values  # [B,N,N]
            nm = node_mask.bool()
            valid = nm.unsqueeze(1) & nm.unsqueeze(2)
            diag = torch.eye(n, dtype=torch.bool, device=self.device).unsqueeze(0)
            valid = valid & (~diag)
            num = (maxp * valid.float()).sum(dim=(1, 2))
            den = torch.clamp(valid.float().sum(dim=(1, 2)), min=1.0)
            edge_conf = num / den

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        #return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)
        if return_edge_conf:
            return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(
                y_t), edge_conf
        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


    ###################################
    def _compute_ion_bias(self, batch, base_y):
        """
        Build precursor/fragment ion bias from MIST PeakFormula fields:
        - ion_vec: [B, L]
        - intens: [B, L]
        - num_peaks: [B]
        Returns: [B, y_dim] additive bias.
        """
        if (not self.use_ion_bias) or ("ion_vec" not in batch) or ("num_peaks" not in batch):
            return torch.zeros_like(base_y)

        ion_vec = batch["ion_vec"]          # float in dataloader; cast to long
        num_peaks = batch["num_peaks"]      # [B]
        intens = batch.get("intens", None)

        ion_idx = ion_vec.long().clamp(min=0, max=self.num_ion_types - 1)
        B, L = ion_idx.shape
        device = ion_idx.device

        # In MIST PeakFormula, cls token is appended at the end (ms1/zeros),
        # and carries root_ion (precursor ion)
        last_pos = (num_peaks - 1).clamp(min=0, max=L - 1).long()
        batch_ids = torch.arange(B, device=device)
        precursor_idx = ion_idx[batch_ids, last_pos]  # [B]
        precursor_emb = self.ion_embed(precursor_idx)  # [B, D]

        # Fragment ions: positions [0, num_peaks-2]
        pos = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        frag_len = (num_peaks - 1).clamp(min=0)            # [B]
        frag_mask = pos < frag_len.unsqueeze(1)            # [B, L]

        frag_emb_all = self.ion_embed(ion_idx)             # [B, L, D]
        if intens is None:
            frag_w = frag_mask.float()
        else:
            frag_w = intens.float() * frag_mask.float()    # [B, L]

        frag_w = frag_w / (frag_w.sum(dim=1, keepdim=True) + 1e-6)
        fragment_emb = (frag_emb_all * frag_w.unsqueeze(-1)).sum(dim=1)  # [B, D]

        # simple meta features
        l_denom = max(float(L), 1.0)
        num_peaks_norm = num_peaks.float().unsqueeze(1) / l_denom
        frag_ratio = frag_len.float().unsqueeze(1) / l_denom
        meta = torch.cat([num_peaks_norm, frag_ratio], dim=1)  # [B, 2]

        ion_feat = torch.cat([precursor_emb, fragment_emb, meta], dim=1)  # [B, 2D+2]
        ion_bias = self.ion_bias_mlp(ion_feat).to(dtype=base_y.dtype)

        return ion_bias

    # ===== 用这个版本替换你当前的 _apply_merge_and_ion_bias =====
    def _apply_merge_and_ion_bias(self, batch, output, aux, data):
        """
        Keep original merge behavior, then add optional ion/heavy-atom residual biases.
        """
        # ---- original merge logic ----
        if self.merge == 'mist_fp':
            data.y = aux["int_preds"][-1]
        if self.merge == 'merge-encoder_output-linear':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'merge-encoder_output-mlp':
            encoder_output = aux['h0']
            data.y = self.merge_function(encoder_output)
        elif self.merge == 'downproject_4096':
            data.y = self.merge_function(output)

        '''
        # ---- ion bias (if enabled) ----
        if self.use_ion_bias:
            y_bias = self._compute_ion_bias(batch, data.y)
            data.y = data.y + self.ion_bias_alpha * y_bias

        # ---- no-leak heavy-atom bias (from h0 only) ----
        if self.use_heavy_atom_bias:
            h_bias = self._compute_heavy_atom_bias_from_h0(aux, data.y)
            data.y = data.y + self.heavy_atom_alpha * h_bias
        '''

        y_base = data.y

        ion_term = 0.0
        if self.use_ion_bias:
            ion_bias = self._compute_ion_bias(batch, y_base)
            alpha_i = self.ion_alpha_max * torch.tanh(self.ion_bias_alpha)
            ion_term = alpha_i * ion_bias

        heavy_term = 0.0
        if self.use_heavy_atom_bias:
            heavy_bias = self._compute_heavy_atom_bias_from_h0(aux, y_base)
            alpha_h = self.heavy_alpha_max * torch.tanh(self.heavy_atom_alpha)
            heavy_term = alpha_h * heavy_bias

        data.y = y_base + ion_term + heavy_term


    # ===== 在类里新增函数 =====
    def _compute_heavy_atom_bias_from_h0(self, aux, base_y):
        """
        No-leak heavy-atom bias:
        only uses aux["h0"], never uses graph ground-truth atom types.
        Returns additive bias with shape [B, y_dim].
        """
        if (not self.use_heavy_atom_bias) or ("h0" not in aux):
            return torch.zeros_like(base_y)

        h0 = aux["h0"]  # [B, hidden_size]
        # unconstrained scalar -> [0,1] normalized heavy-atom proxy
        heavy_norm = torch.sigmoid(self.heavy_atom_predictor(h0))  # [B,1]

        # build 2-d features for stability
        heavy_count = heavy_norm * self.heavy_atom_max
        heavy_log = torch.log1p(heavy_count) / torch.log1p(
            torch.tensor(self.heavy_atom_max, device=heavy_count.device, dtype=heavy_count.dtype)
        )
        heavy_feat = torch.cat([heavy_norm, heavy_log], dim=1)  # [B,2]

        heavy_bias = self.heavy_atom_to_y(heavy_feat).to(dtype=base_y.dtype)  # [B,y_dim]
        return heavy_bias



    def on_fit_end(self) -> None:
        # 多卡下只让 rank0 打印一次
        if getattr(self, "global_rank", 0) != 0:
            return

        # ion alpha
        if hasattr(self, "ion_bias_alpha"):
            ion_raw = self.ion_bias_alpha.detach().float().item()
            if hasattr(self, "ion_alpha_max"):
                ion_eff = (self.ion_alpha_max * torch.tanh(self.ion_bias_alpha)).detach().float().item()
            else:
                ion_eff = ion_raw
            print(f"[FIT_END] ion_alpha_raw={ion_raw:.6f}, ion_alpha_effective={ion_eff:.6f}")

        # heavy-atom alpha
        if hasattr(self, "heavy_atom_alpha"):
            heavy_raw = self.heavy_atom_alpha.detach().float().item()
            if hasattr(self, "heavy_alpha_max"):
                heavy_eff = (self.heavy_alpha_max * torch.tanh(self.heavy_atom_alpha)).detach().float().item()
            else:
                heavy_eff = heavy_raw
            print(f"[FIT_END] heavy_alpha_raw={heavy_raw:.6f}, heavy_alpha_effective={heavy_eff:.6f}")

    def _build_sampling_timesteps(self):
        """
        Returns descending integer timesteps [T, ..., 0].
        Supports:
          1) manual tuned steps via cfg.model.sampling_tuned_steps
          2) generated schedule via sampling_steps + sampling_schedule
        """
        if not self.use_sampling_acceleration:
            return list(range(int(self.T), -1, -1))

        T = int(self.T)

        # 1) Manual tuned timetable (highest priority)
        if self.sampling_tuned_steps is not None and len(self.sampling_tuned_steps) > 0:
            ts = [int(x) for x in self.sampling_tuned_steps]
            ts = [min(max(t, 0), T) for t in ts]
            ts = sorted(set(ts), reverse=True)
            if ts[0] != T:
                ts = [T] + ts
            if ts[-1] != 0:
                ts = ts + [0]
            return ts

        # 2) Auto schedule
        K = int(self.sampling_steps)
        if K >= T:
            return list(range(T, -1, -1))

        if self.sampling_schedule == "uniform":
            # evenly spaced over [0, T]
            grid = np.linspace(0, T, K + 1)
        elif self.sampling_schedule == "quadratic":
            # denser near 0 (late denoising stage)
            u = np.linspace(0.0, 1.0, K + 1)
            grid = (u ** 2) * T
        elif self.sampling_schedule == "cosine":
            # another common non-uniform spacing
            u = np.linspace(0.0, 1.0, K + 1)
            grid = (1.0 - np.cos(0.5 * np.pi * u)) * T
        else:
            raise ValueError(f"Unknown sampling_schedule: {self.sampling_schedule}")

        ts = np.round(grid).astype(int).tolist()
        ts = sorted(set(ts), reverse=True)
        if ts[0] != T:
            ts = [T] + ts
        if ts[-1] != 0:
            ts = ts + [0]
        return ts

    def _get_Qt_between(self, alpha_s_bar, alpha_t_bar):
        """
        Build transition q(z_t | z_s) for s < t (jump step).
        For this discrete transition family, q(z_t | z_s) has same form as Qt_bar
        with alpha_{t|s} = alpha_bar_t / alpha_bar_s.
        """
        alpha_t_given_s = alpha_t_bar / torch.clamp(alpha_s_bar, min=1e-12)
        alpha_t_given_s = torch.clamp(alpha_t_given_s, min=0.0, max=1.0)
        return self.transition_model.get_Qt_bar(alpha_t_given_s, self.device)



    def _apply_sampling_corrector(self, edge_logits, t_norm, step_idx):
        """
        edge_logits: [B, N, N, De]
        t_norm:      [B, 1] in [0,1]
        step_idx:    current outer-loop index
        """
        # fast path: disabled => no extra math
        if not self.use_sampling_corrector:
            return edge_logits

        # interval gate
        if self.corrector_every_n <= 0 or (step_idx % self.corrector_every_n != 0):
            return edge_logits

        # apply only in later denoising stage
        # t_norm is normalized t (1.0 -> noisy, 0.0 -> clean)
        t_scalar = float(t_norm.mean().item())
        if t_scalar > self.corrector_apply_until_t:
            return edge_logits

        out = edge_logits

        # 1) temperature scaling
        temp = max(self.corrector_temperature, 1e-6)
        out = out / temp

        # 2) edge prior re-scoring (small strength recommended)
        if self.corrector_edge_prior_strength > 0:
            prior_log = self.corrector_edge_prior_log.to(out.device, out.dtype)
            out = out + self.corrector_edge_prior_strength * prior_log.view(1, 1, 1, -1)

        # 3) clip for numerical stability
        out = torch.clamp(out, min=-self.corrector_logit_clip, max=self.corrector_logit_clip)
        return out



    def _edge_prior_score_from_E(self, E_onehot, node_mask):
        """
        E_onehot: [B, N, N, De] one-hot
        node_mask: [B, N]
        return: [B] prior score (higher is better)
        """
        if not self.use_multitraj_rerank:
            return torch.zeros(E_onehot.size(0), device=E_onehot.device, dtype=E_onehot.dtype)

        cls = E_onehot.argmax(dim=-1)  # [B,N,N]
        B, N = cls.shape[0], cls.shape[1]

        nm = node_mask.bool()
        valid = nm.unsqueeze(1) & nm.unsqueeze(2)  # [B,N,N]
        diag = torch.eye(N, device=E_onehot.device, dtype=torch.bool).unsqueeze(0)
        valid = valid & (~diag)

        logp = self.rerank_edge_prior_log.to(E_onehot.device)[cls]  # [B,N,N]
        num = (logp * valid.float()).sum(dim=(1, 2))
        den = torch.clamp(valid.float().sum(dim=(1, 2)), min=1.0)
        return num / den

    def _build_timesteps_from_steps_and_power(self, steps: int, power: float):
        T = int(self.T)
        steps = int(max(2, min(steps, T)))
        u = np.linspace(0.0, 1.0, steps + 1)
        grid = (u ** max(power, 1e-4)) * T
        ts = np.round(grid).astype(int).tolist()
        ts = sorted(set(ts), reverse=True)
        if ts[0] != T:
            ts = [T] + ts
        if ts[-1] != 0:
            ts = ts + [0]
        return ts

    def _extract_sample_difficulty_feat(self, batch, b_idx: int, device):
        # 1) num_peaks
        if "num_peaks" in batch:
            num_peaks = batch["num_peaks"][b_idx].float()
            max_peaks = torch.clamp(batch["num_peaks"].float().max(), min=1.0)
            num_peaks_norm = (num_peaks / max_peaks).clamp(0.0, 1.0)
        else:
            num_peaks_norm = torch.tensor(0.5, device=device)

        # 2) intensity entropy
        if ("intens" in batch) and ("num_peaks" in batch):
            intens = batch["intens"][b_idx]  # [L]
            L = int(batch["num_peaks"][b_idx].item())
            if L > 0:
                p = intens[:L].float().clamp(min=0)
                p = p / (p.sum() + self.ctt_entropy_eps)
                ent = -(p * torch.log(p + self.ctt_entropy_eps)).sum()
                ent_norm = (ent / np.log(max(L, 2))).clamp(0.0, 1.0)
            else:
                ent_norm = torch.tensor(0.0, device=device)
        else:
            ent_norm = torch.tensor(0.5, device=device)

        # 3) precursor_mz (可选，不存在则给0)
        if "precursor_mz" in batch:
            pmz = batch["precursor_mz"][b_idx].float()
            precursor_norm = (pmz / 1000.0).clamp(0.0, 1.0)
        else:
            precursor_norm = torch.tensor(0.0, device=device)

        feat = torch.stack([num_peaks_norm, ent_norm, precursor_norm], dim=0)  # [3]
        return feat

    def _get_conditional_timesteps_for_sample(self, batch, b_idx: int):
        device = self.device
        feat = self._extract_sample_difficulty_feat(batch, b_idx, device=device)  # [3]

        if self.conditional_tuner_mode == "mlp":
            # out[0]: steps delta ([-1,1] after tanh)
            # out[1]: power delta
            out = self.ctt_mlp(feat.unsqueeze(0)).squeeze(0)
            d_steps = torch.tanh(out[0])
            d_power = torch.tanh(out[1])

            steps_span = max(self.ctt_max_steps - self.ctt_min_steps, 1)
            steps = self.ctt_min_steps + int(((d_steps + 1) * 0.5 * steps_span).round().item())
            power = float(self.ctt_default_power + 0.8 * d_power.item())
        else:
            # rule-based: difficulty越高 -> 步数越多
            difficulty = (0.45 * feat[0] + 0.45 * feat[1] + 0.10 * feat[2]).clamp(0.0, 1.0)
            steps = self.ctt_min_steps + int((difficulty * (self.ctt_max_steps - self.ctt_min_steps)).round().item())
            # 难样本后期更密一些
            power = float(1.4 - 0.6 * difficulty.item())  # [~0.8, ~1.4]

        return self._build_timesteps_from_steps_and_power(steps, power)



    def _test_pred_path(self, i: int) -> str:
        return f"preds/{self.name}_rank_{self.global_rank}_pred_{i}.pkl"

    def _test_true_path(self, i: int) -> str:
        return f"preds/{self.name}_rank_{self.global_rank}_true_{i}.pkl"

    def _atomic_pickle_dump(self, obj, final_path: str):
        tmp_path = final_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, final_path)

    def _try_load_cached_test_batch(self, i: int):
        pred_path = self._test_pred_path(i)
        true_path = self._test_true_path(i)

        pred_exists = os.path.exists(pred_path)
        true_exists = os.path.exists(true_path)

        if self.test_resume_strict_pair:
            if not (pred_exists and true_exists):
                return None
        else:
            if not pred_exists:
                return None

        try:
            with open(pred_path, "rb") as f:
                predicted_mols = pickle.load(f)
            with open(true_path, "rb") as f:
                true_mols = pickle.load(f)
            return predicted_mols, true_mols
        except Exception as e:
            self._test_resume_broken += 1
            logging.warning(f"[TEST-RESUME] broken cache at batch={i}: {e}")
            if self.test_resume_overwrite_broken:
                return None
            raise

