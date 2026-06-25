# encoder_pretrain.py
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from mist.data import datasets, featurizers, splitter
from mist.models.spectra_encoder import SpectraEncoderGrowing
from rdkit import RDLogger

torch.cuda.empty_cache()
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser()
    # 数据路径与拆分（与 configs/dataset/canopus.yaml 对齐）
    p.add_argument("--labels_file", type=str, default="data/canopus/labels.tsv")
    p.add_argument("--spec_folder", type=str, default="data/canopus/spec_files")
    p.add_argument("--split_file", type=str, default="data/canopus/splits/canopus_hplus_100_0.tsv")
    p.add_argument("--subform_folder", type=str, default="data/canopus/subformulae/subformulae_default")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--max_count", type=int, default=None)

    p.add_argument("--devices", type=int, default=2)
    p.add_argument("--strategy", type=str, default="ddp")

    # 模型/训练超参（与主模型 encoder 超参一致）
    p.add_argument("--output_size", type=int, default=4096)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--magma_modulo", type=int, default=512)
    p.add_argument("--peak_attn_layers", type=int, default=2)
    p.add_argument("--refine_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=1e-12)
    p.add_argument("--fp_name", type=str, default="morgan4096")
    p.add_argument("--save_path", type=str, default="checkpoints/encoder_canopus_new.pt")

    # 1) 在 argparse 增加可调权重（可选）
    p.add_argument("--lambda_inter", type=float, default=0.5)  # 中间层损失权重
    p.add_argument("--lambda_frag", type=float, default=0.0)  # 若无fragment标签，先置0

    # 1) 增加参数  用于测试预训练后的encoder的效果
    p.add_argument("--load_path", type=str, default=None)

    # 早停
    p.add_argument("--use_diff_attn", action="store_true")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/encoder_ckpts")
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_monitor", type=str, default="val_loss")


    return p.parse_args()


# 2) LightningModule 内：改用 RAdam，并实现多目标损失
class EncoderPretrainModule(pl.LightningModule):
    def __init__(self, encoder, lr, weight_decay, lambda_inter=0.5, lambda_frag=0.0):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_inter = lambda_inter
        self.lambda_frag = lambda_frag

    def _bce_with_pos_weight(self, pred, tgt):
        pos_frac = tgt.mean(dim=0).clamp(min=1e-4, max=1 - 1e-4)
        pos_weight = (1.0 - pos_frac) / pos_frac
        return F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pos_weight)

    def _to_logits(self, p, eps=1e-6):
        p = p.clamp(min=eps, max=1.0 - eps)
        return torch.log(p) - torch.log(1.0 - p)

    def _compute_loss(self, batch):
        spec_keys = {k: v for k, v in batch.items()
                     if k not in ["mols", "spec_indices", "mol_indices", "matched"]}
        pred_fp, aux = self.encoder(spec_keys)
        spec_idx = batch["spec_indices"]
        mol_idx = batch["mol_indices"]

        tgt_fp_full = batch["mols"][mol_idx].float()
        tgt_fp_full = (tgt_fp_full > 0).float()

        # 最终输出：4096 位
        pred_final = pred_fp[spec_idx]
        loss_final = self._bce_with_pos_weight(self._to_logits(pred_final), tgt_fp_full)

        # 中间输出：按各自维度截取目标指纹
        inter_preds = aux.get("int_preds", [])
        inter_losses = []
        for p in inter_preds:
            p_sel = p[spec_idx]
            d = p_sel.shape[1]
            tgt_slice = tgt_fp_full[:, :d]
            inter_losses.append(self._bce_with_pos_weight(self._to_logits(p_sel), tgt_slice))
        loss_inter = torch.stack(inter_losses).mean() if inter_losses else 0.0

        # 片段头（若无标签，可保持 0）
        loss_frag = 0.0

        loss = loss_final + self.lambda_inter * loss_inter + self.lambda_frag * loss_frag
        return loss




    '''
    def _compute_loss(self, batch):
        spec_keys = {k: v for k, v in batch.items()
                     if k not in ["mols", "spec_indices", "mol_indices", "matched"]}
        pred_fp, _ = self.encoder(spec_keys)
        spec_idx = batch["spec_indices"]
        mol_idx = batch["mol_indices"]

        tgt_fp = batch["mols"][mol_idx].float()
        tgt_fp = (tgt_fp > 0).float()

        pred = pred_fp[spec_idx]

        # 动态正例权重，避免稀疏标签下梯度过弱
        pos_frac = tgt_fp.mean(dim=0).clamp(min=1e-4, max=1 - 1e-4)
        pos_weight = (1.0 - pos_frac) / pos_frac

        loss = F.binary_cross_entropy_with_logits(pred, tgt_fp, pos_weight=pos_weight)
        return loss
    '''

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self):
        tl = self.trainer.callback_metrics.get("train_loss")
        if tl is not None:
            print(f"[Epoch {self.current_epoch}] train_loss = {tl.item():.6f}")

    def on_validation_epoch_end(self):
        vl = self.trainer.callback_metrics.get("val_loss")
        if vl is not None:
            print(f"[Epoch {self.current_epoch}] val_loss   = {vl.item():.6f}")

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def build_dataloaders(args):
    spectra_list, mol_list = datasets.get_paired_spectra(
        labels_file=args.labels_file,
        spec_folder=args.spec_folder,
        max_count=args.max_count,
        prog_bars=True,
    )

    paired_featurizer = featurizers.get_paired_featurizer(
        spec_features="peakformula",
        mol_features="fingerprint",
        fp_names=[args.fp_name],
        subform_folder=args.subform_folder,
        remove_prob=0.1,
        remove_weights="exp",
        inten_prob=0.1,
        inten_transform="float",
        cls_type="ms1",
        set_pooling="cls",
        cache_featurizers=True,
    )

    pairs = list(zip(spectra_list, mol_list))
    preset_splitter = splitter.PresetSpectraSplitter(split_file=args.split_file)
    _, (train_pairs, val_pairs, _) = preset_splitter.get_splits(pairs)

    def make_loader(pairs_subset, shuffle):
        ds = datasets.SpectraMolDataset(pairs_subset, featurizer=paired_featurizer)
        collate_pairs = datasets._collate_pairs
        mol_collate = ds.get_featurizer().get_mol_collate()
        spec_collate = ds.get_featurizer().get_spec_collate()

        def collate_fn(batch):
            return collate_pairs(batch, mol_collate_fn=mol_collate, spec_collate_fn=spec_collate)

        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    train_loader = make_loader(train_pairs, shuffle=True)
    val_loader = make_loader(val_pairs, shuffle=False)
    return train_loader, val_loader


def main():
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()

    encoder = SpectraEncoderGrowing(
        form_embedder="pos-cos",
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        spectra_dropout=0.1,
        top_layers=1,
        refine_layers=args.refine_layers,
        magma_modulo=args.magma_modulo,
        peak_attn_layers=args.peak_attn_layers,
        num_heads=8,
        pairwise_featurization=True,
        embed_instrument=False,
        cls_type="ms1",
        set_pooling="cls",
        spec_features="peakformula",
        mol_features="fingerprint",
        inten_prob=0.1,
        remove_prob=0.5,
        use_diff_attn=False,
    )

    # 2) 在 main 里构建 encoder 后加载  覆盖encoder权重为预训练后的权重
    if args.load_path:
        ckpt = torch.load(args.load_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)  # 兼容 lightning/纯 state_dict
        # 若是 lightning 且带 "encoder." 前缀，可过滤：
        filtered = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
        try:
            encoder.load_state_dict(state_dict, strict=False)
        except Exception:
            encoder.load_state_dict(filtered, strict=False)
        print(f"Loaded pretrained encoder weights from {args.load_path}")

    train_loader, val_loader = build_dataloaders(args)
    module = EncoderPretrainModule(encoder, lr=args.lr, weight_decay=args.weight_decay)

    use_gpu = torch.cuda.is_available()
    use_ddp = use_gpu and args.devices > 1

    early_stop_cb = EarlyStopping(
        monitor=args.early_stop_monitor,
        min_delta=args.early_stop_min_delta,
        patience=args.early_stop_patience,
        mode="min",
        verbose=True,
    )


    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="best",
        monitor=args.early_stop_monitor,
        mode="min",
        save_top_k=1,
        save_last=False,
    )


    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=args.devices if use_gpu else 1,
        #strategy=args.strategy if use_ddp else "auto",
        strategy= 'ddp_find_unused_parameters_true',
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        gradient_clip_val=None,
        enable_checkpointing=True,
        callbacks=[early_stop_cb, ckpt_cb],
    )

    if args.load_path and args.epochs == 0:
        trainer.validate(module, val_loader)
    else:
        trainer.fit(module, train_loader, val_loader)
        trainer.validate(module, val_loader)  # 可选

        if trainer.is_global_zero:
            torch.save(encoder.state_dict(), args.save_path)
            print(f"Saved encoder state_dict to {args.save_path}")


if __name__ == "__main__":
    main()