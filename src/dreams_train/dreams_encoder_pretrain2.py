# encoder_pretrain_dreams.py
import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from rdkit import RDLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MeanMetric

from src.mist.data import datasets, featurizers, splitter
from src.mist.models.dreams_encoder import DreaMSEncoder


def parse_args():
    p = argparse.ArgumentParser()
    # 数据路径
    p.add_argument("--labels_file", type=str, default="data/canopus/labels.tsv")
    p.add_argument("--spec_folder", type=str, default="data/canopus/spec_files")
    p.add_argument("--split_file", type=str, default="data/canopus/splits/canopus_hplus_100_0.tsv")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--max_count", type=int, default=None)

    # DreaMS encoder 超参
    p.add_argument("--output_size", type=int, default=4096)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_freq", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max_peaks", type=int, default=100)

    # 训练超参
    p.add_argument("--devices", type=int, default=2)
    p.add_argument("--strategy", type=str, default="ddp")
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=1e-12)
    p.add_argument("--fp_name", type=str, default="morgan4096")

    # 保存
    p.add_argument("--save_path", type=str, default="checkpoints/dreams_encoder_canopus.pt")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/dreams_encoder_ckpts")

    # 早停超参
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_monitor", type=str, default="val_loss")

    p.add_argument("--dreams_cache_dir", type=str, default="data/canopus/dreams_cache")
    return p.parse_args()


class EncoderPretrainModule(pl.LightningModule):
    def __init__(self, encoder, lr, weight_decay):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        self.weight_decay = weight_decay

        # 全局加权均值（跨 step + 跨 rank）
        self.train_loss_mean = MeanMetric()
        self.val_loss_mean = MeanMetric()

    def _bce_with_pos_weight(self, pred, tgt):
        pos_frac = tgt.mean(dim=0).clamp(min=1e-4, max=1 - 1e-4)
        pos_weight = (1.0 - pos_frac) / pos_frac
        return F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=pos_weight)

    def _compute_loss(self, batch):
        spec_keys = {k: v for k, v in batch.items()
                     if k not in ["mols", "spec_indices", "mol_indices", "matched"]}

        pred_fp, _ = self.encoder(spec_keys)

        spec_idx = batch["spec_indices"]
        mol_idx = batch["mol_indices"]

        tgt_fp = batch["mols"][mol_idx].float()
        tgt_fp = (tgt_fp > 0).float()

        pred = pred_fp[spec_idx]

        # DreaMS encoder output already Sigmoid -> convert to logits for BCE
        pred_clamped = pred.clamp(min=1e-6, max=1 - 1e-6)
        pred_logits = torch.log(pred_clamped / (1 - pred_clamped))
        loss = self._bce_with_pos_weight(pred_logits, tgt_fp)
        return loss

    def _loss_weight(self, batch):
        # 使用有效 pair 数做权重，避免不同 batch 大小时均值偏差
        return batch["spec_indices"].numel()

    def on_train_epoch_start(self):
        self.train_loss_mean.reset()

    def on_validation_epoch_start(self):
        self.val_loss_mean.reset()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        w = self._loss_weight(batch)
        self.train_loss_mean.update(loss.detach(), weight=w)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        w = self._loss_weight(batch)
        self.val_loss_mean.update(loss.detach(), weight=w)
        return loss

    def on_train_epoch_end(self):
        self.log("train_loss", self.train_loss_mean.compute(), prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        # EarlyStopping/Checkpoint 监控这个全局同步后的 val_loss
        self.log("val_loss", self.val_loss_mean.compute(), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def build_dataloaders(args):
    spectra_list, mol_list = datasets.get_paired_spectra(
        labels_file=args.labels_file,
        spec_folder=args.spec_folder,
        max_count=args.max_count,
        prog_bars=True,
    )

    paired_featurizer = featurizers.get_paired_featurizer(
        spec_features="dreams_cache",
        mol_features="fingerprint",
        fp_names=[args.fp_name],
        cache_featurizers=True,
        cache_dir=args.dreams_cache_dir,
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

    encoder = DreaMSEncoder(
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_freq=args.num_freq,
        dropout=args.dropout,
        max_peaks=args.max_peaks,
        use_int_preds=False,
    )

    train_loader, val_loader = build_dataloaders(args)
    module = EncoderPretrainModule(encoder, lr=args.lr, weight_decay=args.weight_decay)

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

    use_gpu = torch.cuda.is_available()
    use_ddp = use_gpu and args.devices > 1

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=args.devices if use_gpu else 1,
        strategy=args.strategy if use_ddp else "auto",
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        gradient_clip_val=None,
        enable_checkpointing=True,
        callbacks=[early_stop_cb, ckpt_cb],
    )

    trainer.fit(module, train_loader, val_loader)

    # 只在主进程保存，避免多卡重复写文件
    if trainer.is_global_zero:
        best_ckpt_path = ckpt_cb.best_model_path
        if best_ckpt_path:
            best_module = EncoderPretrainModule.load_from_checkpoint(
                best_ckpt_path,
                encoder=encoder,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            torch.save(best_module.encoder.state_dict(), args.save_path)
            print(f"Saved BEST encoder state_dict to {args.save_path}")
        else:
            torch.save(encoder.state_dict(), args.save_path)
            print(f"No best checkpoint found, saved current encoder to {args.save_path}")


if __name__ == "__main__":
    main()