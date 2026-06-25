
'''
import pandas as pd
from pathlib import Path

src = Path(r"../testdata/canopus_hplus_100_0.tsv")
dst = Path(r"../testdata/canopus_hplus_100_0_1.tsv")  # 建议先另存以防覆盖

# 读入 TSV
df = pd.read_csv(src, sep="\t")

# 若列名不是 split，请改成实际列名
col = "split"
mask_test = df[col] == "test"

df_test = df[mask_test]
df_other = df[~mask_test]

# 如果 test 样本不足 32，则全部保留；否则随机保留 32 个
keep_n = min(96, len(df_test))
df_test_kept = df_test.sample(n=keep_n, random_state=42)

df_new = pd.concat([df_other, df_test_kept], ignore_index=True)

# 保存为 TSV（不包含索引）
df_new.to_csv(dst, sep="\t", index=False)

print(f"原始行数: {len(df)}, 剔除后行数: {len(df_new)}, 保留的 test: {keep_n}")
'''




import torch
from rdkit import RDLogger
from torch.utils.data import DataLoader

from src.mist.data import datasets, featurizers, splitter
from src.mist.models.spectra_encoder import SpectraEncoderGrowing

def build_canopus_loader(
    labels_file="../data/canopus/labels.tsv",
    spec_folder="../data/canopus/spec_files",
    split_file="../data/canopus/splits/canopus_hplus_100_0.tsv",
    subform_folder="../data/canopus/subformulae/subformulae_default",
    batch_size=2,
    num_workers=0,
):
    spectra_list, mol_list = datasets.get_paired_spectra(
        labels_file=labels_file,
        spec_folder=spec_folder,
        max_count=None,
        prog_bars=True,
    )

    paired_featurizer = featurizers.get_paired_featurizer(
        spec_features="peakformula",
        mol_features="fingerprint",
        fp_names=["morgan4096"],
        subform_folder=subform_folder,
        remove_prob=0.1,
        remove_weights="exp",
        inten_prob=0.1,
        inten_transform="float",
        cls_type="ms1",
        set_pooling="cls",
        cache_featurizers=True,
    )

    pairs = list(zip(spectra_list, mol_list))
    preset_splitter = splitter.PresetSpectraSplitter(split_file=split_file)
    _, (train_pairs, _, _) = preset_splitter.get_splits(pairs)

    ds = datasets.SpectraMolDataset(train_pairs, featurizer=paired_featurizer)
    collate_pairs = datasets._collate_pairs
    mol_collate = ds.get_featurizer().get_mol_collate()
    spec_collate = ds.get_featurizer().get_spec_collate()

    def collate_fn(batch):
        return collate_pairs(batch, mol_collate_fn=mol_collate, spec_collate_fn=spec_collate)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

def print_tensor(name, x):
    if torch.is_tensor(x):
        x_f = x.float() if x.numel() > 0 else x
        print(f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
              f"mean={x_f.mean():.4f} std={x_f.std():.4f}")
        # 打印少量值，避免刷屏
        flat = x.flatten()
        print(f"  head: {flat[:8].tolist()}")
    else:
        print(f"{name}: type={type(x)}")

def main():
    RDLogger.DisableLog("rdApp.*")

    # 1) 构建与 spec2mol_main 对齐的 encoder（参数与其保持一致）
    encoder = SpectraEncoderGrowing(
        form_embedder="pos-cos",
        output_size=4096,
        hidden_size=256,
        spectra_dropout=0.1,
        top_layers=1,
        refine_layers=4,
        magma_modulo=512,
        peak_attn_layers=2,
        num_heads=8,
        pairwise_featurization=True,
        embed_instrument=False,
        cls_type="ms1",
        set_pooling="cls",
        spec_features="peakformula",
        mol_features="fingerprint",
        inten_prob=0.1,
        remove_prob=0.5,
        use_diff_attn=True,
    )

    # 2) 取一条 canopus batch
    loader = build_canopus_loader(batch_size=2, num_workers=0)
    batch = next(iter(loader))

    # 3) encoder 只用谱图相关 key，过滤掉 mol 索引等
    drop_keys = {"mols", "spec_indices", "mol_indices", "matched"}
    spec_keys = {k: v for k, v in batch.items() if k not in drop_keys}

    # 4) 打印输入
    print("=== Encoder 输入 ===")
    for k in sorted(spec_keys.keys()):
        print_tensor(k, spec_keys[k])

    # 5) 前向并打印输出
    encoder.eval()
    with torch.no_grad():
        out, aux = encoder(spec_keys)

    print("\n=== Encoder 输出 ===")
    print_tensor("out", out)
    print("\n=== Encoder aux ===")
    for k in aux:
        print_tensor(f"aux[{k}]", aux[k])

if __name__ == "__main__":
    main()












