# dreams_dump_feats.py
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from src.mist.data import datasets, featurizers

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_file", type=str, default="data/canopus/labels.tsv")
    p.add_argument("--spec_folder", type=str, default="data/canopus/spec_files")
    p.add_argument("--max_count", type=int, default=None)
    p.add_argument("--max_peaks", type=int, default=100)
    p.add_argument("--out_dir", type=str, default="data/canopus/dreams_cache")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    spectra_list, mol_list = datasets.get_paired_spectra(
        labels_file=args.labels_file,
        spec_folder=args.spec_folder,
        max_count=args.max_count,
        prog_bars=True,
    )

    # 直接用 DreaMSPeakFeaturizer
    spec_featurizer = featurizers.DreaMSPeakFeaturizer(max_peaks=args.max_peaks)

    for i, spec in enumerate(tqdm(spectra_list)):
        feat = spec_featurizer.featurize(spec)
        # feat = {"mz": np.array, "intens": np.array, "precursor_mz": float, "name": str}
        out_path = os.path.join(args.out_dir, f"{feat['name']}.pt")
        torch.save(
            {
                "mz": torch.tensor(feat["mz"], dtype=torch.float32),
                "intens": torch.tensor(feat["intens"], dtype=torch.float32),
                "precursor_mz": torch.tensor(feat["precursor_mz"], dtype=torch.float32),
            },
            out_path
        )

if __name__ == "__main__":
    main()