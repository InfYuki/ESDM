
"""
Visualize DiffMS test results: generate Figure 4-style molecular comparison plots.

Usage:
    # Step 1: Run test first (generates pickle files in preds/)
    python src/spec2mol_main.py general.test_only=/path/to/checkpoint.ckpt

    # Step 2: Generate visualizations
    python visualize_test_results.py --preds_dir preds/ --output_dir output/ --num_cases 10

    # Or run end-to-end (test + visualization):
    python visualize_test_results.py --test_only /path/to/checkpoint.ckpt --output_dir output/
"""

import os
import sys
import pickle
import argparse
import logging
import subprocess
from collections import Counter
from pathlib import Path
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def mol_to_smiles(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def is_valid(mol):
    smiles = mol_to_smiles(mol)
    if smiles is None:
        return False
    try:
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        return len(frags) == 1
    except Exception:
        return False


def compute_tanimoto(mol1, mol2, radius=2, nbits=2048):
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=nbits)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def rank_predictions(generated_mols):
    """Rank generated molecules by frequency (unique InChI sorted by count)."""
    valid_inchis = []
    inchi_to_mol = {}

    for mol in generated_mols:
        if not is_valid(mol):
            continue
        inchi = Chem.MolToInchi(mol)
        if inchi:
            valid_inchis.append(inchi)
            if inchi not in inchi_to_mol:
                inchi_to_mol[inchi] = mol

    counter = Counter(valid_inchis)
    ranked = []
    for inchi, count in counter.most_common():
        ranked.append((inchi_to_mol[inchi], inchi, count))
    return ranked


def mol_to_image(mol, size=(300, 300)):
    if mol is None:
        return None
    try:
        drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Pickle loading
# ──────────────────────────────────────────────────────────────────────────────

def load_preds_from_dir(preds_dir, model_name=None):
    """Load all prediction and true molecule pickle files from preds/ directory."""
    preds_dir = Path(preds_dir)
    pred_files = sorted(preds_dir.glob('*_pred_*.pkl'))
    true_files = sorted(preds_dir.glob('*_true_*.pkl'))

    if not pred_files:
        raise FileNotFoundError(f"No prediction pickle files found in {preds_dir}")

    if model_name:
        pred_files = [f for f in pred_files if model_name in f.name]
        true_files = [f for f in true_files if model_name in f.name]

    def sort_key(f):
        parts = f.stem.split('_')
        rank_idx = parts.index('rank')
        rank = int(parts[rank_idx + 1])
        batch_idx = int(parts[-1])
        return (rank, batch_idx)

    pred_files = sorted(pred_files, key=sort_key)
    true_files = sorted(true_files, key=sort_key)

    all_true = []
    all_pred = []

    for tf in true_files:
        with open(tf, 'rb') as f:
            all_true.extend(pickle.load(f))

    for pf in pred_files:
        with open(pf, 'rb') as f:
            all_pred.extend(pickle.load(f))

    logger.info(f"Loaded {len(all_true)} test cases from {preds_dir}")
    return all_true, all_pred


# ──────────────────────────────────────────────────────────────────────────────
# Test case selection
# ──────────────────────────────────────────────────────────────────────────────

def select_test_cases(true_mols, pred_mols, num_cases=10, top_k=5):
    """
    Select test cases for visualization.
    Priority:
    1. Cases where ground truth appears in top-k generated (exact match)
    2. Cases with highest max Tanimoto similarity (close match)
    """
    """
    Select test cases for visualization.
    Only includes cases with at least top_k unique valid generated structures.
    ...
    """
    logger.info(f"Analyzing {len(true_mols)} test cases, selecting {num_cases}...")

    case_info = []

    for idx in range(len(true_mols)):
        true_mol = true_mols[idx]
        true_smiles = mol_to_smiles(true_mol)
        if true_smiles is None:
            continue
        true_inchi = Chem.MolToInchi(true_mol)
        if not true_inchi:
            continue

        ranked = rank_predictions(pred_mols[idx])
        top_mols = []
        is_exact = False
        max_tan = 0.0

        for rank_idx, (gen_mol, gen_inchi, count) in enumerate(ranked[:top_k]):
            tan = compute_tanimoto(true_mol, gen_mol)
            exact = (gen_inchi == true_inchi)
            if exact:
                is_exact = True
            gen_smiles = mol_to_smiles(gen_mol)
            top_mols.append({
                'mol': gen_mol,
                'smiles': gen_smiles,
                'tanimoto': tan,
                'count': count,
                'is_exact': exact,
                'rank': rank_idx + 1,
            })
            max_tan = max(max_tan, tan)

        #if not top_mols:
        #    continue

        if len(top_mols) < top_k:
            continue

        case_info.append({
            'idx': idx,
            'true_mol': true_mol,
            'true_smiles': true_smiles,
            'true_inchi': true_inchi,
            'top_mols': top_mols,
            'max_tanimoto': max_tan,
            'is_exact_match': is_exact,
        })

    exact_cases = [c for c in case_info if c['is_exact_match']]
    non_exact_cases = [c for c in case_info if not c['is_exact_match']]
    exact_cases.sort(key=lambda x: x['max_tanimoto'], reverse=True)
    non_exact_cases.sort(key=lambda x: x['max_tanimoto'], reverse=True)

    selected = (exact_cases + non_exact_cases)[:num_cases]
    n_exact = sum(1 for c in selected if c['is_exact_match'])
    #logger.info(f"Selected {len(selected)} cases: {n_exact} exact matches, "
    #            f"{len(selected) - n_exact} close matches")

    logger.info(
        f"Eligible cases (>={top_k} unique valid preds): {len(case_info)}, "
        f"selected {len(selected)} ({n_exact} exact, {len(selected) - n_exact} close)"
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation
# ──────────────────────────────────────────────────────────────────────────────

def get_border_color(tanimoto, is_exact):
    if is_exact:
        return '#2ecc71'
    elif tanimoto >= 0.675:
        return '#e67e22'
    else:
        return '#95a5a6'


def draw_molecule_to_ax(ax, mol, smiles, tanimoto=None, is_exact=False,
                        is_ground_truth=False, count=None, title=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # 自下而上：SMILES | T 标签 | T 条 | 分子图
    Y_SMILES = 0.03
    Y_T_LABEL = 0.10
    Y_BAR = 0.16
    BAR_H = 0.035
    Y_MOL_BOTTOM = 0.24   # imshow 下缘，须 > Y_BAR + BAR_H/2
    Y_MOL_TOP = 0.95

    img_data = mol_to_image(mol, size=(400, 400))
    if img_data is None:
        ax.text(0.5, 0.5, 'Invalid', ha='center', va='center',
                fontsize=10, color='red', transform=ax.transAxes)
        return

    img = Image.open(BytesIO(img_data))
    #ax.imshow(img, extent=[0.05, 0.95, 0.15, 0.95], aspect='auto')
    ax.imshow(img, extent=[0.05, 0.95, Y_MOL_BOTTOM, Y_MOL_TOP], aspect='auto')

    if title:
        ax.set_title(title, fontsize=8, fontweight='bold', pad=2)

    if tanimoto is not None:
        color = get_border_color(tanimoto, is_exact)
        ax.barh(Y_BAR, tanimoto, height=BAR_H, left=0.05,
                color=color, alpha=0.8, transform=ax.transAxes)
        label = f'T={tanimoto:.3f}'
        if is_exact:
            label += ' *'
        ax.text(0.5, Y_T_LABEL, label, ha='center', va='center',
                fontsize=7, fontweight='bold', color=color, transform=ax.transAxes)

    if smiles:
        max_len = 35
        display_smi = smiles if len(smiles) <= max_len else smiles[:max_len] + '...'
        ax.text(0.5, Y_SMILES, display_smi, ha='center', va='center',
                fontsize=5, color='#333333', transform=ax.transAxes,
                family='monospace')

    if count is not None:
        ax.text(0.95, 0.95, f'n={count}', ha='right', va='top',
                fontsize=7, color='#666666', transform=ax.transAxes)


def generate_figure(selected_cases, output_path, top_k=5):
    num_cases = len(selected_cases)
    num_cols = 1 + top_k

    fig_width = num_cols * 2.5
    fig_height = num_cases * 2.8

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(num_cases, num_cols, figure=fig,
                           hspace=0.3, wspace=0.15,
                           left=0.02, right=0.98, top=0.96, bottom=0.02)

    for row_idx, case in enumerate(selected_cases):
        ax_gt = fig.add_subplot(gs[row_idx, 0])
        draw_molecule_to_ax(
            ax_gt, case['true_mol'], case['true_smiles'],
            is_ground_truth=True,
            #title=f"Case {case['idx']} (GT)" if row_idx == 0 else "Ground Truth"
            title=None,
        )
        for spine in ax_gt.spines.values():
            spine.set_visible(True)
            spine.set_color('#2ecc71')
            spine.set_linewidth(2.5)

        for col_idx in range(top_k):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            if col_idx < len(case['top_mols']):
                gen = case['top_mols'][col_idx]
                draw_molecule_to_ax(
                    ax, gen['mol'], gen['smiles'],
                    tanimoto=gen['tanimoto'],
                    is_exact=gen['is_exact'],
                    count=gen['count'],
                    title=f"Top-{gen['rank']}" if row_idx == 0 else None
                )
                color = get_border_color(gen['tanimoto'], gen['is_exact'])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(color)
                    spine.set_linewidth(1.5)
            else:
                ax.axis('off')

    legend_elements = [
        plt.Line2D([0], [0], color='#2ecc71', linewidth=3, label='Exact match'),
        plt.Line2D([0], [0], color='#e67e22', linewidth=3, label='Close match (T>=0.675)'),
        plt.Line2D([0], [0], color='#95a5a6', linewidth=3, label='Other'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=9, frameon=True, fancybox=True, shadow=True)

    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Figure saved to {output_path}")


def generate_case_figure(case, output_path, top_k=5, dpi=200, show_legend=True):
    """Save one row: GT + Top-1..Top-k as a single figure."""
    num_cols = 1 + top_k
    fig_width = num_cols * 1.6   # 论文用可再调小，如 1.4
    fig_height = 2.2             # 单行高度  1.8

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(1, num_cols, figure=fig,
                           hspace=0.15, wspace=0.12,
                           left=0.02, right=0.98, top=0.92, bottom=0.18)

    ax_gt = fig.add_subplot(gs[0, 0])
    draw_molecule_to_ax(
        ax_gt, case['true_mol'], case['true_smiles'],
        is_ground_truth=True, title=None,
    )
    for spine in ax_gt.spines.values():
        spine.set_visible(True)
        spine.set_color('#2ecc71')
        spine.set_linewidth(2.5)

    for col_idx in range(top_k):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        gen = case['top_mols'][col_idx]
        draw_molecule_to_ax(
            ax, gen['mol'], gen['smiles'],
            tanimoto=gen['tanimoto'],
            is_exact=gen['is_exact'],
            count=gen['count'],
            title=f"Top-{gen['rank']}",  # 不要列名就改 None
        )
        color = get_border_color(gen['tanimoto'], gen['is_exact'])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(1.5)

    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], color='#2ecc71', linewidth=3, label='Exact match'),
            plt.Line2D([0], [0], color='#e67e22', linewidth=3, label='Close match (T>=0.675)'),
            plt.Line2D([0], [0], color='#95a5a6', linewidth=3, label='Other'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3,
                   fontsize=7, frameon=True)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Case figure saved to {output_path}")

def generate_case_figures(selected_cases, output_dir, top_k=5, dpi=200):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for plot_i, case in enumerate(selected_cases):
        # 文件名用数据集 idx，便于和 CSV 对应
        out_name = f"case_{case['idx']:04d}.png"
        out_path = os.path.join(output_dir, out_name)
        generate_case_figure(case, out_path, top_k=top_k, dpi=dpi)
        paths.append(out_path)
    return paths

# ──────────────────────────────────────────────────────────────────────────────
# CSV generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_csv(selected_cases, output_path, top_k=5):
    rows = []
    for case in selected_cases:
        base_row = {
            'case_idx': case['idx'],
            'true_smiles': case['true_smiles'],
            'max_tanimoto': case['max_tanimoto'],
            'is_exact_match': case['is_exact_match'],
        }
        for gen in case['top_mols']:
            row = dict(base_row)
            row['rank'] = gen['rank']
            row['gen_smiles'] = gen['smiles']
            row['gen_tanimoto'] = gen['tanimoto']
            row['gen_count'] = gen['count']
            row['gen_is_exact'] = gen['is_exact']
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV saved to {output_path}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_test(checkpoint_path, config_overrides=None):
    """Run the test step using spec2mol_main.py."""
    logger.info(f"Running test with checkpoint: {checkpoint_path}")
    cmd = [sys.executable, 'src/spec2mol_main.py',
           f'general.test_only={checkpoint_path}',
           'general.wandb=disabled']
    if config_overrides:
        cmd.extend(config_overrides)

    logger.info(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        logger.error(f"Test failed:\n{result.stderr}")
        raise RuntimeError(f"Test step failed with return code {result.returncode}")

    logger.info("Test completed successfully")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DiffMS test results (Figure 4 style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--test_only', type=str, default=None,
                        help='Path to checkpoint. If provided, runs test first.')
    parser.add_argument('--preds_dir', type=str, default='preds',
                        help='Directory with prediction pickle files (default: preds/)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory (default: output/)')
    parser.add_argument('--num_cases', type=int, default=10,
                        help='Number of test cases to visualize (default: 10)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top-k generated molecules per case (default: 5)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Filter pickle files by model name')
    parser.add_argument('--config', nargs='*', default=None,
                        help='Additional Hydra config overrides for test step')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Run test if checkpoint provided
    if args.test_only:
        if not os.path.exists(args.test_only):
            logger.error(f"Checkpoint not found: {args.test_only}")
            sys.exit(1)
        run_test(args.test_only, args.config)

    # Step 2: Load predictions
    if not os.path.exists(args.preds_dir):
        logger.error(f"Predictions directory not found: {args.preds_dir}")
        sys.exit(1)

    true_mols, pred_mols = load_preds_from_dir(args.preds_dir, args.model_name)
    if len(true_mols) == 0:
        logger.error("No test cases found")
        sys.exit(1)

    # Step 3: Select test cases
    selected = select_test_cases(true_mols, pred_mols,
                                 num_cases=args.num_cases,
                                 top_k=args.top_k)
    if not selected:
        logger.error("No valid test cases found")
        sys.exit(1)

    # Step 4: Generate outputs
    #fig_path = os.path.join(args.output_dir, 'figure4_top5.png')
    #csv_path = os.path.join(args.output_dir, 'analysis.csv')

    cases_dir = os.path.join(args.output_dir, 'cases')
    fig_paths = generate_case_figures(selected, cases_dir, top_k=args.top_k, dpi=200)
    csv_path = os.path.join(args.output_dir, 'analysis.csv')
    df = generate_csv(selected, csv_path, top_k=args.top_k)
    #generate_figure(selected, fig_path, top_k=args.top_k)
    #df = generate_csv(selected, csv_path, top_k=args.top_k)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total test cases loaded: {len(true_mols)}")
    print(f"Cases visualized: {len(selected)}")
    n_exact = sum(1 for c in selected if c['is_exact_match'])
    print(f"  - Exact matches: {n_exact}")
    print(f"  - Close matches (T>=0.675): "
          f"{sum(1 for c in selected if not c['is_exact_match'] and c['max_tanimoto'] >= 0.675)}")
    print(f"\nOutputs:")
    print(f"  Figure: {fig_paths}")
    print(f"  CSV:    {csv_path}")

    print(f"\n{'Case':>6} {'Exact':>6} {'MaxTan':>8} {'Top-1 SMILES':<40}")
    print("-" * 65)
    for case in selected:
        top1_smi = case['top_mols'][0]['smiles'] if case['top_mols'] else 'N/A'
        if len(top1_smi) > 38:
            top1_smi = top1_smi[:35] + '...'
        print(f"{case['idx']:>6} {'Yes' if case['is_exact_match'] else 'No':>6} "
              f"{case['max_tanimoto']:>8.4f} {top1_smi:<40}")


if __name__ == '__main__':
    main()