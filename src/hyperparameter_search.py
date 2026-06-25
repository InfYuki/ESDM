import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


############################################################
# Search space
############################################################

SPACE = [
    Real(0.6, 1.3, name="tau"),
    Real(0.0, 0.10, name="lam"),
    Real(0.10, 0.60, name="t_apply"),
    Real(5.0, 30.0, name="clip"),
]

RESULTS = []


############################################################
# Defaults
############################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_CKPT = PROJECT_ROOT / "checkpoints" / "canopus_fs100_org" / "last-v1.ckpt"
FALLBACK_CKPT = PROJECT_ROOT / "checkpoints" / "diffms_canopus.ckpt"
DEFAULT_CKPT = PREFERRED_CKPT if PREFERRED_CKPT.exists() else FALLBACK_CKPT
DEFAULT_DATA_DIR = Path("/root/autodl-tmp/DMS/data/canopus")
if not DEFAULT_DATA_DIR.exists():
    DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "canopus"


def as_hydra_path(path):
    """Return a path string that Hydra can consume on Linux/Windows."""
    return str(Path(path).expanduser().resolve()).replace("\\", "/")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter search for Spec2Mol sampling corrector."
    )
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT), help="Checkpoint used by general.test_only.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="CANOPUS data directory.")
    parser.add_argument("--split-file", default=None, help="Override split TSV path.")
    parser.add_argument("--subform-folder", default=None, help="Override subformulae directory.")
    parser.add_argument("--labels-file", default=None, help="Override labels.tsv path.")
    parser.add_argument("--spec-folder", default=None, help="Override spec_files directory.")
    parser.add_argument("--n-calls", type=int, default=12, help="Total gp_minimize evaluations.")
    parser.add_argument("--n-initial-points", type=int, default=5, help="Random initial evaluations.")
    parser.add_argument("--test-samples", type=int, default=10, help="general.test_samples_to_generate.")
    parser.add_argument("--seed", type=int, default=42, help="Bayesian search random seed.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess runs.")
    parser.add_argument("--wandb", default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--fail-score", type=float, default=-1.0)
    parser.add_argument("--results-csv", default="hyper_search_results.csv")
    parser.add_argument("--best-csv", default="best_hyperparameters.csv")
    return parser.parse_args()


def build_paths(args):
    data_dir = Path(args.data_dir).expanduser().resolve()
    paths = {
        "ckpt": Path(args.ckpt).expanduser().resolve(),
        "data_dir": data_dir,
        "split_file": Path(args.split_file).expanduser().resolve()
        if args.split_file
        else data_dir / "splits" / "canopus_hplus_100_0.tsv",
        "subform_folder": Path(args.subform_folder).expanduser().resolve()
        if args.subform_folder
        else data_dir / "subformulae" / "subformulae_default",
        "labels_file": Path(args.labels_file).expanduser().resolve()
        if args.labels_file
        else data_dir / "labels.tsv",
        "spec_folder": Path(args.spec_folder).expanduser().resolve()
        if args.spec_folder
        else data_dir / "spec_files",
    }
    return paths


def validate_inputs(paths):
    required_files = {
        "checkpoint": paths["ckpt"],
        "split file": paths["split_file"],
        "labels file": paths["labels_file"],
    }
    required_dirs = {
        "data directory": paths["data_dir"],
        "subformulae directory": paths["subform_folder"],
        "spectra directory": paths["spec_folder"],
    }

    missing = []
    for label, path in required_files.items():
        if not path.is_file():
            missing.append(f"{label}: {path}")
    for label, path in required_dirs.items():
        if not path.is_dir():
            missing.append(f"{label}: {path}")

    if missing:
        message = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "The following required paths do not exist:\n"
            f"{message}\n\n"
            "Fix the paths or pass them explicitly, for example:\n"
            "python src/hyperparameter_search.py "
            "--ckpt checkpoints/diffms_canopus.ckpt "
            "--data-dir /root/autodl-tmp/DMS/data/canopus"
        )


############################################################
# Read metrics.csv
############################################################

def get_latest_metrics(run_name):
    run_log_dir = PROJECT_ROOT / "logs" / run_name / run_name
    metric_files = list(run_log_dir.rglob("metrics.csv"))

    if len(metric_files) == 0:
        fallback_dir = PROJECT_ROOT / "logs"
        metric_files = list(fallback_dir.rglob("metrics.csv"))

    if len(metric_files) == 0:
        raise RuntimeError("No metrics.csv found after the experiment finished.")

    metric_file = sorted(metric_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Reading metrics: {metric_file}")

    df = pd.read_csv(metric_file)
    if df.empty:
        raise RuntimeError(f"Metrics file is empty: {metric_file}")

    columns = df.columns.tolist()
    top1_col = find_metric_column(columns, ["test/acc_at_1", "acc_at_1", "top1"])
    top10_col = find_metric_column(columns, ["test/acc_at_10", "acc_at_10", "top10"])
    tan_col = find_metric_column(
        columns,
        ["test/tanimoto_at_1", "tanimoto_at_1", "top1_tanimoto"],
    )

    if top1_col is None:
        raise RuntimeError(
            "Cannot find Top-1 metric in metrics.csv. "
            f"Available columns are: {columns}"
        )

    top1 = latest_value(df, top1_col)
    top10 = latest_value(df, top10_col) if top10_col else None
    tanimoto = latest_value(df, tan_col) if tan_col else None
    return top1, top10, tanimoto


def find_metric_column(columns, candidates):
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for column in columns:
        low = column.lower()
        for candidate in candidates:
            if candidate.lower().replace("test/", "") in low:
                return column
    return None


def latest_value(df, column):
    values = df[column].dropna()
    if values.empty:
        return None
    value = values.iloc[-1]
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


############################################################
# Run one experiment
############################################################

def run_experiment(args, paths, tau, lam, t_apply, clip):
    run_name = (
        f"hyper_tau{tau:.3f}"
        f"_lam{lam:.3f}"
        f"_t{t_apply:.3f}"
        f"_c{clip:.1f}"
    )

    cmd = [
        args.python,
        str(PROJECT_ROOT / "src" / "spec2mol_main.py"),
        f"general.name={run_name}",
        "dataset=canopus",
        f"general.test_only={as_hydra_path(paths['ckpt'])}",
        "general.resume=null",
        "general.load_weights=null",
        "hydra.job.chdir=false",
        "hydra.run.dir=.",
        f"general.test_samples_to_generate={args.test_samples}",
        f"general.wandb={args.wandb}",
        f"general.gpus={args.gpus}",
        f"train.eval_batch_size={args.eval_batch_size}",
        f"train.num_workers={args.num_workers}",
        f"dataset.datadir={as_hydra_path(paths['data_dir'])}",
        f"dataset.split_file={as_hydra_path(paths['split_file'])}",
        f"dataset.subform_folder={as_hydra_path(paths['subform_folder'])}",
        f"dataset.labels_file={as_hydra_path(paths['labels_file'])}",
        f"dataset.spec_folder={as_hydra_path(paths['spec_folder'])}",
        "dataset.spec_features=peakformula",
        "model.encoder_type=mist",
        "general.encoder_finetune_strategy=null",
        "model.use_ion_bias=false",
        "model.use_heavy_atom_bias=false",
        "model.sampling_steps=80",
        "model.sampling_schedule=quadratic",
        "model.use_sampling_corrector=true",
        "model.use_per_sample_early_stop=false",
        "model.use_multitraj_rerank=false",
        f"model.corrector_temperature={tau}",
        f"model.corrector_edge_prior_strength={lam}",
        f"model.corrector_apply_until_t={t_apply}",
        f"model.corrector_logit_clip={clip}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    if args.wandb == "disabled":
        env["WANDB_MODE"] = "disabled"
    elif args.wandb == "offline":
        env["WANDB_MODE"] = "offline"

    print("\n" + "=" * 80)
    print(f"Running experiment: {run_name}")
    print(" ".join(cmd))
    print("=" * 80 + "\n")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print(f"Experiment failed with exit code {e.returncode}: {run_name}")
        print("Assigning fail score and continuing search.")
        print("=" * 80)
        print()
        return args.fail_score, None, None, None, "failed"

    top1, top10, tanimoto = get_latest_metrics(run_name)
    score = top1 if top1 is not None else 0.0

    if tanimoto is not None:
        score += 5.0 * tanimoto

    return score, top1, top10, tanimoto, "ok"


############################################################
# Main
############################################################

def main():
    args = parse_args()
    paths = build_paths(args)
    validate_inputs(paths)

    @use_named_args(SPACE)
    def objective(tau, lam, t_apply, clip):
        score, top1, top10, tanimoto, status = run_experiment(
            args,
            paths,
            tau,
            lam,
            t_apply,
            clip,
        )

        result = {
            "tau": tau,
            "lambda": lam,
            "t_apply": t_apply,
            "clip": clip,
            "top1": top1,
            "top10": top10,
            "tanimoto": tanimoto,
            "score": score,
            "status": status,
        }

        RESULTS.append(result)
        pd.DataFrame(RESULTS).to_csv(PROJECT_ROOT / args.results_csv, index=False)

        print()
        print("=" * 80)
        print(result)
        print("=" * 80)
        print()

        return -score

    result = gp_minimize(
        objective,
        SPACE,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        acq_func="EI",
        random_state=args.seed,
    )

    print()
    print("=" * 80)
    print("BEST PARAMETERS")
    print(result.x)
    print()
    print("BEST SCORE")
    print(-result.fun)
    print("=" * 80)

    best = pd.DataFrame(
        [{
            "tau": result.x[0],
            "lambda": result.x[1],
            "t_apply": result.x[2],
            "clip": result.x[3],
            "score": -result.fun,
        }]
    )
    best.to_csv(PROJECT_ROOT / args.best_csv, index=False)


if __name__ == "__main__":
    main()
