#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General First-level GLM Pipeline (nilearn)
Configured for multi-session BIDS datasets and parametric modulation.
No run entity — filenames follow: {sub}_ses-{ses}_task-{task}_*
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

warnings.filterwarnings('ignore')

# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
HPC          = '/nfs/roberts/pi/pi_il77/Alex'
PROJECT_NAME = 'foodvaluation'

DATA_DIR         = f'{HPC}/{PROJECT_NAME}'
BIDS_ROOT        = f'{DATA_DIR}/foodvaluation_BIDS'
DERIVATIVES_ROOT = f'{BIDS_ROOT}/derivatives'
OUTPUT_DIR       = f'{DATA_DIR}/results/nilearn_task_BI'

# ── Subjects ─────────────────────────────────────────────────────────────────
SUBJECT_LIST = ['003', '005', '013', '020', '023', '025', '026', '029', '030', '101', '108', '117', '121', '201', '214', '215', '220', '008', '017', '022', '027', '112', '124', '212', '219', '002', '019', '004', '021', '006', '007', '024', '010', '011', '012', '028', '016', '114', '102', '104', '119', '110', '116', '213', '204', '205', '207', '209', '211', '221'] #203

#decoupling, set in the notebook not here -- just labeled here for easy finding
#SUBJECT_LIST = ['003', '005', '013', '020', '023', '025', '026', '029', '030', '101', '108', '117', '121', '201', '214', '215', '220', '008', '017', '022', '027', '112', '124', '212', '219'] #25 subjects, exclude 203
# OTHER HALF (n=25) for concentation in notebook: ['002', 019', '004', '021', '006', '007', '024', '010', '011', '012', '028', '016', '114', '102', '104', '119', '110', '116', '213', '204', '205', '207', '209', '211', '221']

# ── Experiment Parameters ────────────────────────────────────────────────────
FWHM      = 6
TR        = 1.0
REMOVE_TR = 4
HIGH_PASS = 1/128.
N_JOBS    = 4
N_MOTION  = 6    # 6 = basic motion params; 24 = include derivatives & squares
N_COMPCOR = 5    # number of aCompCor noise regressors to include

# ── Session/Task Identifiers ─────────────────────────────────────────────────
RUNS = [
    {'session': '1', 'task': 'learning'},
    # {'session': '2', 'task': 'learning'},
]

# ── Contrasts ────────────────────────────────────────────────────────────────
CONTRASTS = {
    # --- Main effects ---
    'CorrectAligned':     'aligned_correctresp',
    'CorrectUnaligned':   'unaligned_correctresp',
    'IncorrectAligned':   'aligned_incorrectresp',
    'IncorrectUnaligned': 'unaligned_incorrectresp',

    # --- "Any stim / resp / outcome" ---
    'stim':    'baseline + unaligned_incorrectresp + unaligned_correctresp + aligned_correctresp + aligned_incorrectresp + '
               'unaligned_incorrectoutcome + unaligned_correctoutcome + aligned_correctoutcome + aligned_incorrectoutcome + toolate',
    'resp':    'unaligned_incorrectresp + unaligned_correctresp + aligned_correctresp + aligned_incorrectresp',
    'outcome': 'unaligned_incorrectoutcome + unaligned_correctoutcome + aligned_correctoutcome + aligned_incorrectoutcome', ## !!! note that based on event files, correct/incorrect outcome is encoding probabilistic events, not correct/incorrect resp trials as outcomes !!!

    # --- Aligned vs Unaligned ---
    'Aligned':              'aligned_correctresp + aligned_incorrectresp',
    'Unaligned':            'unaligned_correctresp + unaligned_incorrectresp',
    'Aligned_gt_Unaligned': '(aligned_correctresp + aligned_incorrectresp) - (unaligned_correctresp + unaligned_incorrectresp)',

    # --- Correct vs Incorrect during resp ---
    'Correct':             'aligned_correctresp + unaligned_correctresp',
    'Incorrect':           'aligned_incorrectresp + unaligned_incorrectresp',
    'Correct_gt_Incorrect':'(aligned_correctresp + unaligned_correctresp) - (aligned_incorrectresp + unaligned_incorrectresp)',

    # --- Within-alignment accuracy effects ---
    'CorrectAligned_gt_IncorrectAligned':     'aligned_correctresp - aligned_incorrectresp',
    'CorrectUnaligned_gt_IncorrectUnaligned': 'unaligned_correctresp - unaligned_incorrectresp',

    # --- Interaction-style contrasts ---
    'CorrectAligned_gt_CorrectUnaligned':     'aligned_correctresp - unaligned_correctresp',
    'IncorrectAligned_gt_IncorrectUnaligned': 'aligned_incorrectresp - unaligned_incorrectresp',
}

# =============================================================================
# 2. BACKEND (EDIT WITH CAUTION)
# =============================================================================

def get_stim_contrast(design_matrix):
    """Build stim contrast from whatever conditions are actually present."""
    all_stim_conds = [
        'baseline', 'unaligned_incorrectresp', 'unaligned_correctresp',
        'aligned_correctresp', 'aligned_incorrectresp',
        'unaligned_incorrectoutcome', 'unaligned_correctoutcome',
        'aligned_correctoutcome', 'aligned_incorrectoutcome', 'toolate',
    ]
    present = [c for c in all_stim_conds if c in design_matrix.columns]
    return ' + '.join(present)


def _get_nuisance_cols(n_motion, n_compcor):
    """Generates the lists of fMRIPrep column names based on user settings."""
    base_motion = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    motion_cols = []

    if n_motion >= 6:
        motion_cols.extend(base_motion)
    if n_motion == 24:
        for col in base_motion:
            motion_cols.extend([
                f'{col}_derivative1',
                f'{col}_power2',
                f'{col}_derivative1_power2'
            ])

    noise_cols = ['std_dvars', 'framewise_displacement']
    noise_cols.extend([f'a_comp_cor_{i:02d}' for i in range(n_compcor)])

    return motion_cols, noise_cols


def _bids_paths(sub, ses, task):
    func_base = f"{DERIVATIVES_ROOT}/{sub}/ses-{ses}/func/{sub}_ses-{ses}_task-{task}"
    return {
        'func':       f"{func_base}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'mask':       f"{func_base}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'regressors': f"{func_base}_desc-confounds_timeseries.tsv",
        'events':     f"{DATA_DIR}/event_files/{sub}/{task}_{sub}_NILEARNtaskBI.csv",
    }


def _drop_rank_deficient_columns(dm):
    """
    Remove columns from a design matrix that cause rank deficiency.

    Strategy:
      1. Drop any column with zero (or near-zero) variance — these are
         always safe to remove as they carry no information.
      2. Iteratively drop columns that are linearly dependent on earlier
         columns using QR decomposition with column pivoting.

    The 'constant' column generated by nilearn's cosine drift basis can
    become linearly dependent with the drift regressors for long scans
    (~32 min+). This function catches that case automatically.
    """
    # Step 1: drop zero-variance columns
    zero_var = [c for c in dm.columns if dm[c].std() < 1e-10]
    if zero_var:
        print(f"    [FIX] Dropping zero-variance columns: {zero_var}")
    dm = dm.drop(columns=zero_var)

    # Step 2: QR-based rank check — drop linearly dependent columns
    X = dm.values.astype(float)
    _, R, pivot = np.linalg.qr(X, mode='complete'), None, None

    # Use pivoted QR via scipy for reliable pivot detection
    from scipy.linalg import qr
    _, _, pivot = qr(X, pivoting=True)
    rank = np.linalg.matrix_rank(X)

    if rank < X.shape[1]:
        # Keep only the first `rank` pivot columns
        keep_idx    = sorted(pivot[:rank])
        drop_idx    = sorted(pivot[rank:])
        drop_cols   = [dm.columns[i] for i in drop_idx]
        print(f"    [FIX] Dropping linearly dependent columns (QR pivot): {drop_cols}")
        dm = dm.iloc[:, keep_idx]

    return dm


def process_events(events_path):
    """
    Cond-only events builder (no parametric modulators).

    Reads a CSV/TSV with at least: onset, duration, cond
    Returns a nilearn-format events DataFrame with: onset, duration, trial_type

    Notes:
    - Applies the global dummy-scan removal by subtracting (REMOVE_TR * TR)
      from all onsets, then drops events that start before time 0.
    - Does NOT create 'stim', 'resp', or 'outcome' regressors.
    """
    sep = '\t' if str(events_path).endswith('.tsv') else ','
    events = pd.read_csv(events_path, sep=sep, na_values=['', ' ', 'NA', 'NaN'])

    # Required columns for a condition-only model
    required = {'onset', 'duration', 'cond'}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            f"Events file {events_path} is missing columns: {missing}\n"
            f"Found columns: {list(events.columns)}"
        )

    # Basic cleaning
    events = events.copy()
    events['trial_type'] = events['cond'].astype(str).str.strip()
    events['onset'] = pd.to_numeric(events['onset'], errors='coerce')
    events['duration'] = pd.to_numeric(events['duration'], errors='coerce')

    bad = events['onset'].isna() | events['duration'].isna() | (events['duration'] <= 0)
    if bad.any():
        bad_rows = events.loc[bad, ['cond', 'onset', 'duration']]
        raise ValueError(
            f"Events file {events_path} has invalid onset/duration rows:\n{bad_rows}"
        )

    # Apply dummy-scan removal shift
    shift = REMOVE_TR * TR
    raw_median = float(events['onset'].median())
    if raw_median < shift + 5:
        print(f"    [WARN] Median onset={raw_median:.1f}s is low vs shift={shift:.1f}s. "
              f"Events might already be dummy-scan corrected; verify timing origin.")

    events['onset'] = events['onset'] - shift

    # Drop events that would start before the first kept volume
    events = events.loc[events['onset'] >= 0].reset_index(drop=True)
    if events.empty:
        raise ValueError(
            f"All events have negative onset after removing {REMOVE_TR} TRs "
            f"(shift={shift}s). Check REMOVE_TR and the events time origin."
        )

    nilearn_events = events[['onset', 'duration', 'trial_type']].copy()

    trial_counts = nilearn_events['trial_type'].value_counts()
    print(f"    [INFO] Trial counts: {trial_counts.to_dict()}")

    sparse = [c for c, n in trial_counts.items() if n < 3]
    if sparse:
        print(f"    [WARN] Sparse conditions (< 3 trials): {sparse}")

    return nilearn_events


def _build_design_matrix(func_img, events_df, confounds, n_scans):
    """
    Build the design matrix explicitly using make_first_level_design_matrix,
    then clean it to remove rank-deficient columns before passing to GLM.
    Returns a cleaned DataFrame ready to use as add_regs in a no-drift GLM.
    """
    frame_times = np.arange(n_scans) * TR

    dm = make_first_level_design_matrix(
        frame_times,
        events=events_df,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=HIGH_PASS,
        add_regs=confounds,
        oversampling=50,
    )

    dm_clean = _drop_rank_deficient_columns(dm)
    return dm_clean


def load_run(sub, ses, task):
    """Load and preprocess a single session/task."""
    paths = _bids_paths(sub, ses, task)

    req_missing = [k for k in ['func', 'events', 'regressors'] if not Path(paths[k]).exists()]
    if req_missing:
        print(f"    [SKIP] ses-{ses}_task-{task}: missing files:")
        for k in req_missing:
            print(f"           {k}: {paths[k]}")
        return None

    try:
        events_df = process_events(Path(paths['events']))
    except Exception as e:
        print(f"    [SKIP] ses-{ses}_task-{task}: events error — {e}")
        return None

    func_img = image.load_img(paths['func'])
    func_img = image.index_img(func_img, slice(REMOVE_TR, None))
    func_img = image.smooth_img(func_img, fwhm=FWHM)

    regressors = pd.read_csv(paths['regressors'], sep='\t')
    regressors = regressors.iloc[REMOVE_TR:].reset_index(drop=True)

    motion_cols, noise_cols = _get_nuisance_cols(N_MOTION, N_COMPCOR)
    avail_motion = [c for c in motion_cols if c in regressors.columns]
    avail_noise  = [c for c in noise_cols  if c in regressors.columns]
    confounds = pd.concat([
        regressors[avail_motion],
        regressors[avail_noise],
    ], axis=1).fillna(0.0)

    return func_img, events_df, confounds


def _patch_contrast_for_missing(contrast_def, present_cols, contrast_name):
    """Zero-out any missing regressors in a contrast string."""
    ops = {'+', '-', '*', '/', '(', ')'}
    buf = contrast_def
    for ch in ops:
        buf = buf.replace(ch, ' ')
    tokens = [t for t in buf.split() if t and not t.replace('.', '').lstrip('-').isdigit()]

    missing = [t for t in tokens if t not in present_cols]
    if not missing:
        return contrast_def

    if set(tokens).issubset(set(missing)):
        return None  # entire contrast is non-estimable

    patched = contrast_def
    for m in missing:
        patched = re.sub(rf'\b{re.escape(m)}\b', '0', patched)
    return patched


def save_design_qc(design_matrices, out_path, sub, run_labels):
    for dm, label in zip(design_matrices, run_labels):
        csv_path = out_path / f'{sub}_{label}_design_matrix.csv'
        png_path = out_path / f'{sub}_{label}_design_matrix.png'

        dm.to_csv(csv_path)

        n_cols = len(dm.columns)
        fig_w  = max(12, n_cols * 0.35)
        ax     = plot_design_matrix(dm)
        ax.set_title(f'{sub}  |  {label}', fontsize=11)
        fig    = ax.get_figure()
        fig.set_size_inches(fig_w, 10)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)


def run_subject(subject_id):
    sub = f'sub-{subject_id}'
    print(f"\n{'─'*55}\n  {sub}\n{'─'*55}")

    out_path = Path(OUTPUT_DIR) / sub
    out_path.mkdir(parents=True, exist_ok=True)

    func_imgs, design_matrices, run_labels = [], [], []

    for run_info in RUNS:
        ses   = run_info['session']
        task  = run_info['task']
        label = f"ses-{ses}_task-{task}"
        print(f"  Loading {label}...")

        result = load_run(sub, ses, task)
        if result is None:
            continue

        func_img, events_df, confounds_df = result
        n_scans = func_img.shape[-1]

        print(f"  Building design matrix ({n_scans} scans)...")
        dm_clean = _build_design_matrix(func_img, events_df, confounds_df, n_scans)

        rank   = np.linalg.matrix_rank(dm_clean.values)
        n_cols = dm_clean.shape[1]
        print(f"  Design matrix: {n_scans} x {n_cols}  |  rank={rank}  "
              f"({'FULL RANK ✓' if rank == n_cols else '*** STILL RANK DEFICIENT ***'})")

        func_imgs.append(func_img)
        design_matrices.append(dm_clean)
        run_labels.append(label)

    if not func_imgs:
        print(f"  [SKIP] No valid runs for {sub}.")
        return

    # ── Fit GLM using pre-built (cleaned) design matrices ────────────────────
    # drift_model=None because drift regressors are already in dm_clean
    print(f"\n  Fitting GLM on {len(func_imgs)} session(s)...")

    glm = FirstLevelModel(
        t_r=TR,
        hrf_model=None,          # HRF already convolved in dm_clean
        drift_model=None,        # drift already in dm_clean
        noise_model='ar1',
        standardize=False,
        smoothing_fwhm=None,
        minimize_memory=False,
        n_jobs=1,
    )

    try:
        # When passing pre-built design matrices, use the design_matrices arg
        glm.fit(func_imgs, design_matrices=design_matrices)
    except np.linalg.LinAlgError as e:
        print(f"  [CRASH] {sub}: SVD failed — {e}")
        return
    except Exception as e:
        print(f"  [CRASH] {sub}: GLM fit failed — {e}")
        return

    save_design_qc(design_matrices, out_path, sub, run_labels)

    present_cols = set(glm.design_matrices_[0].columns)

    print(f"\n  Computing contrasts...")
    for contrast_name, contrast_def in CONTRASTS.items():
        try:
            if contrast_name == 'stim':
                contrast_def = get_stim_contrast(glm.design_matrices_[0])
            else:
                contrast_def = _patch_contrast_for_missing(
                    contrast_def, present_cols, contrast_name
                )

            if contrast_def is None:
                print(f"    – {contrast_name}: skipped (all regressors absent)")
                continue

            z_map   = glm.compute_contrast(contrast_def, stat_type='t', output_type='z_score')
            t_map   = glm.compute_contrast(contrast_def, stat_type='t', output_type='stat')
            eff_map = glm.compute_contrast(contrast_def, stat_type='t', output_type='effect_size')
            var_map = glm.compute_contrast(contrast_def, stat_type='t', output_type='effect_variance')

            z_map.to_filename(  out_path / f'{sub}_{contrast_name}_z.nii.gz')
            t_map.to_filename(  out_path / f'{sub}_{contrast_name}_t.nii.gz')
            eff_map.to_filename(out_path / f'{sub}_{contrast_name}_effect.nii.gz')
            var_map.to_filename(out_path / f'{sub}_{contrast_name}_variance.nii.gz')

            print(f"    ✓ {contrast_name}")

        except Exception as e:
            print(f"    ✗ {contrast_name}: {e}")

    print(f"\n  Saved to: {out_path}")


# =============================================================================
# 3. ENTRY POINT
# =============================================================================

if __name__ == '__main__':

    print("=" * 55)
    print("  PRE-FLIGHT FILE CHECK")
    print("=" * 55)
    any_missing = False
    for sid in SUBJECT_LIST:
        sub = f'sub-{sid}'
        for run_info in RUNS:
            paths = _bids_paths(sub, run_info['session'], run_info['task'])
            for key in ['func', 'events', 'regressors']:
                p = Path(paths[key])
                if not p.exists():
                    print(f"  [MISSING] {sub} | {key}:\n           {p}")
                    any_missing = True
    if not any_missing:
        print("  All required files found.")
    print("=" * 55 + "\n")

    Parallel(n_jobs=N_JOBS, verbose=10, backend='loky')(
        delayed(run_subject)(sid) for sid in SUBJECT_LIST
    )