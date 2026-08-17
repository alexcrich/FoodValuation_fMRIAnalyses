#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General First-level GLM Pipeline (nilearn)
Configured for multi-session BIDS datasets and parametric modulation.
No run entity — filenames follow: {sub}_ses-{ses}_task-{task}_*
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import Parallel, delayed

from nilearn import image
from nilearn.glm.first_level import FirstLevelModel
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
OUTPUT_DIR       = f'{DATA_DIR}/results/nilearn_parametric'

# ── Subjects ─────────────────────────────────────────────────────────────────
SUBJECT_LIST = ['007', '030', '002', '003', '004', '005', '006', '008', '010',
                '011', '012', '013', '016', '017', '019', '020', '021', '022',
                '023', '024', '025', '026', '027', '028', '029', '101', '102',
                '104', '108', '110', '112', '114', '116', '117', '119', '121',
                '124', '201', '204', '205', '207', '209', '211', '212', '213',
                '214', '215', '219', '220', '221']

# ── Experiment Parameters ────────────────────────────────────────────────────
FWHM      = 6
TR        = 1.0
REMOVE_TR = 4
HIGH_PASS = 1/128.
N_JOBS    = 4
N_MOTION  = 6    # 6 = basic motion params; 24 = include derivatives & squares
N_COMPCOR = 5    # number of aCompCor noise regressors to include

# ── Session/Task Identifiers ─────────────────────────────────────────────────
# No run entity. Each dict entry maps to: ses-{session}_task-{task} in filenames.
# Multiple entries here will be concatenated into a single fixed-effects model.
RUNS = [
    {'session': '1', 'task': 'rating'},
    # {'session': '2', 'task': 'rating'},  # add more sessions/tasks as needed
]

# ── Parametric Modulators ────────────────────────────────────────────────────
# Events files now contain the column "parametric" (formerly "modulation").
MODULATOR_COL       = 'parametric'
MEAN_CENTER_MODS    = True

# ── Contrasts ────────────────────────────────────────────────────────────────
CONTRASTS = {
    # Main effects (one regressor per condition)
#    'Liked':    'Liked',
#    'Disliked': 'Disliked',
#    'Neutral':  'Neutral',

    # Binary comparisons
#    'Liked_gt_Disliked': 'Liked - Disliked',
#    'Liked_gt_Neutral':  'Liked - Neutral',

    # stim, on/off
    'stim': 'stim',

    # Parametric: voxels whose activity tracks subjective value
    'Value': 'value',
}

# =============================================================================
# 2. BACKEND (EDIT WITH CAUTION)
# =============================================================================

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
    """
    Build file paths for a session/task with no run entity.
    Derivatives: {sub}_ses-{ses}_task-{task}_space-..._bold.nii.gz
    Events:      event_files_pilot_GLM/{sub}/{task}_{sub}_NILEARNparametric.csv
    """
    func_base = f"{DERIVATIVES_ROOT}/{sub}/ses-{ses}/func/{sub}_ses-{ses}_task-{task}"

    return {
        'func':       f"{func_base}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        'mask':       f"{func_base}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
        'regressors': f"{func_base}_desc-confounds_timeseries.tsv",
        'events':     f"{DATA_DIR}/event_files_pilot_GLM/{sub}/{task}_{sub}_NILEARNparametric.csv",
    }


def process_events(events_path):
    """
    Builds a nilearn-format events DataFrame.

    Expects columns in the *source* events file:
      onset, duration, cond, parametric, stim

    Behavior:
      - Always includes: 'stim' and pooled parametric regressor 'value'
      - Includes per-condition regressors from 'cond' ONLY IF any contrast
        appears to require them (e.g., 'Liked', 'Disliked', 'Neutral', or any
        contrast expression referencing them).
    """
    sep = '\t' if str(events_path).endswith('.tsv') else ','
    events = pd.read_csv(events_path, sep=sep, na_values=['', ' '])

    expected = {'onset', 'duration', 'cond', MODULATOR_COL, 'stim'}
    missing  = expected - set(events.columns)
    if missing:
        raise ValueError(
            f"Events file {events_path} is missing columns: {missing}\n"
            f"Found columns: {list(events.columns)}"
        )

    events['trial_type'] = events['cond'].astype(str).str.strip()

    # Remove TR shift from onset times
    shift = REMOVE_TR * TR
    events['onset'] = events['onset'] - shift
    events = events[events['onset'] >= 0].reset_index(drop=True)

    # Mean-center the parametric modulator across ALL trials
    if MEAN_CENTER_MODS and MODULATOR_COL in events.columns:
        events[MODULATOR_COL] = events[MODULATOR_COL] - events[MODULATOR_COL].mean()

    # Decide whether to include condition regressors (from cond)
    # We include them only if the CONTRASTS definitions reference anything
    # other than 'stim' and 'value' (and basic operators).
    reserved = {'stim', 'value'}
    ops = {'+', '-', '*', '/', '(', ')'}
    tokens = set()

    for cdef in CONTRASTS.values():
        if isinstance(cdef, str):
            # crude tokenizer: split on whitespace and operators
            parts = []
            buf = cdef
            for ch in ops:
                buf = buf.replace(ch, ' ')
            parts = buf.split()
            tokens.update(parts)

    include_cond = len(tokens - reserved) > 0

    nilearn_events = []
    for _, row in events.iterrows():

        # ── Optional: condition main effects (only if needed by contrasts) ──
        if include_cond:
            nilearn_events.append({
                'onset':      row['onset'],
                'duration':   row['duration'],
                'trial_type': row['trial_type'],
                'modulation': 1.0,
            })

        # ── stim regressor ────────────────────────────────────────────────
        if pd.notna(row['stim']) and float(row['stim']) != 0.0:
            nilearn_events.append({
                'onset':      row['onset'],
                'duration':   row['duration'],
                'trial_type': 'stim',
                'modulation': float(row['stim']),
            })

        # ── Parametric value regressor ────────────────────────────────────
        if pd.notna(row[MODULATOR_COL]):
            nilearn_events.append({
                'onset':      row['onset'],
                'duration':   row['duration'],
                'trial_type': 'value',
                'modulation': float(row[MODULATOR_COL]),
            })

    return pd.DataFrame(nilearn_events)


def load_run(sub, ses, task):
    """Load and preprocess a single session/task (no run entity)."""
    paths = _bids_paths(sub, ses, task)

    req_missing = [k for k in ['func', 'events', 'regressors'] if not Path(paths[k]).exists()]
    if req_missing:
        print(f"    [SKIP] ses-{ses}_task-{task}: missing {req_missing}")
        return None

    # Isolate events errors so one bad subject doesn't crash the whole parallel pool
    try:
        events_df = process_events(paths['events'])
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


def save_design_qc(glm, out_path, sub, run_labels):
    for dm, label in zip(glm.design_matrices_, run_labels):
        csv_path = out_path / f'{sub}_{label}_design_matrix.csv'
        png_path = out_path / f'{sub}_{label}_design_matrix.png'

        dm.to_csv(csv_path)

        n_cols = len(dm.columns)
        fig_w  = max(12, n_cols * 0.35)
        ax = plot_design_matrix(dm)   # nilearn creates its own figure; returns axes
        ax.set_title(f'{sub}  |  {label}', fontsize=11)
        fig = ax.get_figure()
        fig.set_size_inches(fig_w, 10)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)


def run_subject(subject_id):
    sub = f'sub-{subject_id}'
    print(f"\n{'─'*55}\n  {sub}\n{'─'*55}")

    out_path = Path(OUTPUT_DIR) / sub
    out_path.mkdir(parents=True, exist_ok=True)

    func_imgs, events_list, confounds_list, run_labels = [], [], [], []

    for run_info in RUNS:
        ses  = run_info['session']
        task = run_info['task']
        label = f"ses-{ses}_task-{task}"
        print(f"  Loading {label}...")

        result = load_run(sub, ses, task)
        if result is None:
            continue

        func_img, events_df, confounds_df = result
        func_imgs.append(func_img)
        events_list.append(events_df)
        confounds_list.append(confounds_df)
        run_labels.append(label)

    if not func_imgs:
        print(f"  No valid runs for {sub}. Skipping.")
        return

    print(f"\n  Fitting GLM on {len(func_imgs)} session(s)...")

    glm = FirstLevelModel(
        t_r=TR,
        hrf_model='spm',
        drift_model='cosine',
        high_pass=HIGH_PASS,
        noise_model='ar1',
        standardize=False,
        smoothing_fwhm=None,
        minimize_memory=False,
        n_jobs=1,
    )

    glm.fit(func_imgs, events=events_list, confounds=confounds_list)
    save_design_qc(glm, out_path, sub, run_labels)

    print(f"\n  Computing contrasts...")
    for contrast_name, contrast_def in CONTRASTS.items():
        try:
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


if __name__ == '__main__':
    Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(run_subject)(sid) for sid in SUBJECT_LIST
    )