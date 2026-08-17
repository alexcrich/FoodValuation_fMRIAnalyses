# Chapter 4 fMRI Analysis Scripts — README

This folder contains the **first-level fMRI GLM scripts (Python/Nilearn)** and **associated Jupyter notebooks** used for the analyses described in **Chapter 4: “Neural Correlates of Food-Value Learning: Value Updating and Behavioral Inhibition”** of *Prior Preferences Interfere with the Associative Learning of Food Values*.

The three scripts and associated Notebooks correspond to **three first-level modeling “families”** described in Chapter 4:
1) **Value localizer (food rating task)** → define/validate value-responsive vmPFC ROI  
2) **Reversal learning (parametric value GLM)** → dissociate *Initial Value* vs *Model-Derived Value*  
3) **Reversal learning (behavioral inhibition GLM)** → accuracy/policy-adherence contrasts, split by congruency and epoch

---

- **Chapter 4.3 Results: Model-Free Behavior and Region of Interest (ROI) Functional Activity**
  - Behavioral Results
  - MRI Results: Value Representation (Fig 19-20)
  - MRI Results: Behavioral Inhibition (Fig 21-22)
  
# Script-by-script quick reference

## `firstlevel_parametric_nilearn.py`, `visualize-nilearn-rankregression.ipynb`
- **Use for:** rating/value localizer (value representation; ROI definition/visualization support)
- **Chapter 4 figures:** 19

## `firstlevel_task_valuation_nilearn.py`, `visualize-nilearn-taskvaluation.ipynb`
- **Use for:** reversal learning **parametric** value GLM (Initial Value and Model-Derived Value)
- **Chapter 4 figures:** 20

## `firstlevel_task_BI_nilearn.py`, `visualize-nilearn-taskBI.ipynb`
- **Use for:** reversal learning behavioral inhibition GLM (accuracy × congruency × epoch)
- **Chapter 4 figures:** 21, 22




---

## Shared assumptions across scripts (from Chapter 4 Methods)

see **FmriPrep_singularity.sh** script for preprocessing, which uses BIDS format files created using **createBIDS_master.ipynb**!

All three scripts implement **first-level GLMs in Nilearn** using fMRIPrep-derived inputs (events + confounds). Chapter 4 describes shared modeling choices used throughout the first-level analyses, including:

- Preprocessing via **fMRIPrep** (MNI152NLin2009cAsym, 2mm; confounds include motion, FD, DVARS, aCompCor)
- **SPM canonical HRF**
- **Cosine drift / high-pass (128s)**
- **6mm FWHM smoothing**
- Nuisance regressors: 6 motion params, FD, standardized DVARS, 5 aCompCor components (missing set to zero)
- (For the long reversal-learning scan) design matrices screened for **rank deficiency** and problematic columns removed



---

### A) Food item rating task (value localizer)
**Chapter 4 context:** Used to obtain inherent preference ratings and define a **value-related vmPFC ROI**.

**Script:** `firstlevel_parametric_nilearn.py`  
**Purpose:** First-level GLM for the **food rating (value localizer) task**, supporting value-related contrasts/estimation used for ROI definition/visualization.

**What the script does (high-level):**
- Loads BIDS-style images and run metadata (via internal `_bids_paths` / `load_run`)
- Reads **events** and **fMRIPrep confounds**
- Builds and fits a **Nilearn `FirstLevelModel`**
- Computes contrasts (`compute_contrast`) from a contrast definition built from design-matrix columns
- Produces design-matrix QC plots (`plot_design_matrix`) via `save_design_qc`

**Key functions worth pointing to:**
- `process_events(...)`: converts events into the format used by Nilearn GLMs
- `_get_nuisance_cols(...)`: selects nuisance regressors from confounds
- `build_contrasts_for_columns(...)`: convenience for contrast specification based on column names
- `run_subject(...)`: subject-level execution

**Outputs (conceptual):**
- First-level contrast maps for rating/value regressors
- QC artifacts (design matrix plots)

**Notebook (Results visualization, creation of ROI):** `visualize-nilearn-rankregression.ipynb`  
**What it contains (results/figures summary):**
- Creation of spherical 10mm-radius ROI mask (Fig 19), used for later value analysis in the task: ".../results/nilearn_parametric/ROI/sphere_r10mm_x-4_y43_z-9_mask_RANKREGRESSION.nii.gz"
- ROI-oriented plots consistent with the Results narrative (e.g., activity in that ROI across discrete rating levels, Strongly Like → Strongly Dislike, Fig 19c)



---

### B) Reversal learning task — Value Representation (parametric GLM)
**Chapter 4 section:** **4.3 Results → MRI Results: Value Representation**  
**Chapter 4 analysis description:** vmPFC ROI analysis using a reversal-learning GLM with **two simultaneous parametric modulators** on the prediction/response epoch:
    - **Initial Value** (participant’s initial rating category for each item)
    - **Model-Derived Value** (trial-wise RW value estimate)

**Script:** `firstlevel_task_valuation_nilearn.py`  
**Purpose:** First-level GLM for the reversal-learning task that includes **parametric value regressors** (consistent with Chapter 4’s “Initial Value” vs “Model-Derived Value” model).

**What the script does (high-level):**
- Loads reversal-learning fMRI runs (BIDS/fMRIPrep outputs)
- Loads events + confounds, selects nuisance columns (`_get_nuisance_cols`)
- Fits `FirstLevelModel` with HRF + high-pass drift settings
- Computes specified contrasts (`contrast_def`)
- Saves design-matrix QC plots (`save_design_qc`)

**Key functions:**
- `process_events(...)`: where parametric modulators would be incorporated (check how events columns are constructed)
- `run_subject(...)`: end-to-end subject-level execution

**Outputs (conceptual):**
- Subject-level maps for parametric effects (e.g., “Initial Value”, “Model-Derived Value”) and any derived contrasts

**Notebook (Results visualization):** `visualize-nilearn-taskvaluation.ipynb`  
**What it contains (results/figures summary):**
- Code under 2nd level -> "ROI: nilearn parametric ROI obtained from food item ratings" heading corresponds to corresponding to Figure 20a-b, plotting the effects of **Initial Value** ("none_glm_Value_init_all_sub_4d.nii.gz") and **Model-Derived Value** (none_glm_Value_Model_Derived_all_sub_4d.nii.gz) in the spherical ROI made via visualize-nilearn-rankregression.ipynb; outputs "/output/Valuation/Value_init_figures" and "/output/Valuation/Value_Model_Derived_figures"
           - exploratory Start Group comparison, BES correlation, and BMI correlation within rating-derived ROI for initial & model-derived value also under this heading,              which are not included in Chapter 4
- Code under "Making sure initial value and model-derived value predictors are not correlated" heading checks that Value_init and Value_Model_Derived are not correlated so they can be in the same GLM (reported in-text of Chapter 4!)
- Code under "Comparing Value_init to Value_model_derived: which accounts better for activation in vmPFC spherical ROI?" corresponds to the analysis shown in Fig 20c; outputs "output/Valuation/Value_compare_figures" 



---

### C) Reversal learning task — Behavioral Inhibition GLM (accuracy × congruency × epoch)
**Chapter 4 section:** **4.3 Results → MRI Results: Behavioral Inhibition**  
**Chapter 4 analysis description:** first-level model separating activity by:
      - **Congruent vs Incongruent** blocks/trials
      - **Correct vs Incorrect** predictions (policy adherence / accuracy)
      - **Response (prediction) vs Outcome** epoch

**Script:** `firstlevel_task_BI_nilearn.py`  
**Purpose:** First-level GLM for reversal-learning designed to test **behavioral inhibition/control** signals via contrasts involving **accuracy and congruency**, separately for response vs outcome epochs.

**What the script does (high-level):**
- Loads BIDS/fMRIPrep images, events, and confounds
- Builds a first-level design matrix explicitly (uses `make_first_level_design_matrix`)
- Includes safeguards for common GLM issues in long tasks:
  - `_drop_rank_deficient_columns(...)`
  - `_patch_contrast_for_missing(...)` (handles missing regressors for some subjects/runs)
- Fits `FirstLevelModel` and computes contrasts

**Key functions:**
- `_build_design_matrix(...)`: creates the expanded condition structure (epoch × accuracy × congruency)
- `get_stim_contrast(...)`: helper for building stimulus/condition contrasts by name patterns
- `_drop_rank_deficient_columns(...)`: removes problematic columns to stabilize estimation

**Outputs (conceptual):**
- Contrast maps for behavioral inhibition hypotheses (e.g., correct > incorrect, especially in incongruent blocks but also for congruent blocks; response-epoch vs outcome-epoch effects), depending on `contrast_def`

**Notebook (Results visualization):** `visualize-nilearn-taskBI.ipynb`  
**What it contains (results/figures summary):**
- Visualization of behavioral inhibition contrasts/maps
- Summaries aligned with the dissertation’s “MRI Results: Behavioral Inhibition” narrative (e.g., highlighting dlPFC/dmPFC involvement), where:
    - **code under "Whole sample" heading contains atlas-based analyses in Figure 21** (outputs in .../output/BI/fullsample/Atlas_Analyses_Figures; also contains exploratory start group comparison, BES and BMI analyses within atlas ROIs not reported in Chapter 4)
    - **code under "Decoupling" heading contains split-sample cross validation approach in Figure 22** (creates x14_y31_z47 based ROI mask from 1st half of sample findings seen in Fig 22a; then tests other half of sample's activation, "none_glm_CONTRASTID_decoupled_test_group_4d.nii.gz" where contrast IDs are correct/incorrect Congruent/Incongruent combos, in that same mask. tried different spherical versions, with "results/nilearn_task_BI/ROI/sphere_r13mm_x14_y31_z47_mask_DECOUPLED_FIRSTHALF.nii.gz" representing significnat 13mm-radius sphere seen in Fig 22b)
    - code under "full sample (check inputs!): Start Group comparison, BES correlation, and BMI correlation in the extracted sphere (13mm)" not reported in Chapter 4 (goes back to whole sample, but in sphere identified from decoupling, to do exploratory start group, BES, and BMI analyses; outputs in .../output/BI/fullsample)




---

run any of the above .py scripts with **run_1stlevel_nilearn.sh**, edit accordingly

---