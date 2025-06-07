🧠 EEG Preprocessing Steps (for MI Classification)
1. Load Raw Data
Read EEG files (CSV, EDF, etc.).

Inspect and visualize the data.

2. Channel Selection
Select only relevant channels (e.g., C3, CZ, C4 for MI).

3. Re-referencing (optional, but common)
Re-reference signals (e.g., average reference or linked ears).

4. Bandpass Filtering
Apply a bandpass filter (e.g., 8–30 Hz for MI; covers mu and beta rhythms).

Removes slow drifts and high-frequency noise.

5. Artifact Removal
Remove or correct for eye blinks, muscle activity, or bad channels.

Simple: Manual inspection and channel rejection.

Advanced: ICA (Independent Component Analysis).

6. Epoching (Segmentation)
Split continuous data into epochs around events/trials (e.g., −1 to 4 sec around cue).

Each epoch = one MI trial.

7. Baseline Correction
Subtract mean signal from a pre-stimulus period (e.g., −200 to 0 ms before cue) for each epoch.

8. Feature Extraction
Compute features such as:

Power Spectral Density (PSD)

Band power (mu/beta)

CSP (Common Spatial Patterns)

Or use raw epochs for deep learning

9. Normalization/Standardization
Normalize features/epochs across trials for model compatibility.

10. Save Processed Data
Save clean, segmented, and feature-extracted data for modeling.