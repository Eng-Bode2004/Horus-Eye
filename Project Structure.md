horus_eye/
│
├── 📁 data/                    # Raw and processed EEG data
│   ├── raw/                   # Original dataset (train/test)
│   ├── processed/             # Filtered, segmented, labeled data
│   └── metadata.json          # Task type, subject info, trial IDs
│
├── 📁 notebooks/              # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_demo.ipynb
│   └── 03_model_analysis.ipynb
│
├── 📁 preprocessing/          # EEG signal preprocessing scripts
│   ├── filter.py              # Bandpass filter, noise removal
│   ├── segment.py             # Epoching / windowing
│   └── feature_extraction.py  # PSD, FFT, ERD/ERS features
│
├── 📁 models/                 # Deep learning models
│   ├── mi_model.py           # EEGNet or CNN for Motor Imagery
│   ├── ssvep_model.py        # FFT-based CNN or MLP for SSVEP
│   └── base_model.py         # Shared base classes (optional)
│
├── 📁 training/               # Training logic
│   ├── train_mi.py           # Train MI model
│   ├── train_ssvep.py        # Train SSVEP model
│   └── utils.py              # Metrics, callbacks, plotting
│
├── 📁 inference/              # Inference pipeline
│   ├── predict.py            # Main prediction runner
│   ├── predict_mi.py         # MI classifier only
│   ├── predict_ssvep.py      # SSVEP classifier only
│   └── router.py             # Routes EEG trial to MI or SSVEP model
│
├── 📁 evaluation/             # Scoring, confusion matrix, visualizations
│   ├── evaluate.py
│   └── metrics.py
│
├── 📁 deployment/             # Optional: Drone control integration
│   ├── bci_to_command.py     # Map MI/SSVEP output to drone instructions
│   └── flight_controller.py  # Drone SDK integration (e.g., DJI SDK, ArduPilot)
│
├── 📁 submission/             # Files to submit to AIC-3
│   ├── predictions.csv        # Final test predictions
│   ├── model_weights/         # .pth or .h5 files
│   ├── inference_script.py    # To load & run models
│   └── system_description.pdf # Required documentation
│
├── 📁 configs/                # YAML/JSON configs
│   ├── mi_config.yaml
│   └── ssvep_config.yaml
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project description
└── main.py                    # Optional entry point


