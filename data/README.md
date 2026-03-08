# Data directories

Raw and processed data are **not tracked by git** (see `.gitignore`).

## Expected structure

```
data/
├── raw/
│   ├── Branched/          ← unprocessed signals from Mendeley
│   └── Looped/
└── processed/
    ├── cwt_branched_binary.h5
    ├── cwt_branched_multiclass.h5
    ├── stft_looped_binary.h5
    └── ...
```

Download the Mendeley dataset and place it under `data/raw/`,
or point `data.data_dir` in `configs/default.yaml` to your Google Drive path.
