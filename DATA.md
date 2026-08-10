# Data access and expected layout

This repository does not redistribute ultrasound images or source annotations. Obtain the
data from the original public records:

1. Meiburger et al., *Carotid Ultrasound Boundary Study (CUBS): Clinical Dataset* (version
   1), Mendeley Data. <https://doi.org/10.17632/fpv535fss7.1>
2. Meiburger et al., *Carotid Ultrasound Boundary Study (CUBS): Technical Dataset* (version
   1), Mendeley Data. <https://doi.org/10.17632/m7ndn58sv6.1>

Set `CUBS_DATA_ROOT` to the parent directory in this layout:

```text
<CUBS_DATA_ROOT>/
├── cubs_clinical/
│   ├── IMAGES/
│   ├── SEGMENTATIONS/
│   │   └── Manual-A1/
│   └── ClinicalDatabase-CUBS.csv
└── cubs_technical/
    ├── images/
    ├── LIMA-Profiles/
    │   └── Manual-A1/
    └── TechnicalDatabase-CUBS.*
```

The exact capitalization in the downloaded archive should be preserved. If the technical
metadata filename differs between release mirrors, update only its locator in
`notebooks/E0_build_dataset_index.py`.

`data/master_index.csv` contains relative paths, pseudonymous public-dataset identifiers,
acquisition strata, calibration factors, and split-related metadata. It contains no image
pixels or direct identifiers. The fixed partitions are patient-disjoint. Do not replace
them with image-level random splits because bilateral images from one participant must
remain in the same partition.

Users are responsible for checking and following the licenses, citation requests, and
ethical conditions attached to the original datasets.
