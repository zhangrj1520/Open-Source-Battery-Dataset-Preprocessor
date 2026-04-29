# Open-Source-Battery-Dataset-Preprocessor

A toolkit for preprocessing open-source battery datasets. 

This project aims to convert publicly available battery aging datasets from different research institutions (Tongji, XJTU, MIT, HUST, etc.) and various formats (`.mat`, `.csv`, `.pkl`, etc.) into a high-performance, easy-to-analyze, and structurally unified standardized `.parquet` format.

---

## 🌟 Overview

When performing SOH estimation or lifetime prediction for batteries, researchers often spend a significant amount of time on tedious data cleaning. This project helps researchers convert battery data from different datasets into a unified format, facilitating data visualization, feature extraction, and model training.

**Key features:**
- **Unified Format**: Whether original data are complex nested `.mat` files or large `.csv` files, the output is consistently a uniform `.parquet` format.
- **Operating Stage Segmentation**: Identifies charging, discharging, and resting phases, and assigns corresponding flags.
- **Time Alignment**: Resets the relative time within each cycle, making data alignment and feature extraction more convenient.
- **Storage Optimization**: Uses `.parquet` format for storage, resulting in smaller file sizes and faster reads.

---

## 📋 Unified Data Format

The preprocessed data will strictly adhere to the following naming conventions:

| Field Name | Description | Unit |
| :--- | :--- | :--- |
| `cycle_number` | Battery cycle number | / |
| `time` | Relative time within a cycle (starts at 0) | s |
| `voltage` | Terminal voltage | V |
| `current` | Current (positive for charge, negative for discharge) | A |
| `charge_stage` | Operating stage (1: charging, 2: post-charge rest, 3: discharging, 4: post-discharge rest) | / |
| `capacity` | Cumulative capacity for the current charge stage | Ah |
| (if available) `temperature` | Temperature | ℃ |

---

## 📊 Planned Datasets

- [x] **Tongji Dataset:** [Source](https://doi.org/10.5281/zenodo.6379165")
- [x] **XJTU Dataset:** [Source](https://zenodo.org/records/10963339)
- [ ] **MIT Dataset:** [Source](https://data.matr.io/1/#projects/5c48dd2bc625d700019f3204)
- [ ] **HUST Dataset:** [Source](https://data.mendeley.com/datasets/nsc7hnsg4s/2)
- [ ] **CACLE Dataset:** [Source](https://calce.umd.edu/battery-data)
- [ ] **SNU Dataset:** [Source](https://doi.org/10.17632/h2y7mj4kt7.2)

*(More to be added...)*

---

## 🚀 Quick Start

(Coming soon...)

---

## 🚧 Project Status
This repository is in its **early stages**. The core parsing logic for the aforementioned datasets is being implemented.