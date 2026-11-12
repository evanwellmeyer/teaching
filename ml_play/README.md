# ML play

Notebooks for exploring ERA5 with machine learning.

| Notebook | Contents |
|---|---|
| [unsupervised_era5.ipynb](unsupervised_era5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ml_play/unsupervised_era5.ipynb) | For after an introductory lecture on machine learning: an example of unsupervised learning, and of using the data to explore a curiosity. Uses 64 years of monthly ERA5 (WeatherBench 2's 5.6° copy, 1959-2022). Parts: the Niño 3.4 index from ERA5 SST and its Fourier spectrum (seasonal cycle, then the El Niño band of the anomaly, and a rebuild from the strongest waves); EOFs of one or two fields in a chosen box and months (the SVD in plain view, regression maps, PCs, PC 1 vs. Niño 3.4); and spherical harmonics (example modes, the power spectrum of a field, and rebuilds from more and more modes). |

It runs on a CPU runtime. ERA5 is streamed from WeatherBench 2's public copy on Google Cloud, so no accounts or data
downloads are needed. Single-level fields load in seconds; fields on pressure levels take a few minutes. The notebook
is stored without outputs.
