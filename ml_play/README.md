# ML play

Notebooks for exploring ERA5 with machine learning.

| Notebook | Contents |
|---|---|
| [enso_fourier.ipynb](enso_fourier.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ml_play/enso_fourier.ipynb) | Fourier analysis of NOAA's monthly Niño 3.4 record (1870-2025): the first few waves and their sums, the spectrum with and without the seasonal cycle, a rebuild from the strongest waves, and a train/test check of whether continuing the waves predicts El Niño. |
| [eofs_era5.ipynb](eofs_era5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ml_play/eofs_era5.ipynb) | Unsupervised learning on 64 years of daily ERA5 (WeatherBench 2's 1.5° copy, 00 UTC, 1959-2022): EOFs of a field in a chosen box (the SVD in plain view, regression maps, PCs, PC 1 vs. Niño 3.4), and spherical harmonics (example modes, the power spectrum of a field, and rebuilds from more and more modes). |
| [train_era5.ipynb](train_era5.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ml_play/train_era5.ipynb) | A small CNN in PyTorch trained on 10 years of daily-mean Z500 (WeatherBench 2's daily 1.5° copy) to predict the next day's map from today's, compared with persistence. Use a GPU runtime. |

ERA5 is streamed from WeatherBench 2's public copy on Google Cloud, so no accounts are needed. Single-level fields load in
a few minutes; fields on pressure levels (geopotential) download all 13 levels and are much slower. The notebooks are
stored without outputs.
