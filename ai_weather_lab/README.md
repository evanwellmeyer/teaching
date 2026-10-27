# AI weather model lab

Two Colab notebooks in which students run AI weather models from an ERA5 initial state, perturb the initial
state, and compare the response with a control forecast. Initial conditions are streamed from Google's public
ARCO-ERA5 store, so no accounts or data downloads are needed.

| Notebook | Contents |
|---|---|
| [neuralgcm_lab.ipynb](neuralgcm_lab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ai_weather_lab/neuralgcm_lab.ipynb) | [NeuralGCM](https://neuralgcm.readthedocs.io), a hybrid model (dynamical core plus learned physics), at 2.8° or 1.4°. Parts: a 5-day forecast vs. ERA5; the butterfly effect (growth of 0.01 K temperature noise); a dynamical perturbation that each student injects on a start date of their choosing (a cyclonic or anticyclonic vortex built from a streamfunction, mid-tropospheric subsidence or ascent built from a velocity potential, a warm or cold anomaly, or drying, optionally with a geostrophically and hydrostatically balanced mass field), with animated maps and cross-sections, a linearity test and a chaos baseline; comparing results across the class. |
| [earth2studio_lab.ipynb](earth2studio_lab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evanwellmeyer/teaching/blob/main/ai_weather_lab/earth2studio_lab.ipynb) | Purely data-driven models through NVIDIA's [Earth2Studio](https://github.com/NVIDIA/earth2studio): Pangu-Weather (24 h and 6 h), FourCastNet and DLWP. Parts: a 10-day forecast vs. ERA5; the butterfly effect; a localized warm anomaly; the same experiments compared across models; a student-designed experiment. |

Both notebooks need a GPU runtime (`Runtime → Change runtime type`).

## For instructors

* Setup and helper code is hidden behind `#@title` (Colab's form view). Students edit the cells with Colab form
  fields (marked ✏️ in the Earth2Studio notebook). Each part ends with discussion questions; there are no answer keys. The
  notebooks are stored without outputs.
* The default start date is 2021-02-08 00 UTC, just before the February 2021 Texas cold-air outbreak. Any date
  from 1940 to 2025 works.
* Package versions: `neuralgcm` (latest; a pinned fallback line is in the install cell) and
  `earth2studio[pangu,fcn,dlwp]==0.19.0`.
* The notebooks have been run on Colab; the NeuralGCM Part 3 perturbation experiment is new and has only been
  tested offline (its perturbation math and plots, with a stand-in for the model). Things to check:
  * the installs and imports (in the Earth2Studio notebook, a warning prints if ONNX Runtime cannot see the GPU,
    in which case Pangu runs on the CPU and is slow);
  * GPU memory on a free T4 (16 GB): FourCastNet and DLWP should fit; Pangu at 0.25° is the most likely to run
    out of memory;
  * run times: NeuralGCM Part 3 runs three 10-day forecasts with 6-hourly output (four with the optional
    linearity test), and each new forecast length or output interval triggers one JAX compilation (~1 min).
