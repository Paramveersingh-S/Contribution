# Google Summer of Code (GSoC) 2026 - DeepMind Contributions

This repository serves as a portfolio of my open-source contributions to Google DeepMind and Google Gemma projects. 

Below is a summary of the issues I tackled, the architectural challenges resolved, and the corresponding Pull Requests opened across various repositories.

---

## 🧬 Google DeepMind `torax`
Torax is a differentiable plasma transport simulator in JAX.

| Issue | Description | Pull Request / Branch |
| :--- | :--- | :--- |
| **#844** | Fix missing profile geometry initialization. | [PR Link / Branch `fix/issue-844`](https://github.com/Paramveersingh-S/torax) |
| **#1369** | Resolve dependency injection failures in the core simulation loop. | [PR Link / Branch `fix/issue-1369`](https://github.com/Paramveersingh-S/torax) |
| **#689** | Optimize JAX JIT compilation for mesh gradients. | [PR Link / Branch `fix/issue-689`](https://github.com/Paramveersingh-S/torax) |

---

## 🧠 Google DeepMind `gemma`
The core PyTorch and JAX implementation of the Gemma open models.

| Issue | Description | Pull Request / Branch |
| :--- | :--- | :--- |
| **#675** | **Gemma 4 Inference Context Overflow:** Implemented a rolling "Turn Pruning" cache mechanism in the `ChatSampler` to evict oldest dialogue turns and prevent JAX graph recompilation limits. | [PR Link / Branch `fix/gemma-context-overflow`](https://github.com/Paramveersingh-S/gemma) |

---

## 📚 Google Gemma `cookbook`
Official recipes, examples, and agentic workflows for Gemma models.

| Issue | Description | Pull Request / Branch |
| :--- | :--- | :--- |
| **#399** | **Distributed DGX Spark Orchestration:** Designed an architecture mapping the MacOS terminal multi-agent concurrency pipeline onto PySpark `mapInPandas` for true cluster execution. | [PR Link / Branch `feat/spark-concurrent-dgx`](https://github.com/Paramveersingh-S/cookbook) |
| **#372** | **Ollama Web Search Structured Parsing:** Created a fixed `Modelfile` instructing Ollama to respect Gemma 4's strict `<|tool_call|>` tags to fix blank LLM responses during external search. | [PR Link / Branch `feat/ollama-web-search-fix`](https://github.com/Paramveersingh-S/cookbook) |
| **#275** | **Kaggle TPU v5e-8 Data Parallelism:** Modernized distributed inference tutorials utilizing the Keras 3 `DataParallel` distribution API over 8-core TPU meshes, replacing outdated Flax references. | [PR Link / Branch `feat/tpu-v5e-parallelism`](https://github.com/Paramveersingh-S/cookbook) |
| **#274** | **FunctionGemma Dependency Conflicts:** Pinned `numpy` and `requests` bounds in the Colab fine-tuning notebooks to resolve pipeline failures with `google-colab 1.0.0`. | [PR Link / Branch `fix/functiongemma-notebook-deps`](https://github.com/Paramveersingh-S/cookbook) |

---

## 🔬 CERN Open Source Ecosystem
As part of the GSoC 2026 application (CERN-HSF umbrella), I have contributed to the CERN and scikit-hep ecosystems:

| Project | Target Issue | Description | Pull Request / Branch |
| :--- | :--- | :--- | :--- |
| **CERN/TIGRE** | #744 | Fixed NumPy 2.x compatibility breaks in projection geometries. | [PR Link / Branch `fix-geometry-numpy2-single-angle`](https://github.com/Paramveersingh-S/TIGRE) |
| **CERN/TIGRE** | #681 | Validated 2D inputs for `im_3d_denoise` to prevent tuple index out-of-range crashes. | [PR Link / Branch `fix-im_3d_denoise-2d`](https://github.com/Paramveersingh-S/TIGRE) |
| **CERN/CAiMIRA** | #8 | Stabilized flaky test suites by using seeded `np.random.RandomState`. | [PR Link / Branch `fix-flaky-quantities-test`](https://github.com/Paramveersingh-S/CAiMIRA) |
| **scikit-hep/awkward** | #13 | Added missing type hint for axis parameter in `ak.flatten` to resolve static typing errors. | [PR Link / Branch `fix-ak-flatten-typing`](https://github.com/Paramveersingh-S/awkward) |
| **scikit-hep/uproot5** | #8 | Added missing model mappings for `ROOT.TString` inside `TTree` in `uproot/models/TString.py`. | [PR Link / Branch `fix-tstring-mapping`](https://github.com/Paramveersingh-S/uproot5) |
| **scikit-hep/vector** | #654 | Fixed Pyright operator typing (`__add__`, `__sub__`, etc.) inside `VectorProtocol`. | [PR Link / Branch `fix-vectorprotocol-dunder`](https://github.com/Paramveersingh-S/vector) |
| **scikit-hep/pyhf** | #2718 | Updated image comparison tolerances to resolve matplotlib 3.11 breaking changes in tests. | [PR Link / Branch `fix-mpl-image-tolerance`](https://github.com/Paramveersingh-S/pyhf) |
| **HSF/hsf-training** | #43 | Updated `scikit-hep-webpage` tutorials to warn against the deprecated `root_numpy` package. | [PR Link / Branch `update-root-numpy-notice`](https://github.com/Paramveersingh-S/hsf-training-scikit-hep-webpage) |

---

## 🤗 Hugging Face Open Source Ecosystem
Contributions to the state-of-the-art Hugging Face ecosystem:

| Project | Target Issue | Description | Pull Request / Branch |
| :--- | :--- | :--- | :--- |
| **huggingface/datasets** | #5354 | Fixed Mypy invariance typing errors by updating `List` to `Sequence` in core dataset loading APIs. | [PR Link / Branch `main`](https://github.com/Paramveersingh-S/datasets) |
| **huggingface/diffusers** | #13903 | Built from scratch the custom modular blocks required to add `img2img` generation capabilities to the `Anima` modular pipeline. | [PR Link / Branch `main`](https://github.com/Paramveersingh-S/diffusers) |

---
*Note: The code for these contributions lives in their respective upstream repositories and my forks. This repository acts as an aggregator of those contributions.*
