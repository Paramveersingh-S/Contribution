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

## 🔬 CERN Open Source Ecosystem (Upcoming Contributions)
As part of the GSoC 2026 application (CERN-HSF umbrella), I will be contributing to the CERN and scikit-hep ecosystems:

| Project | Target Issue | Description |
| :--- | :--- | :--- |
| **CERN/TIGRE** | #744 | Fix NumPy 2.x compatibility breaks (`np.bool` deprecations, etc.) in projection geometries. |
| **CERN/TIGRE** | #681 | Validate 2D inputs for `im_3d_denoise` to prevent tuple index out-of-range crashes. |
| **CERN/TIGRE** | #678 | Fix geometry utility integer coercion by ensuring attributes like `nDetector` stay as ints. |
| **CERN/CAiMIRA** | #13 & #8 | Resolve OAuth social sign-in timeouts and stabilize flaky test suites. |
| **scikit-hep** | TBD | Broaden compatibility and resolve bugs for `awkward`, `uproot5`, `vector`, and `pyhf`. |

---
*Note: The code for these contributions lives in their respective upstream repositories and my forks. This repository acts as an aggregator of those contributions.*
