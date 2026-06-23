# CERN Open Source Contribution Guide
## A Complete Operating Manual for AI-Assisted Senior-Level Contributions

> **Purpose:** Any model reading this file should operate as a senior SDE contributor across
> CERN's open source ecosystem — identifying real issues, writing production-quality
> code, opening proper PRs, and building a recognized profile with CERN researchers.

---

## SECTION 0 — CONTEXT AND MISSION

CERN's open source footprint is split across two tiers:

**Tier 1 — Direct CERN GitHub Org** (`github.com/CERN`)
41 public repos, most actively maintained being TIGRE (781 stars, 87 open issues) and
CAiMIRA (airborne risk model). These have the CERN brand and highest recognition value.

**Tier 2 — CERN-Affiliated Ecosystem** (where most of the real physics software lives)
CERN's actual scientific computing runs through partner orgs that CERN funds, staffs,
and participates in via HSF (HEP Software Foundation) and IRIS-HEP:

| Org | Key repos | Why it matters |
|-----|-----------|---------------|
| `scikit-hep` | `awkward`, `uproot5`, `vector`, `pyhf`, `hist` | Used by ATLAS, CMS, LHCb — cited in hundreds of Nature/PRD papers |
| `root-project` | `root` | THE data format for all HEP experiments |
| `HSF` | `hsf.github.io`, training materials | GSoC 2026 umbrella org (CERN-HSF) |
| `CERN` | `TIGRE`, `CAiMIRA` | Direct CERN-branded repos |

**Your goal:** Build a contributor profile visible to CERN researchers and recruiters.
The fastest path is the scikit-hep ecosystem (high activity, friendly maintainers, used
daily in LHC analyses). TIGRE is your CERN-branded credential. CAiMIRA is the entry point.

---

## SECTION 1 — TARGET REPOS, PRIORITY ORDER

### Priority 1: CERN/TIGRE (Direct CERN brand, 781 stars, 87 open issues)
**What:** Tomographic Iterative GPU-based Reconstruction Toolbox. Python + CUDA for CT imaging.
**Stack:** Python, CUDA C++, NumPy, matplotlib
**Why:** Most active CERN repo with open issues. Medical imaging + physics = dual relevance.
**Issue URL:** `https://github.com/CERN/TIGRE/issues`
**Active open issues (as of June 2026):**
- #744: `single angle geometry issues with np 2.4` — NumPy 2.x compatibility break
- #681: `Issue with using 2D images in tigre.utilities.im_3d_denoise` — function crashes on 2D input
- #678: `Geometry utilities cast to single, casts integers to single too` — dtype casting bug
- #686: `ArbitrayAxisOfRotation` — unclear rotation axis behavior
- #679: `DBT Reconstruction — images are shifted` — geometric alignment bug

### Priority 2: CERN/CAiMIRA (Actively developed, most recent commits)
**What:** CERN Airborne Model for Indoor Risk Assessment (COVID-era tool, now generalized)
**Stack:** Python, TypeScript, FastAPI, Vue.js, Monte Carlo simulations
**Why:** Most recently active CERN repo (updated June 22, 2026). Python + web stack.
**Active issues:**
- #13: `social sign in timing out` — OAuth/SSO authentication bug (June 2026)
- #8: `Flaky Tests` — test reliability issue, open since 2023

### Priority 3: scikit-hep/awkward (957 stars, actively maintained, CERN-affiliated)
**What:** Manipulate jagged/irregular arrays with NumPy-like idioms. Used in every LHC experiment.
**Stack:** Python, C++, pybind11, NumPy, Numba, JAX
**Why:** CERN/ATLAS/CMS/LHCb use this daily. Being a contributor makes you visible to HEP researchers.
**Issue URL:** `https://github.com/scikit-hep/awkward/issues`

### Priority 4: scikit-hep/uproot5 (ROOT file I/O in pure Python)
**What:** Read/write ROOT files — the data format of ALL LHC experiments — in pure Python.
**Stack:** Python, NumPy, fsspec
**Why:** Every physicist at CERN who uses Python touches this library.
**Issue URL:** `https://github.com/scikit-hep/uproot5/issues`

### Priority 5: scikit-hep/vector (Lorentz vectors for HEP, published in JOSS 2025)
**What:** JIT-compilable mathematical manipulation of physics vectors (momentum, spacetime).
**Stack:** Python, JAX, Numba, SymPy, NumPy
**Why:** Published in Journal of Open Source Software (2025). Actively maintained. JAX backend
is new and has room for contributions.
**Issue URL:** `https://github.com/scikit-hep/vector/issues`

### Priority 6: scikit-hep/pyhf (Statistical fitting for HEP, 299 stars)
**What:** Pure-Python HistFactory — the statistical model used by ATLAS for Higgs discovery.
**Stack:** Python, JAX, NumPy, PyTorch, TensorFlow
**Why:** Used in Nobel Prize-level physics. Maintainers are extremely welcoming.
**Issue URL:** `https://github.com/scikit-hep/pyhf/issues`

### Priority 7: HSF/hsf-training-* (CERN-HSF training materials)
**What:** Training notebooks and documentation for HEP software used at CERN workshops.
**Why:** Easiest entry point. Directly linked to GSoC 2026 via CERN-HSF umbrella.
**Contribution:** Fix broken notebook cells, update for new package versions, add exercises.

---

## SECTION 2 — ENVIRONMENT SETUP

### 2.1 Universal Prerequisites

```bash
# Python 3.10+ required for all repos
python --version  # must be >= 3.10

# Essential HEP toolchain
pip install numpy scipy matplotlib uproot awkward vector pyhf hist

# For TIGRE (GPU/CUDA required for full functionality)
# CPU-only testing is possible without CUDA for Python logic
pip install tigre  # from PyPI, or install from source

# For CAiMIRA
pip install caimira
```

### 2.2 TIGRE Setup

```bash
git clone https://github.com/YOUR_USERNAME/TIGRE.git
cd TIGRE
git remote add upstream https://github.com/CERN/TIGRE.git

# Python-only install (no CUDA required for Python layer testing)
pip install -e ".[dev]"

# Verify install
python -c "import tigre; print('TIGRE OK')"

# Run the Python tests (no GPU needed)
python -m pytest Python/tests/ -v

# If you have CUDA (optional but ideal):
# Install CUDA toolkit first, then:
pip install tigre  # rebuilds with CUDA
# or from source with CUDA:
python setup.py install
```

**TIGRE directory structure:**
```
TIGRE/
  Python/
    tigre/
      algorithms/        # Iterative reconstruction algorithms (your main target)
        _CTrecon.py      # Base class
        SART.py          # SART family
        Krylov.py        # Krylov subspace methods
        TV_algorithms.py # Total variation regularization
      utilities/
        geometry_default.py  # Geometry definitions
        im_3d_denoise.py     # Denoising utilities (issue #681 is here)
        Measure.py           # Quality metrics
      projectors/            # CUDA projection/backprojection (C++/CUDA)
    tests/               # Python-level tests
    demos/               # Example scripts
  MATLAB/                # MATLAB version (parallel implementation)
  Source/                # CUDA C++ source
  contributing.md        # ALWAYS READ THIS FIRST
```

### 2.3 CAiMIRA Setup

```bash
git clone https://github.com/YOUR_USERNAME/CAiMIRA.git
cd CAiMIRA
git remote add upstream https://github.com/CERN/CAiMIRA.git

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest caimira/tests/ -v

# Run the application locally
python -m caimira.apps.calculator
```

**CAiMIRA directory structure:**
```
CAiMIRA/
  caimira/
    models/           # Physics models (aerosol, virus emission)
    apps/
      calculator/     # FastAPI backend + Vue.js frontend
    tests/            # Tests
  docs/
```

### 2.4 scikit-hep/awkward Setup

```bash
git clone https://github.com/YOUR_USERNAME/awkward.git
cd awkward
git remote add upstream https://github.com/scikit-hep/awkward.git

# Install in editable mode (builds C++ extension)
pip install -e ".[dev]"

# This compiles pybind11 C++ — takes ~5 minutes first time
# Requires: cmake, C++ compiler (g++ or clang++)

# Run tests
python -m pytest tests/ -x -q

# Run tests for specific module
python -m pytest tests/v2/test_2369-concatenate-axis-none.py -v
```

**awkward directory structure:**
```
awkward/
  src/awkward/
    _connect/         # Connections to other libraries (JAX, ROOT, Numba)
    contents/         # Array content types (ListArray, RecordArray, etc.)
    operations/       # User-facing operations (ak.sum, ak.flatten, etc.)
    _reducers.py      # Reduction operations
  tests/              # Test files, named by issue number (e.g., test_2369-*.py)
  src/cpu-kernels/    # C++ CPU kernels
  src/cuda-kernels/   # CUDA GPU kernels
```

**KEY CONVENTION:** Tests are named `test_{issue_number}-{description}.py`. When you fix
issue #1234, your test file is `tests/test_1234-description.py`. This is mandatory.

### 2.5 scikit-hep/uproot5 Setup

```bash
git clone https://github.com/YOUR_USERNAME/uproot5.git
cd uproot5
git remote add upstream https://github.com/scikit-hep/uproot5.git
pip install -e ".[dev]"

# Download sample ROOT files for testing
python -c "import skhep_testdata; skhep_testdata.data('uproot-HZZ.root')"

# Run tests
python -m pytest tests/ -x -q
```

### 2.6 scikit-hep/vector Setup

```bash
git clone https://github.com/YOUR_USERNAME/vector.git
cd vector
git remote add upstream https://github.com/scikit-hep/vector.git
pip install -e ".[dev]"
pip install nox  # scikit-hep uses nox for task automation
nox -s tests    # run full test suite
```

### 2.7 scikit-hep/pyhf Setup

```bash
git clone https://github.com/YOUR_USERNAME/pyhf.git
cd pyhf
git remote add upstream https://github.com/scikit-hep/pyhf.git
pip install -e ".[dev,jax,torch]"
python -m pytest tests/ -x -q
```

---

## SECTION 3 — THE PHYSICS DOMAIN KNOWLEDGE YOU NEED

Unlike DeepMind repos where ML knowledge is sufficient, CERN repos require understanding
the physics domain. This section gives you the minimum viable physics context.

### 3.1 HEP Data Basics (for awkward, uproot, vector, pyhf)

**ROOT files** are the universal data format for particle physics experiments. Every event
(collision) at the LHC produces particles, and their properties are stored as arrays
in ROOT files. The data is "jagged" because different events have different numbers
of particles.

```python
import uproot
import awkward as ak

# Open a ROOT file from LHC data
f = uproot.open("https://scikit-hep.org/uproot3/examples/HZZ.root")
events = f["events"]

# Read particle data — this is "jagged" because each event has different # of muons
muon_pt = events["NJet"].array()  # transverse momentum of jets

# awkward handles this natively
jets = ak.zip({
    "pt": events["Jet_Pt"].array(),
    "eta": events["Jet_Eta"].array(),
    "phi": events["Jet_Phi"].array(),
})

# Select events with > 2 jets
selected = jets[ak.num(jets) > 2]
```

**Lorentz vectors** are 4-vectors describing particle momentum + energy:
- Components: (px, py, pz, E) or (pT, η, φ, mass)
- pT (transverse momentum): momentum perpendicular to beam axis
- η (pseudorapidity): -ln(tan(θ/2)), a rapidity-like variable
- φ (azimuthal angle): angle around the beam axis
- Invariant mass: m² = E² - |p|² (this is what you reconstruct for Higgs, Z, etc.)

```python
import vector

# Create Lorentz vectors
muons = vector.array({
    "pt": [20.5, 38.2],
    "eta": [1.2, -0.8],
    "phi": [0.5, 3.1],
    "mass": [0.106, 0.106],  # muon mass in GeV
})

# Compute invariant mass of the pair (should be ~91 GeV for Z boson)
z_candidate = muons[0] + muons[1]
print(z_candidate.mass)  # ~91 GeV if it's a real Z
```

**HistFactory / pyhf** is the statistical framework used to discover the Higgs boson:
- A "workspace" defines signal, background, and systematic uncertainties
- `CLs` (confidence level) is used to set exclusion limits
- pyhf makes this available in pure Python with JAX autodiff

### 3.2 CT Reconstruction Basics (for TIGRE)

TIGRE reconstructs 3D images from X-ray projections. The key concepts:

**Geometry:** The scanner geometry defines where the source and detector are relative to
the object being imaged. TIGRE uses a flexible cone-beam geometry.

```python
import tigre
import numpy as np

# Define scan geometry
geo = tigre.geometry(mode='cone', default=True, nVoxel=np.array([256, 256, 256]))
geo.DSD = 1536  # Distance Source-Detector (mm)
geo.DSO = 1000  # Distance Source-Origin (mm)
geo.nDetector = np.array([512, 512])  # Detector pixels

# Define projection angles
angles = np.linspace(0, 2*np.pi, 360, dtype=np.float32)

# Generate projections (forward project)
projections = tigre.Ax(phantom, geo, angles)

# Reconstruct (backward project)
reconstruction = tigre.algorithms.FDK(projections, geo, angles)
```

**Iterative algorithms:** Instead of FDK (simple backprojection), iterative algorithms
minimize a cost function to get better quality reconstructions:
- SART: Simultaneous Algebraic Reconstruction Technique
- FISTA: Fast Iterative Shrinkage-Thresholding (TV regularization)
- ASD-POCS: Adaptive Steepest Descent - Projection Onto Convex Sets

**NumPy 2.x compatibility** (issue #744): NumPy 2.0 changed several dtype behaviors.
`np.single` is now stricter, `np.bool` is deprecated, and some casting rules changed.
Issue #744 is about single-angle projections breaking with NumPy 2.4.

### 3.3 Airborne Risk Modeling (for CAiMIRA)

CAiMIRA models viral aerosol transmission in indoor spaces. Key concepts:

- **Viral emission rate:** How much virus an infected person breathes out
- **Ventilation:** How quickly virus-laden air is replaced with fresh air
- **Monte Carlo simulation:** Uncertainty quantification via random sampling

---

## SECTION 4 — ISSUE FINDING AND TRIAGE

### 4.1 Where to Find Issues

```
# TIGRE — 87 open issues, very active
https://github.com/CERN/TIGRE/issues?q=is:open&sort=updated

# CAiMIRA
https://github.com/CERN/CAiMIRA/issues?q=is:open

# awkward
https://github.com/scikit-hep/awkward/issues?q=is:open+label:"help+wanted"

# uproot5
https://github.com/scikit-hep/uproot5/issues?q=is:open+label:"good+first+issue"

# vector
https://github.com/scikit-hep/vector/issues?q=is:open

# pyhf
https://github.com/scikit-hep/pyhf/issues?q=is:open+label:"good+first+issue"
```

### 4.2 Issue Triage Protocol

**Before claiming any issue, do all of these:**

1. Read the full issue thread including every comment
2. Check if there's already an open PR (look at the PR tab, filter by the issue number)
3. Try to reproduce the bug locally
4. If it's a NumPy/dependency compatibility issue, check whether it's already fixed
   in a newer version of the dependency
5. Comment on the issue before starting: "I'd like to work on this. I've reproduced it
   locally — [paste the error or confirm it works]. My plan is [one sentence]. Is there
   anything I should know before starting?"

### 4.3 Specific High-Value Issues to Target Now

**TIGRE #744 (numpy 2.x compatibility, single angle geometry)**
This is likely a `np.bool` → `bool` deprecation or dtype casting change in NumPy 2.x.
Fix approach:
```python
# WRONG (NumPy 2.x breaks this)
if isinstance(x, np.bool):  # np.bool deprecated

# CORRECT
if isinstance(x, (bool, np.bool_)):

# WRONG
arr = arr.astype(np.single)  # may cast incorrectly in np 2.x

# CORRECT — be explicit
arr = arr.astype(np.float32)
```

**TIGRE #681 (im_3d_denoise crashes on 2D input)**
The function expects 3D input but gets 2D. Fix: add a guard at the top of the function
that either raises a clear error message or extends 2D to 3D:
```python
def im_3d_denoise(img, ...):
    if img.ndim == 2:
        raise ValueError(
            f"im_3d_denoise requires a 3D image, but got shape {img.shape}. "
            "Use im_denoise for 2D images."
        )
    # or: img = img[np.newaxis, ...]  # promote to 3D
```

**TIGRE #678 (geometry utilities cast integers to single)**
Integer fields like `nDetector` or `nVoxel` should NOT be cast to float32.
Fix: in the `cast_to_single` utility function, add a type check:
```python
def cast_to_single(geo):
    for attr in float_attributes:  # only cast known float fields
        setattr(geo, attr, np.float32(getattr(geo, attr)))
    # Do NOT cast integer attributes (nDetector, nVoxel, etc.)
    return geo
```

**CAiMIRA #8 (flaky tests)**
Flaky tests are usually due to: random seeds not being set, time-dependent behavior,
or ordering assumptions. Fix: find which test is flaky, add `np.random.seed(42)` or
use `pytest-randomly` with a fixed seed, or mock the time-dependent call.

**scikit-hep/vector — JAX backend gaps**
The SymPy backend was recently added (CHEP 2024 paper). Look for missing operations
in the JAX backend by comparing `vector/_backends/jax.py` with `numpy.py`.
Any operation in numpy.py that isn't in jax.py is a potential contribution.

---

## SECTION 5 — THE CONTRIBUTION WORKFLOW (END TO END)

### The Flow

```
UNDERSTAND → REPRODUCE → PLAN → DISCUSS → IMPLEMENT → TEST → DOCUMENT → PR → REVIEW
```

### Phase 1: UNDERSTAND (2 hours minimum)

```bash
# Read the contributing guide FIRST
cat CONTRIBUTING.md
# or
cat contributing.md  # TIGRE uses lowercase

# Read the file you're going to touch
cat Python/tigre/utilities/im_3d_denoise.py

# Read its tests
cat Python/tests/test_denoise.py

# Read git history for context on this file
git log --oneline Python/tigre/utilities/im_3d_denoise.py

# Find how similar things are handled elsewhere in the codebase
grep -r "ndim" Python/tigre/ --include="*.py"
grep -r "np.single\|float32\|dtype" Python/tigre/ --include="*.py" | head -20
```

### Phase 2: REPRODUCE

```python
# Write a minimal reproduction script BEFORE touching any code
# Save as repro.py and run it to confirm you see the bug

import tigre
import numpy as np

# Issue #681: 2D image crashes im_3d_denoise
img_2d = np.random.rand(256, 256).astype(np.float32)
try:
    result = tigre.utilities.im_3d_denoise(img_2d, iters=5)
    print("No error — bug may be fixed")
except Exception as e:
    print(f"Reproduced: {type(e).__name__}: {e}")
    # Now you know what's happening and can fix it
```

### Phase 3: PLAN

Write (for yourself, before any code):
1. Exact file(s) to change
2. Exact lines/functions to modify
3. New test to write
4. Whether any docs need updating
5. Whether this is a breaking change

If the change touches more than 3 files, it's probably two PRs. Split it.

### Phase 4: DISCUSS

Post on the issue before coding. Template:

```
**I'd like to work on this.**

I've reproduced the bug on:
- Python 3.11.9
- TIGRE 3.x (installed from source, commit abc1234)
- NumPy 2.4.0

**Reproduction:**
```python
import tigre, numpy as np
img = np.random.rand(256, 256).astype(np.float32)
tigre.utilities.im_3d_denoise(img, iters=5)
# → IndexError: tuple index out of range
```

**My proposed fix:**
Add a dimension check at the top of `im_3d_denoise` that raises a `ValueError`
with a clear message if the input is not 3D.

**Files:** `Python/tigre/utilities/im_3d_denoise.py`, `Python/tests/test_denoise.py`

Happy to proceed unless there's a preferred approach. Let me know!
```

### Phase 5: IMPLEMENT

**Style rules for each repo:**

**TIGRE (Python layer):**
```python
# Use numpy explicitly, not np aliases for types
arr = arr.astype(np.float32)   # CORRECT
arr = arr.astype(np.single)    # AVOID — single is deprecated in NumPy 2.x

# Error messages should be descriptive
raise ValueError(
    f"Expected a 3D array (nZ, nY, nX), got shape {img.shape}. "
    f"For 2D denoising, use tigre.utilities.im_2d_denoise()."
)

# Type annotations (optional but appreciated)
def im_3d_denoise(img: np.ndarray, iters: int = 50, ...) -> np.ndarray:
```

**scikit-hep/awkward:**
```python
# awkward uses a "builder" pattern for creating arrays
import awkward as ak

# CORRECT — use ak operations, never Python loops on arrays
result = ak.sum(array, axis=1)

# WRONG — never iterate element by element
result = [sum(event) for event in array]  # defeats the purpose

# Type annotations are required for new public functions
def my_function(array: ak.Array, axis: int = None) -> ak.Array:
    """Short summary.

    Args:
        array: Input array.
        axis: Axis along which to operate. None means all axes.

    Returns:
        Result array.

    Examples:
        >>> import awkward as ak
        >>> array = ak.Array([[1, 2], [3]])
        >>> my_function(array, axis=1)
        <Array [3, 3] type='2 * int64'>
    """
```

**scikit-hep/pyhf:**
```python
# pyhf uses a backend system — write backend-agnostic code
import pyhf

# WRONG — NumPy-specific
import numpy as np
result = np.sum(array)

# CORRECT — uses whatever backend is set (NumPy, JAX, PyTorch, TensorFlow)
tensorlib = pyhf.tensorlib
result = tensorlib.sum(array, axis=-1)

# Type annotations with pyhf types
from pyhf.typing import PathOrStr
def my_function(workspace_path: PathOrStr) -> pyhf.Workspace:
    ...
```

**scikit-hep/vector:**
```python
# vector uses dispatch — the same method works on all backends
import vector

# Every backend (NumPy, Awkward, Uproot, SymPy, JAX) shares the same interface
# Implement new methods in: vector/src/vector/compute/
# Then register them in each backend

# New coordinate computation goes in:
# vector/src/vector/compute/lorentz/my_quantity.py

def dispatch(v):
    """Compute my_quantity from a Lorentz vector."""
    # Extract needed components
    px, py, pz, energy = v.px, v.py, v.pz, v.energy
    return (px**2 + py**2 + pz**2) / energy  # example
```

### Phase 6: TEST

**TIGRE tests:**
```python
# File: Python/tests/test_denoise_2d_guard.py

import pytest
import numpy as np
from tigre.utilities import im_3d_denoise

class TestIm3dDenoiseInputValidation:

    def test_3d_input_works(self):
        """Confirm 3D input runs without error."""
        img = np.random.rand(64, 64, 64).astype(np.float32)
        result = im_3d_denoise(img, iters=2)
        assert result.shape == img.shape

    def test_2d_input_raises_value_error(self):
        """Confirm 2D input raises a clear ValueError (issue #681)."""
        img = np.random.rand(256, 256).astype(np.float32)
        with pytest.raises(ValueError, match="3D"):
            im_3d_denoise(img, iters=5)

    def test_1d_input_raises_value_error(self):
        """Confirm 1D input also raises ValueError."""
        img = np.random.rand(256).astype(np.float32)
        with pytest.raises(ValueError):
            im_3d_denoise(img, iters=5)

    def test_output_dtype_preserved(self):
        """Output dtype should match input dtype."""
        img = np.random.rand(32, 32, 32).astype(np.float32)
        result = im_3d_denoise(img, iters=2)
        assert result.dtype == np.float32
```

**awkward tests (naming convention is critical):**
```python
# File: tests/test_2745-my-feature-description.py
# (The number must match the issue you're fixing)

import awkward as ak
import numpy as np
import pytest

def test_my_fix_basic():
    """Test the basic case from issue #2745."""
    array = ak.Array([[1, 2, 3], [4, 5]])
    result = ak.my_new_function(array)
    assert ak.to_list(result) == [[1, 4], [2, 5], [3]]  # expected output

def test_my_fix_empty():
    """Edge case: empty array."""
    array = ak.Array([[], []])
    result = ak.my_new_function(array)
    assert len(result) == 0

def test_my_fix_type_preserved():
    """Output type should be consistent."""
    array = ak.Array([[1, 2], [3]])
    result = ak.my_new_function(array)
    assert result.type == ak.types.ArrayType(...)
```

**pyhf tests:**
```python
# File: tests/test_myfeature.py
import pyhf
import pytest
import numpy as np

@pytest.fixture
def simple_model():
    return pyhf.simplemodels.uncorrelated_background(
        signal=[12.0, 11.0], bkg=[50.0, 52.0], bkg_uncertainty=[3.0, 7.0]
    )

def test_my_feature_numpy(simple_model):
    """Test with NumPy backend."""
    pyhf.set_backend("numpy")
    result = my_new_function(simple_model)
    assert result is not None

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_my_feature_backends(simple_model, backend):
    """Test works across all backends."""
    pyhf.set_backend(backend)
    result = my_new_function(simple_model)
    assert abs(result - expected_value) < 1e-5
```

**Run tests before every PR:**
```bash
# TIGRE
python -m pytest Python/tests/ -x -v

# awkward
python -m pytest tests/ -x -q

# pyhf
python -m pytest tests/ -x -q --tb=short

# vector
nox -s tests
```

### Phase 7: DOCUMENT

For TIGRE:
- Update the function's docstring
- If it's a new algorithm, add a reference to the paper in the docstring
- Update `CHANGELOG.md` if one exists (check `git log --oneline CHANGELOG.md`)

For scikit-hep repos:
- Docstrings use NumPy docstring style
- New public functions must have an `Examples:` section with runnable code
- Update `CHANGELOG.md` or `CHANGES.rst` (these repos use `towncrier` for changelogs)

```bash
# For scikit-hep repos, create a changelog fragment:
# File: changelog/{issue_number}.bugfix.rst (or .feature.rst, .maintenance.rst)
# Content: One sentence describing the change.

echo "Fixed crash when 2D image passed to :func:`tigre.utilities.im_3d_denoise` (issue #681)." \
  > changelog/681.bugfix.rst
```

### Phase 8: PR SUBMISSION

```bash
# Sync with upstream
git fetch upstream
git rebase upstream/master  # TIGRE uses master, not main
# or
git rebase upstream/main   # scikit-hep repos use main

# Run full test suite
python -m pytest tests/ -x -q

# Stage your changes carefully
git add Python/tigre/utilities/im_3d_denoise.py
git add Python/tests/test_denoise_2d_guard.py
# DO NOT: git add .

# Commit message format:
git commit -m "fix: raise ValueError when im_3d_denoise receives 2D input

Previously, passing a 2D array to im_3d_denoise would cause an
IndexError deep in the function, giving no clear indication of
what went wrong.

Now, a descriptive ValueError is raised immediately, pointing the
user toward im_2d_denoise for 2D inputs.

Fixes #681"

git push origin fix/im-3d-denoise-2d-input-guard
```

**PR Template:**

```markdown
## Summary

Fixes a crash in `im_3d_denoise` when a 2D image is passed as input.

## Problem

When users pass a 2D array instead of 3D, the function fails with an
`IndexError: tuple index out of range` deep inside the algorithm,
with no indication of what went wrong.

**Reproducible with:**
```python
import tigre, numpy as np
img = np.random.rand(256, 256).astype(np.float32)
tigre.utilities.im_3d_denoise(img, iters=5)
# → IndexError: tuple index out of range
```

## Solution

Added an explicit `ndim` check at the entry point of `im_3d_denoise`.
If a non-3D array is received, a `ValueError` is raised with a clear
message telling the user to use `im_2d_denoise` for 2D inputs.

## Changes

- `Python/tigre/utilities/im_3d_denoise.py`: Added dimension guard
- `Python/tests/test_denoise_2d_guard.py`: Added 4 test cases

## Checklist

- [x] Tests pass locally (`pytest Python/tests/ -v`)
- [x] New tests added for the bug
- [x] Docstring updated
- [x] No unrelated changes included

Closes #681
```

---

## SECTION 6 — REPO-SPECIFIC DEEP DIVES

### 6.1 CERN/TIGRE — Complete Contribution Guide

**Algorithms directory** is where new CT reconstruction algorithms live:

```python
# File: Python/tigre/algorithms/MyAlgorithm.py

import numpy as np
from tigre.algorithms._CTrecon import IterativeReconAlg

class MyAlgorithm(IterativeReconAlg):
    """My new CT reconstruction algorithm.

    Implements [Algorithm Name] from [Author et al., Year].

    Parameters
    ----------
    proj : np.ndarray, shape (num_angles, nDetector[0], nDetector[1])
        The measured projections.
    geo : tigre.geometry
        The scanner geometry.
    angles : np.ndarray, shape (num_angles,)
        Projection angles in radians.
    niter : int
        Number of iterations.
    **kwargs
        Additional keyword arguments passed to the base class.

    References
    ----------
    .. [1] Author, A., et al. "Title." Journal (Year). DOI.

    Examples
    --------
    >>> import tigre
    >>> import tigre.algorithms as algs
    >>> import numpy as np
    >>> geo = tigre.geometry(mode='cone', default=True)
    >>> angles = np.linspace(0, 2*np.pi, 100, dtype=np.float32)
    >>> proj = tigre.Ax(phantom, geo, angles)
    >>> recon = algs.MyAlgorithm(proj, geo, angles, niter=50)
    """

    def __init__(self, proj, geo, angles, niter, **kwargs):
        # Call parent __init__ — this handles parameter validation
        IterativeReconAlg.__init__(self, proj, geo, angles, niter, **kwargs)

    def run_main_iter(self):
        """The core iteration loop. Override this in your algorithm."""
        for i in range(self.niter):
            # Forward project current estimate
            proj_est = tigre.Ax(self.res, self.geo, self.angles)

            # Compute residual
            residual = self.proj - proj_est

            # Update (algorithm-specific)
            update = tigre.Atb(residual, self.geo, self.angles) * self.lmbda

            # Apply update
            self.res += update

            # TV regularization (if applicable)
            if hasattr(self, 'alpha'):
                self.res = self.apply_tv(self.res)

    # Export to Python module
algs.MyAlgorithm = MyAlgorithm
```

**After implementing, register in `__init__.py`:**
```python
# Python/tigre/algorithms/__init__.py
from tigre.algorithms.MyAlgorithm import MyAlgorithm
```

**NumPy 2.x porting checklist (for issue #744 and similar):**
```python
# Check for these patterns that break in NumPy 2.x:

# 1. np.bool deprecated → use bool or np.bool_
np.bool  # BREAKS in 2.x
np.bool_  # OK

# 2. np.int deprecated → use int or np.int_
np.int   # BREAKS in 2.x
np.int_  # OK

# 3. np.float deprecated → use float or np.float_
np.float  # BREAKS in 2.x
np.float64  # OK (explicit is better)

# 4. Implicit array creation from Python scalars changed
# 5. String-based dtype specs changed

# Search for all deprecated usages:
grep -r "np\.bool\b\|np\.int\b\|np\.float\b\|np\.complex\b" Python/ --include="*.py"
```

### 6.2 scikit-hep/awkward — Complete Contribution Guide

**How awkward is structured (critical to understand):**

```
Layer 1: Python API (ak.sum, ak.flatten, etc.) → src/awkward/operations/
Layer 2: Array types (ListArray, RecordArray) → src/awkward/contents/
Layer 3: C++ kernels (CPU) → src/cpu-kernels/
Layer 4: CUDA kernels (GPU) → src/cuda-kernels/
Layer 5: Connections → src/awkward/_connect/ (JAX, ROOT, Numba integrations)
```

**For a typical Python-layer contribution** (fixing a bug in an ak.* function):

```python
# File: src/awkward/operations/ak_my_function.py

import awkward as ak
from awkward._dispatch import named_unary_dispatch
from awkward._nplikes.numpy_like import NPLike

def my_function(array, *, highlevel=True, behavior=None, attrs=None):
    """
    Short description of what this function does.

    Longer description with physics context if relevant.

    Parameters
    ----------
    array : array_like
        Input array (any awkward array type).
    highlevel : bool
        If True (default), return a high-level :class:`ak.Array`.
        If False, return a low-level layout.
    behavior : mapping or None
        Override the array's behavior.
    attrs : mapping or None
        Override the array's attributes.

    Returns
    -------
    ak.Array

    Examples
    --------
    >>> import awkward as ak
    >>> array = ak.Array([[1, 2, 3], [4, 5]])
    >>> ak.my_function(array)
    <Array [[...], [...]] type='...'>
    """
    with ak._nplikes.nplike_of(array, default=ak._nplikes.Numpy):
        return _my_function(
            ak.to_layout(array),
            highlevel=highlevel,
            behavior=behavior,
            attrs=attrs,
        )
```

**Running specific test types:**
```bash
# Run tests matching a pattern
python -m pytest tests/ -k "test_flatten" -v

# Run tests with coverage
python -m pytest tests/ --cov=src/awkward --cov-report=html

# Run doctests
python -m pytest --doctest-modules src/awkward/operations/ak_my_function.py
```

### 6.3 scikit-hep/pyhf — Complete Contribution Guide

**pyhf's backend system — the most important architectural concept:**

```python
# pyhf supports 4 backends: numpy, jax, pytorch, tensorflow
# Your code MUST work with all of them or explicitly restrict to specific ones

import pyhf

# Getting the active tensorlib
tb = pyhf.tensorlib

# Tensor operations (use these, never numpy directly):
tb.sum(x, axis=-1)          # not np.sum
tb.stack([x, y], axis=0)    # not np.stack
tb.ones(shape)               # not np.ones
tb.abs(x)                    # not np.abs

# Shape operations
tb.shape(x)                  # not x.shape (JAX returns a tuple, not an ndarray)
tb.tolist(x)                 # convert to Python list

# Constraint pdfs
pyhf.probability.Normal(mu, sigma).log_prob(x)
pyhf.probability.Poisson(rate).log_prob(k)
```

**Common pyhf contribution types:**
- New modifier types (nuisance parameters) in `src/pyhf/modifiers/`
- New optimizer integrations in `src/pyhf/optimize/`
- New backend operations in `src/pyhf/tensor/`
- Documentation improvements in `docs/`
- Validation against reference HistFactory workspaces

---

## SECTION 7 — HOW TO FIND CERN-VISIBILITY CONTRIBUTIONS

Not all contributions are equal in terms of profile building. This section tells you
which types of contributions get you noticed by CERN researchers.

### High Visibility (CERN researchers will see your name)

1. **Fixing a bug that users have been complaining about** — TIGRE #744 (NumPy 2.4 break)
   is a perfect example. Lots of users hit it. Your name appears in the git log forever.

2. **Adding a new reconstruction algorithm to TIGRE** — Physics people care about this.
   Find a recent CT paper (arXiv, IEEE TMI, PMB), implement the algorithm, and cite it.
   Your contribution will be cited if they use it in their work.

3. **Fixing awkward/uproot compatibility issues** — When LHC analyses break because of
   an uproot bug, hundreds of physicists are affected. Fixing it makes you a hero.

4. **Contributing to HSF training materials** — CERN runs workshops where hundreds of
   physicists learn these tools. Your contribution literally teaches CERN scientists.

5. **GSoC 2026 proposal** — CERN-HSF is an umbrella org for GSoC 2026. The application
   period opens February 19. Start now by contributing to one of their projects and
   contacting mentors AFTER February 19 (explicitly required by their rules).

### Medium Visibility (Maintainers notice)

- Documentation improvements
- Test coverage improvements
- Dependency compatibility fixes

### Lower Visibility (But still builds the record)
- Typo fixes
- Minor refactors with no user impact

---

## SECTION 8 — WRITING ISSUES LIKE A SENIOR SDE

### Bug Report Template for TIGRE/scikit-hep

```markdown
## Environment

- Package version: (run `python -c "import tigre; print(tigre.__version__)"`)
- NumPy version: (run `python -c "import numpy; print(numpy.__version__)"`)
- Python version:
- OS (and CUDA version if relevant):

## Description

One clear sentence: what is wrong?

## Minimal Reproduction

```python
# This is the minimal code that reproduces the bug
import tigre
import numpy as np

geo = tigre.geometry(mode='cone', default=True)
img = np.random.rand(256, 256).astype(np.float32)  # 2D input
result = tigre.utilities.im_3d_denoise(img)
# → IndexError: tuple index out of range
```

## Expected Behavior

What should happen? (e.g., "A clear ValueError explaining the input must be 3D")

## Actual Behavior

What actually happens? Paste the full traceback.

## Additional Context

Any context about why this matters (e.g., "This breaks our medical CT pipeline when
processing scout images, which are 2D slices.")
```

### Feature Request Template

```markdown
## Feature Request: [Algorithm/Feature Name]

## Summary

What feature are you proposing and why?

## Physics Motivation

(For HEP/CT repos) What is the physics application?
For TIGRE: what paper does the algorithm come from?
For scikit-hep: what LHC analysis workflow needs this?

## Proposed API

```python
# What would the new API look like?
result = tigre.algorithms.MyNewAlgorithm(proj, geo, angles, niter=50)
# or
result = ak.new_operation(array, axis=1)
```

## Reference

For algorithms: cite the paper.
For HEP operations: cite an LHC analysis that needs this.
```

---

## SECTION 9 — THE CERN-HSF GSOC 2026 PATHWAY

This is the highest-recognition path and it's achievable if you start now.

**Timeline:**
- Now → Feb 18: Contribute to scikit-hep and TIGRE repos. Build your record.
- Feb 19, 2026: HSF accepted organizations announced. Begin contacting mentors.
- Mar 16, 2026: GSoC application period opens.
- Mid-April 2026: Final selections announced.

**What makes a strong CERN-HSF proposal:**
1. Evidence of existing contributions (PRs merged to target repos)
2. Understanding of the physics motivation
3. Technical feasibility (your proposal must be realistic in 3 months)
4. A clear week-by-week timeline
5. Familiarity with the codebase (demonstrated by your prior contributions)

**Target projects for GSoC 2026 (historically recurring):**
- ROOT RDataFrame improvements
- awkward Array GPU acceleration
- uproot write performance
- pyhf new backends
- TIGRE new algorithms and Python modernization

**How to find mentors:** Look at the last 3 years of git commits in the repo. The
same handful of people (Jim Pivarski for awkward/uproot, Matthew Feickert for pyhf,
Ander Biguri for TIGRE) are the mentors. Interact with them through issue comments
and PRs — not cold emails — before February 19.

---

## SECTION 10 — QUICK REFERENCE: KEY PEOPLE AND CONTACTS

| Repo | Key maintainer(s) | How to interact |
|------|------------------|----------------|
| TIGRE | Ander Biguri | Issue comments, PR reviews |
| awkward | Jim Pivarski, Angus Hollands | Issue comments, Discussions tab |
| uproot5 | Jim Pivarski | Issue comments |
| vector | Saransh Chopra, Henry Schreiner | Issue comments |
| pyhf | Matthew Feickert, Lukas Heinrich, Giordon Stark | Issue comments, pyhf Gitter |
| HSF/GSoC | hsf-gsoc-admin@googlegroups.com | Email ONLY after Feb 19 |

**Communication norms in the HEP community:**
- Discourse over quick fixes — maintainers appreciate understanding of the physics
- Reference papers by arXiv number (e.g., "as described in arXiv:2302.12022")
- Use proper terminology: "pseudorapidity" not "some angle variable"
- Mattermost is used internally at CERN; GitHub Discussions/Issues are public

---

## SECTION 11 — HEP PHYSICS GLOSSARY (FOR CODE CONTEXT)

| Term | Meaning | Code relevance |
|------|---------|----------------|
| ROOT file | Binary data format for LHC data | uproot reads/writes these |
| TTree | Table-like data structure in ROOT | `uproot.open("f.root")["MyTree"]` |
| TBranch | A column in a TTree | `tree["branch_name"].array()` |
| jagged array | Array where sub-arrays have different lengths | core concept of awkward |
| Lorentz vector | 4-vector: (px, py, pz, E) or (pT, η, φ, m) | vector library handles this |
| pT | Transverse momentum (GeV/c) | `vector.array({"pt": [...]})` |
| η (eta) | Pseudorapidity = -ln(tan(θ/2)) | `v.eta` in vector |
| φ (phi) | Azimuthal angle | `v.phi` in vector |
| invariant mass | √(E²-|p|²), conserved quantity | `v1 + v2).mass` |
| HistFactory | Statistical model for histograms | pyhf implements this |
| CLs | Modified frequentist p-value | `pyhf.infer.hypotest()` |
| FDK | Feldkamp-Davis-Kress, fast CT algorithm | `tigre.algorithms.FDK` |
| SART | Simultaneous Algebraic Reconstruction | `tigre.algorithms.SART` |
| CBCT | Cone-Beam CT | TIGRE's primary geometry |
| Shepp-Logan | Standard CT test phantom | `tigre.utilities.phantoms.headphantom()` |

---

## SECTION 12 — QUICK REFERENCE COMMANDS

```bash
# Start of every session — sync with upstream
git fetch upstream && git rebase upstream/main  # (or upstream/master for TIGRE)

# Find your issue
open "https://github.com/CERN/TIGRE/issues?q=is:open+sort:updated-desc"

# Reproduce before fixing
python repro.py

# Run targeted tests
python -m pytest Python/tests/test_denoise.py -v    # TIGRE
python -m pytest tests/ -k "test_my_fix" -v          # scikit-hep

# Check NumPy 2.x issues in TIGRE
grep -r "np\.bool\b\|np\.int\b\|np\.float\b" Python/ --include="*.py"

# Check what you're about to commit
git diff --stat upstream/master...HEAD   # TIGRE
git diff --stat upstream/main...HEAD     # scikit-hep

# Build awkward C++ from scratch (after changing C++ kernels)
pip install -e ".[dev]" --no-build-isolation
```

---

## SECTION 13 — RESOURCES AND LINKS

**Repos:**
- TIGRE: `https://github.com/CERN/TIGRE`
- CAiMIRA: `https://github.com/CERN/CAiMIRA`
- awkward: `https://github.com/scikit-hep/awkward`
- uproot5: `https://github.com/scikit-hep/uproot5`
- vector: `https://github.com/scikit-hep/vector`
- pyhf: `https://github.com/scikit-hep/pyhf`

**Documentation:**
- awkward docs: `https://awkward-array.org/`
- uproot docs: `https://uproot.readthedocs.io/`
- vector docs: `https://vector.readthedocs.io/`
- pyhf docs: `https://pyhf.readthedocs.io/`
- TIGRE paper: `https://arxiv.org/abs/2412.10129` (v3, 2024)

**Learning HEP software:**
- HSF training: `https://hsf-training.github.io/`
- Scikit-HEP tutorials: `https://scikit-hep.org/`
- IRIS-HEP: `https://iris-hep.org/`
- PyHEP workshops (annual): `https://indico.cern.ch/` (search PyHEP)

**CERN open source:**
- CERN Open Source: `https://openscience.cern/software`
- awesome-cern (all CERN software): `https://github.com/CERN/awesome-cern`
- HSF GSoC: `https://hepsoftwarefoundation.org/activities/gsoc.html`

**Physics references (for algorithm implementations):**
- arXiv hep-ex: `https://arxiv.org/list/hep-ex/recent` (experimental HEP papers)
- IEEE TMI: CT reconstruction papers
- Biomedical Physics & Engineering Express: TIGRE's home journal

---

*Last updated: June 2026. All issue data verified against live GitHub repositories.*
*Primary org: `https://github.com/orgs/CERN/repositories`*
*Extended ecosystem: `https://github.com/scikit-hep`*
