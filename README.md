# Wavenet.jl

A Julia library for **Wavelet Networks** — neural-like models that use wavelet functions as activation units — with tools for candidate construction, selection, and fine-tuning.  

This package implements the ideas introduced in:

> Qinghua Zhang,  
> *Using Wavelet Network in Nonparametric Estimation*,  
> IEEE Transactions on Neural Networks, 8(2):227–236, 1997.  
> DOI: [10.1109/72.557660](https://doi.org/10.1109/72.557660)

---

## ✨ Features

- **Radial wavelet activations**  
  Currently includes the *Mexican Hat* radial function, but the framework is extensible to other radial wavelets.

- **Candidate generation**  
  - Lattice covering (`find_covering_wavelets`)  
  - Clustering-based wavelet placement (k-means strategy)

- **Model selection**  
  - Stepwise selection (`stepwise_wavelet_selection_wt`)  
  - Backward elimination (`backward_elimination_wt`)

- **Fine-tuning**  
  - Mini-batch SGD with L2 regularization  
  - Gradient updates for:
    - Output weights  
    - Wavelet dilation (`w`)  
    - Wavelet translation (`t`)

- **Evaluation utilities**  
  - Scaling data to support  
  - Fast evaluation on matrix inputs  
  - Helpers for extracting parameter matrices

- **Examples included**  
  - **Van der Pol oscillator prediction**  
  - **Sine function reconstruction** from randomly sampled indices  

---

## 📖 Background

Wavelet Networks (WNs) combine ideas from **wavelet analysis** and **neural networks**.  
Instead of using sigmoids or ReLU, each hidden unit is a *wavelet* with a learnable **translation** (shift) and **dilation** (scale).  

This allows:
- Sparse representations for functions with localized features
- Multi-resolution modeling
- Nonparametric estimation with strong theoretical guarantees (see Zhang, 1997)

---

## 🏗 Architecture

The computation graph of a Wavenet looks like this:

```
 Input x ─────────────┐
                      │
        ┌─────────────▼──────────────┐
        │   Wavelet candidates        │
        │ (each with dilation + shift)│
        └─────────────┬──────────────┘
                      │ ψ_j(x)
                      │
     α_j (output weight) · ψ_j(x)
                      │
        ┌─────────────▼──────────────┐
        │        Summation            │
        └─────────────┬──────────────┘
                      │
                   Output ŷ
```

---

## 🚀 Installation

Clone this repo locally and include in your project:

```julia
] dev /path/to/Wavenet.jl
```

---

## 🔧 Usage

### Example 1: Van der Pol Oscillator

```julia
include("examples/VanDerPolExample.jl")
using .VanDerPolExample

VanDerPolExample.run_vanderpol_example()
```

Produces:
- Training on a 2-lag prediction task  
- One-step and rollout predictions  
- Comparison plots

### Example 2: Sine Function (Index Input)

```julia
include("examples/SineIndexExample.jl")
using .SineIndexExample

SineIndexExample.run_sine_index_example()
```

Produces:
- Initial lattice-based approximation  
- Fine-tuned Wavenet prediction  
- Comparison with ground truth sine curve  

---

## 🧩 API Highlights

```julia
Wavelet = Wavenet.RadialActivation.MexicanHat

# Scale inputs to wavelet support
Xs = Wavenet.data_scale_to_support(X, domain, Wavelet.support)

# Generate candidate wavelets
cands = Wavenet.find_covering_wavelets(Xs, Wavelet.support, -2:6, 0.5)

# Stepwise selection
(w_init, sel_idx, A, Q) = Wavenet.stepwise_wavelet_selection_wt(Wavelet, cands, Xs, y, s)

# Fine-tune
(w, cands_tr, tr_loss, val_loss) = Wavenet.finetune_from_candidates!(
    Wavelet, cands[sel_idx], w_init, Xs, y; epochs=2000, lr=0.1, batch=64, l2=1e-4
)

# Evaluate
ŷ = Wavenet.wavenet_evaluate(Wavelet, w, [c.w for c in cands_tr], hcat(c.t for c in cands_tr), Xs)
```

---

## 📌 Notes

- Current implementation is CPU-only but threaded in core routines.  
- Designed for **research and prototyping** rather than production-scale deployment.  
- Contributions are welcome: new wavelet activations, GPU support, improved optimizers.  
