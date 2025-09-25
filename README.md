# OCE–CVaR Policy Gradient on a Risky Bandit

This project implements a **Conditional Value-at-Risk (CVaR)** optimization method using the **Optimized Certainty Equivalent (OCE)** representation.  
The OCE formulation makes CVaR differentiable and convex, enabling direct optimization in reinforcement learning with policy gradients.

The implementation is in **PyTorch**, with automatic device selection (**Apple MPS → CUDA → CPU**).

---

## ✨ What’s Implemented

- **Toy Risky Bandit Environment**  
  - Two actions: `SAFE` (fixed cost) vs `RISKY` (heavy-tailed stochastic cost).  
  - Flexible probabilities and outcomes to illustrate tail risks.

- **Policy**  
  - Tiny 2-action categorical policy.  
  - Safe action sampling on CPU, log-prob/gradients on device (avoids MPS sampling hangs).  

- **Risk Objective**  
  - CVaR is represented via the **Optimized Certainty Equivalent (OCE)**:  
    \[
    \text{CVaR}_\alpha(C) = \inf_{t \in \mathbb{R}} \; t + \frac{1}{1-\alpha}\,\mathbb{E}[(C - t)_+]
    \]  
  - This Rockafellar–Uryasev surrogate is used for optimization.  

- **Optimization**  
  - CVaR can be optimized as an **objective** or enforced as a **constraint** with Lagrangian dual updates.  
  - Includes entropy regularization, baselines, and gradient clipping.

- **Logging & Visualization**  
  - Tracks mean cost, VaR, CVaR (via OCE), λ (dual multiplier), and π(risky).  
  - Produces plots showing risk dynamics and policy adaptation.

---

## 📊 Why OCE–CVaR Matters

- **Naive CVaR** depends on the quantile (VaR) and is non-smooth.  
- **OCE formulation** restores convexity and differentiability, making CVaR amenable to gradient-based optimization.  
- This is crucial for reinforcement learning, where smooth surrogates are needed for policy gradient updates.  

The toy example demonstrates how an agent reduces risky behavior to satisfy a CVaR constraint, even if the mean cost looks acceptable.

---

## 🚀 How to Run

### Requirements
- Python 3.11+ (arm64 on macOS recommended)  
- Install dependencies:
  ```bash
  python3 -m pip install --upgrade pip
  pip install -r requirements.txt
  

---

## 📖 References

1. Rockafellar, R. T., & Uryasev, S. (2000).  
   *Optimization of Conditional Value-at-Risk.* Journal of Risk.  
   [https://doi.org/10.21314/JOR.2000.038](https://doi.org/10.21314/JOR.2000.038)

2. Ben-Tal, A., & Teboulle, M. (2007).  
   *An Old-New Concept of Convex Risk Measures: The Optimized Certainty Equivalent.*  
   Mathematical Finance, 17(3), 449–476.  
   [https://doi.org/10.1111/j.1467-9965.2007.00296.x](https://doi.org/10.1111/j.1467-9965.2007.00296.x)

3. Chow, Y., Tamar, A., Mannor, S., & Pavone, M. (2015).  
   *Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach.*  
   NeurIPS 2015.  
   [PDF](https://proceedings.neurips.cc/paper/2015/file/9a49a25d7cdc7b71924a70c0c7d2b0f9-Paper.pdf)

4. Chow, Y., & Ghavamzadeh, M. (2014).  
   *Algorithms for CVaR Optimization in MDPs.*  
   NeurIPS 2014.  
   [PDF](https://proceedings.neurips.cc/paper/2014/file/9dcb88e0137649590b755372b040afad-Paper.pdf)