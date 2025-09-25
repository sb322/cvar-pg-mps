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