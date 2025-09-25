"""
oce_cvar_pg_mps_explained.py
===============================================================
A teaching-first, PyCharm-friendly implementation of OCE-CVaR
policy gradient on a tiny risky bandit, with Apple MPS support.

READ THIS FIRST (how to study with PyCharm / IntelliJ):
  • Hover a symbol and press Quick Documentation (Mac: ⌘J, Win/Linux: Ctrl+Q)
    to read the docstring explanations below.
  • Use the Structure tool window (⌘7 / Alt+7) to jump between sections.
  • Collapse/expand # region blocks from the gutter to read chunk by chunk.
  • Set a breakpoint in `train_oce_cvar()` and step through once; watch variables
    `c_tensor`, `t`, `oce_val`, `lam`, and `w`.

Command-line quickstart:
  $ python oce_cvar_pg_mps_explained.py --steps 200 --batch 2048 --mode constraint --alpha 0.9 --kappa 10

Why an OCE layer?
  OCE turns a risk measure into:    ρ_φ(Z) = min_t [ t + E[ φ(Z - t) ] ].
  For CVaR at level α:              φ(u) = (1/(1-α)) * relu(u).
  This is exactly Rockafellar–Uryasev but in a general, extensible wrapper.
"""

# region Imports & Device ------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use("Agg")  # Render plots to files; avoids GUI blocking on macOS.
import matplotlib.pyplot as plt

# Make MPS safer: unsupported kernels transparently fall back to CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
torch.set_default_dtype(torch.float32)


def get_device(force: Optional[str] = None) -> torch.device:
    """
    Decide which accelerator to use.

    Tip (PyCharm): place caret on 'torch.device' → ⌘J to view type and docs.

    Args:
      force: Optional override string: 'cpu' | 'mps' | 'cuda'.

    Returns:
      A torch.device to move tensors and modules onto.
    """
    if force is not None:
        m = force.lower()
        if m == "cpu":
            return torch.device("cpu")
        if m == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if m == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        print(f"[warn] requested device '{force}' not available; falling back")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: torch.device = get_device()

# endregion --------------------------------------------------------------------


# region Environment -----------------------------------------------------------

class RiskyBanditEnv:
    """
    A one-step episodic 'environment' illustrating tail risk.

    States:
      • Single dummy state (0). We only care about which action you pick.

    Actions:
      • 0 = SAFE   → cost = safe_cost (deterministic)
      • 1 = RISKY  → cost = 0   with prob (1 - p)
                      cost = M   with prob p  (catastrophe)

    Why this env?
      • The catastrophic branch makes CVaR very different from the mean.
      • It isolates the effect of tail risk without confounding dynamics.

    Attributes:
      p        : probability of catastrophe on risky
      M        : catastrophic cost
      safe_cost: constant SAFE cost
    """
    def __init__(self, p: float = 0.1, M: float = 10.0, safe_cost: float = 1.0, seed: int = 0) -> None:
        self.p, self.M, self.safe_cost = p, M, safe_cost
        self.rng = np.random.default_rng(seed)

    def reset(self) -> int:
        """Reset returns the single state (0)."""
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Apply action and emit a cost in info['cost'].

        Returns:
          (next_state, reward, done, info)
          Note: reward = -cost only for completeness.
        """
        if action == 0:
            cost = self.safe_cost
        else:
            cost = 0.0 if self.rng.random() > self.p else self.M
        return 0, -cost, True, {"cost": cost}


# endregion --------------------------------------------------------------------


# region Policy ----------------------------------------------------------------

class TinyPolicy(nn.Module):
    """
    Minimal categorical policy with two logits [SAFE, RISKY].

    IMPORTANT (Apple Silicon / MPS):
      PyTorch's multinomial sampling on MPS can stall on some setups.
      To avoid that, we *sample action indices on CPU with NumPy*,
      then compute log_prob on DEVICE for a correct REINFORCE gradient.

    Why this is mathematically sound:
      REINFORCE needs ∇θ log πθ(a|s), not a gradient *through* the sampler.
      Computing log_prob on-device keeps the gradient path intact.

    Methods:
      dist() -> Categorical distribution object (on DEVICE).
      act()  -> (action:int, logp:Tensor on DEVICE)
    """
    def __init__(self) -> None:
        super().__init__()
        # Two logits; initialized to 0 → uniform policy.
        self.logits: nn.Parameter = nn.Parameter(torch.zeros(2))

    def dist(self) -> torch.distributions.Categorical:
        """Distribution parameterized by self.logits."""
        return torch.distributions.Categorical(logits=self.logits)

    def act(self) -> Tuple[int, torch.Tensor]:
        """
        Sample an action and return its log-prob (on DEVICE).

        Returns:
          a   : sampled action index (0 or 1) as Python int (CPU)
          logp: log πθ(a) as Tensor on DEVICE (for REINFORCE)
        """
        # 1) Sample on CPU for robustness.
        with torch.no_grad():
            probs = torch.softmax(self.logits, dim=-1).detach().cpu().numpy()
        a = int(np.random.choice(len(probs), p=probs))

        # 2) Compute log_prob on DEVICE (keeps gradient path).
        d = self.dist()
        a_tensor = torch.tensor(a, device=self.logits.device)
        logp = d.log_prob(a_tensor)
        return a, logp


# endregion --------------------------------------------------------------------


# region OCE Layer -------------------------------------------------------------

class OCE(nn.Module):
    """
    Optimized Certainty Equivalent (OCE) surrogate:

      ρ_φ(Z) = min_t [ t + E φ(Z - t) ]

    For a batch of trajectory costs c (shape [N]) and a scalar t:

      L_OCE(c, t) = t + mean( φ(c - t) )

    This is differentiable in t and in the policy parameters (via c).
    """
    def __init__(self, phi: Callable[[torch.Tensor], torch.Tensor], name: str = "OCE") -> None:
        super().__init__()
        self.phi = phi
        self.name = name

    def forward(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
          c: [N] tensor of per-trajectory costs on DEVICE.
          t: scalar tensor acting as a VaR-like threshold.

        Returns:
          Scalar OCE surrogate: t + mean(phi(c - t))
        """
        return t + self.phi(c - t).mean()


def phi_cvar(alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    CVaR penalty φ(u) = 1/(1-α) * ReLU(u).

    Notes:
      • Plugging this φ into OCE yields CVaR_α(Z).
      • Equivalent to Rockafellar–Uryasev representation.
    """
    scale = 1.0 / (1.0 - alpha)
    return lambda u: scale * torch.relu(u)


# Optional example of another OCE (not used by default).
def phi_entropic(beta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Entropic penalty: φ(u) = (exp(βu) - 1)/β (β>0)."""
    inv = 1.0 / beta
    return lambda u: inv * (torch.exp(beta * u) - 1.0)

# endregion --------------------------------------------------------------------


# region Utilities: data types & stats -----------------------------------------

@dataclass
class Traj:
    """
    Container for one trajectory outcome.

    Fields:
      cost     : scalar trajectory cost (Python float; CPU)
      logp_sum : Tensor on DEVICE with sum_t log πθ(a_t|s_t)
                 (for this bandit it's just the single-step log-prob)
    """
    cost: float
    logp_sum: torch.Tensor


def collect_batch(env: RiskyBanditEnv, policy: TinyPolicy, batch_size: int) -> List[Traj]:
    """
    Roll out a batch of one-step trajectories.

    Returns:
      List[Traj] of length batch_size.
    """
    out: List[Traj] = []
    for _ in range(batch_size):
        env.reset()
        a, logp = policy.act()
        _, _, done, info = env.step(a)
        assert done
        out.append(Traj(cost=float(info["cost"]), logp_sum=logp))
    return out


def empirical_var(c: np.ndarray, alpha: float) -> float:
    """
    Empirical VaR_α for costs: α-quantile (start of worst tail).

    Implementation detail:
      Using 'nearest' gives a stable threshold for discrete samples.
      For research-grade code, you may prefer 'linear' interpolation.
    """
    return float(np.percentile(c, 100 * alpha, method="nearest"))


def empirical_cvar(c: np.ndarray, alpha: float, t: Optional[float] = None) -> float:
    """
    Empirical CVaR_α as the mean of samples strictly above t.

    Note:
      If the tail is empty (can happen for small batches), we return t.
    """
    if t is None:
        t = empirical_var(c, alpha)
    tail = c[c > t]
    return float(t if tail.size == 0 else tail.mean())

# endregion --------------------------------------------------------------------


# region Training --------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Configuration for training.

    Fields:
      mode        : 'constraint' (Lagrangian RCMDP) or 'objective' (minimize tail)
      alpha       : CVaR level (e.g., 0.9 focuses on worst 10%)
      kappa       : CVaR budget (constraint RHS) when mode='constraint'
      batch_size  : number of trajectories per update
      steps       : number of gradient steps
      lr_theta    : policy learning rate
      lr_t        : VaR parameter learning rate (if not using plug-in)
      lr_lambda   : dual ascent step for λ
      use_plugin_t: if True, set t to batch α-quantile; else learn t
      entropy_coeff: optional entropy regularization (helps exploration)
      baseline    : 'mean' or 'none' (variance reduction)
      grad_clip   : clip policy grad norm for stability
      print_every : iterations between console prints
    """
    mode: str = "constraint"
    alpha: float = 0.9
    kappa: float = 10.0  # set 10 to make feasibility easy in this env
    batch_size: int = 2048
    steps: int = 300
    lr_theta: float = 0.05
    lr_t: float = 0.05
    lr_lambda: float = 0.02
    use_plugin_t: bool = True
    entropy_coeff: float = 0.0
    baseline: str = "mean"
    grad_clip: float = 10.0
    print_every: int = 50


def train_oce_cvar(env: RiskyBanditEnv, cfg: TrainConfig, seed: int = 0) -> Tuple[TinyPolicy, Dict[str, List[float]]]:
    """
    Main training loop for OCE-CVaR policy gradient.

    High-level view (teach-first):
      1) Sample N trajectories → get costs C_i and score functions (log-probs).
      2) Pick t (VaR proxy): batch quantile or a learned parameter.
      3) Build OCE-CVaR surrogate:    t + mean( φ(C - t) ).
      4) Form REINFORCE weights:
           constraint: (C - baseline) + λ * φ(C - t)
           objective :  φ(C - t) - mean(φ(C - t))
      5) Update policy parameters θ by descending E[-logπ * w].
      6) Update t (if learned) by descending OCE value.
      7) Update λ by ascending (OCE - κ) and clamp to ≥0.
      8) Log diagnostics.

    Returns:
      (policy, logs) where logs has keys:
        'mean_cost', 'var', 'cvar', 't', 'lambda', 'risk_prob'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -- model & optimizers
    policy = TinyPolicy().to(DEVICE)
    opt_theta = optim.SGD(policy.parameters(), lr=cfg.lr_theta)

    # OCE layer for CVaR
    oce = OCE(phi=phi_cvar(cfg.alpha), name=f"OCE-CVaR@{cfg.alpha}")

    # Learnable t (only used if cfg.use_plugin_t == False)
    t_param = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt_t = optim.SGD([t_param], lr=cfg.lr_t)

    # Dual variable λ (constraint mode); keep as non-negative scalar tensor
    lam = torch.tensor(0.0, device=DEVICE)

    # Logs (simple Python lists → easy to plot or export)
    logs: Dict[str, List[float]] = {k: [] for k in ["mean_cost", "var", "cvar", "t", "lambda", "risk_prob"]}

    for step in range(cfg.steps):
        # 1) Rollouts
        trajs = collect_batch(env, policy, cfg.batch_size)
        costs_np = np.array([tr.cost for tr in trajs], dtype=np.float32)  # CPU float array for stats
        logps = torch.stack([tr.logp_sum for tr in trajs], dim=0).to(DEVICE)  # [N] DEVICE tensor
        c_tensor = torch.tensor(costs_np, dtype=torch.float32, device=DEVICE)  # [N] DEVICE tensor

        # 2) Choose t: plug-in quantile (simple) or learnable scalar
        if cfg.use_plugin_t:
            t_val: float = empirical_var(costs_np, cfg.alpha)
            t = torch.tensor(t_val, dtype=torch.float32, device=DEVICE)
        else:
            t = t_param

        # 3) OCE-CVaR surrogate (differentiable)
        oce_val = oce(c_tensor, t)  # = t + mean(1/(1-α) * relu(c - t))

        # 4) Variance-reducing baseline for REINFORCE
        baseline = c_tensor.mean() if cfg.baseline == "mean" else torch.tensor(0.0, device=DEVICE)

        if cfg.mode == "constraint":
            # REINFORCE weights: (C - baseline) + λ * φ(C - t)
            tail_pen = phi_cvar(cfg.alpha)(c_tensor - t)
            w = (c_tensor - baseline) + lam * tail_pen

            # Optional entropy bonus to avoid premature collapse (not essential here)
            dist = policy.dist()
            entropy = dist.entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            # 5) Policy update
            opt_theta.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt_theta.step()

            # 6) t update (only if learned)
            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                oce_val.backward()  # d/dt via autograd through φ
                opt_t.step()

            # 7) λ update (ascent on violation), then clamp to ≥0
            lam = torch.clamp(lam + cfg.lr_lambda * (oce_val.detach() - cfg.kappa), min=0.0)

        else:
            # Objective mode: minimize the tail itself
            tail_pen = phi_cvar(cfg.alpha)(c_tensor - t)
            pen_base = tail_pen.mean() if cfg.baseline == "mean" else torch.tensor(0.0, device=DEVICE)
            w = (tail_pen - pen_base)

            dist = policy.dist()
            entropy = dist.entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            opt_theta.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt_theta.step()

            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                oce_val.backward()
                opt_t.step()

        # 8) Logging (convert tiny tensors to CPU scalars)
        mean_c = float(c_tensor.mean().detach().cpu().item())
        t_cpu = float(t.detach().cpu().item())
        logs["mean_cost"].append(mean_c)
        logs["var"].append(float(np.var(costs_np)))
        logs["t"].append(t_cpu)
        logs["lambda"].append(float(lam.detach().cpu().item()))
        logs["cvar"].append(float(empirical_cvar(costs_np, cfg.alpha, t=t_cpu)))
        logs["risk_prob"].append(float((costs_np > t_cpu).mean()))

        if (step + 1) % cfg.print_every == 0:
            print(
                f"[{step+1}/{cfg.steps}] device={DEVICE}  "
                f"mean={mean_c:.3f}  OCE-CVaR(est)={logs['cvar'][-1]:.3f}  "
                f"t={t_cpu:.3f}  λ={logs['lambda'][-1]:.3f}"
            )

    return policy, logs

# endregion --------------------------------------------------------------------


# region Main & CLI ------------------------------------------------------------

def main() -> None:
    """
    CLI entry point.

    Try these configurations from PyCharm's Run Configuration:
      • Feasible constraint (λ stays ~0):
          --mode constraint --alpha 0.9 --kappa 10 --steps 300 --batch 2048
      • Tight constraint (requires eliminating risky completely):
          --mode constraint --alpha 0.9 --kappa 3  --steps 400 --batch 2048
      • Tail-only optimization:
          --mode objective  --alpha 0.9 --steps 300 --batch 2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None, help="cpu|mps|cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--mode", type=str, default="constraint", choices=["constraint", "objective"])
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--kappa", type=float, default=10.0)
    parser.add_argument("--print_every", type=int, default=50)
    args = parser.parse_args()

    global DEVICE
    DEVICE = get_device(args.device)
    print("Using device:", DEVICE)

    # --- Environment (tweak M to make constraints feasible/infeasible)
    env = RiskyBanditEnv(p=0.1, M=10.0, safe_cost=1.0, seed=42)

    # --- Training configuration
    cfg = TrainConfig(
        mode=args.mode,
        alpha=args.alpha,
        kappa=args.kappa,
        batch_size=args.batch,
        steps=args.steps,
        lr_theta=0.05,
        lr_t=0.05,
        lr_lambda=0.02,
        use_plugin_t=True,
        entropy_coeff=0.0,
        baseline="mean",
        grad_clip=10.0,
        print_every=args.print_every,
    )

    # --- Train
    policy, logs = train_oce_cvar(env, cfg, seed=0)

    # --- Sanity checks
    mean_cost = np.array(logs["mean_cost"])
    cvars = np.array(logs["cvar"])
    ts = np.array(logs["t"])
    assert not np.isnan(mean_cost).any()
    assert not np.isnan(cvars).any()
    # Rough check: CVaR ≥ VaR (up to sampling noise) over the last quarter
    tail_slice = slice(max(0, len(cvars) * 3 // 4), None)
    assert np.all(cvars[tail_slice] + 1e-6 >= ts[tail_slice]), "CVaR should be ≥ VaR"

    # --- Plots to files (open from PyCharm's Project pane)
    x = np.arange(len(mean_cost))
    plt.figure()
    plt.plot(x, mean_cost, label="Mean cost")
    plt.plot(x, cvars, label=f"OCE-CVaR@{cfg.alpha}")
    plt.plot(x, ts, label=f"VaR t@{cfg.alpha}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.title(f"Mean vs OCE-CVaR vs VaR ({DEVICE})")
    plt.tight_layout()
    plt.savefig("oce_cvar_curves.png")
    plt.close()

    plt.figure()
    plt.plot(x, np.array(logs["lambda"]), label="lambda")
    plt.xlabel("Iteration")
    plt.ylabel("λ")
    plt.legend()
    plt.title("Lagrange Multiplier (constraint mode)")
    plt.tight_layout()
    plt.savefig("oce_cvar_lambda.png")
    plt.close()

    print("Saved: oce_cvar_curves.png, oce_cvar_lambda.png")


if __name__ == "__main__":
    main()

# endregion --------------------------------------------------------------------