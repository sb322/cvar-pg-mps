# oce_cvar_pg_mps.py
# --------------------------------------------------------
# OCE risk layer + CVaR (OCE instance) policy gradient on a tiny env.
# Auto-device: MPS (Apple Silicon) -> CUDA -> CPU.
# --------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # avoid GUI blocking
import matplotlib.pyplot as plt

# -----------------------------
# Device selection
# -----------------------------
def get_device(force: Optional[str] = None) -> torch.device:
    if force is not None:
        m = force.lower()
        if m == "cpu":  return torch.device("cpu")
        if m == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
        if m == "mps"  and torch.backends.mps.is_available(): return torch.device("mps")
        print(f"[warn] requested device '{force}' not available; falling back")
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
torch.set_default_dtype(torch.float32)

# ========================================================
# 1) Tiny one-step env with safe vs risky action
# ========================================================
class RiskyBanditEnv:
    """One-step episodic env; action 0=SAFE, 1=RISKY."""
    def __init__(self, p: float = 0.1, M: float = 10.0, safe_cost: float = 1.0, seed: int = 0):
        self.p, self.M, self.safe_cost = p, M, safe_cost
        self.rng = np.random.default_rng(seed)

    def reset(self) -> int: return 0

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if action == 0:
            cost = self.safe_cost
        else:
            cost = 0.0 if self.rng.random() > self.p else self.M
        return 0, -cost, True, {"cost": cost}

# ========================================================
# 2) Policy (Categorical, two logits)
# ========================================================
class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))  # start uniform

    def dist(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)

    def act(self) -> Tuple[int, torch.Tensor]:
        # Sample action and get log prob (on DEVICE)
        d = self.dist()
        a = d.sample()
        logp = d.log_prob(a)
        return int(a.item()), logp

# ========================================================
# 3) OCE Risk Layer
# ========================================================
class OCE(nn.Module):
    """
    Optimized Certainty Equivalent:
      rho_phi(Z) = min_t [ t + E[phi(Z - t)] ]
    We implement it as a differentiable surrogate:
      L_OCE(c, t) = t + mean(phi(c - t))
    where:
      - c: tensor of trajectory costs on DEVICE
      - t: scalar tensor (learned) or plug-in quantile (detached)
      - phi: convex penalty function (callable tensor->tensor)
    """
    def __init__(self, phi: Callable[[torch.Tensor], torch.Tensor], name: str = "OCE"):
        super().__init__()
        self.phi = phi
        self.name = name

    def forward(self, c: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # c: [N] costs, t: scalar; all on same device
        return t + self.phi(c - t).mean()

# ---- specific OCE instances ----
def phi_cvar(alpha: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    CVaR penalty: phi(u) = (1/(1-alpha)) * relu(u)
    This yields OCE-CVaR, identical to RU-CVaR.
    """
    scale = 1.0 / (1.0 - alpha)
    return lambda u: scale * torch.relu(u)

# (optional) example of another OCE:
def phi_entropic(beta: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Entropic risk (for comparison): phi(u) = (1/beta) * (exp(beta*u) - 1).
    Not used by default; shown for extensibility.
    """
    inv = 1.0 / beta
    return lambda u: inv * (torch.exp(beta * u) - 1.0)

# ========================================================
# 4) Batch collection
# ========================================================
@dataclass
class Traj:
    cost: float
    logp_sum: torch.Tensor

def collect_batch(env: RiskyBanditEnv, policy: TinyPolicy, batch_size: int) -> List[Traj]:
    out: List[Traj] = []
    for _ in range(batch_size):
        env.reset()
        a, logp = policy.act()
        _, _, done, info = env.step(a)
        assert done
        out.append(Traj(cost=float(info["cost"]), logp_sum=logp))
    return out

# ========================================================
# 5) Utilities: empirical VaR / CVaR for logging
# ========================================================
def empirical_var(c: np.ndarray, alpha: float) -> float:
    return float(np.percentile(c, 100 * alpha, method="nearest"))

def empirical_cvar(c: np.ndarray, alpha: float, t: Optional[float] = None) -> float:
    if t is None: t = empirical_var(c, alpha)
    tail = c[c > t]
    return float(t if tail.size == 0 else tail.mean())

# ========================================================
# 6) Training
# ========================================================
@dataclass
class TrainConfig:
    mode: str = "constraint"     # "constraint" or "objective"
    alpha: float = 0.9
    kappa: float = 3.0
    batch_size: int = 1024
    steps: int = 600
    lr_theta: float = 0.05
    lr_t: float = 0.05
    lr_lambda: float = 0.05
    use_plugin_t: bool = True    # if True, set t to empirical VaR each batch
    entropy_coeff: float = 0.0
    baseline: str = "mean"       # "none" or "mean"
    grad_clip: float = 10.0

def train_oce_cvar(env: RiskyBanditEnv, cfg: TrainConfig, seed: int = 0):
    torch.manual_seed(seed); np.random.seed(seed)

    # Policy
    policy = TinyPolicy().to(DEVICE)
    opt_theta = optim.SGD(policy.parameters(), lr=cfg.lr_theta)

    # OCE (CVaR instance)
    oce = OCE(phi=phi_cvar(cfg.alpha), name=f"OCE-CVaR@{cfg.alpha}")

    # VaR t parameter (if learnable)
    t_param = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt_t = optim.SGD([t_param], lr=cfg.lr_t)

    # Multiplier λ for constraint
    lam = torch.tensor(0.0, device=DEVICE)

    logs: Dict[str, list] = {k: [] for k in ["mean_cost","var","cvar","t","lambda","risk_prob"]}

    for step in range(cfg.steps):
        # --- rollouts
        trajs = collect_batch(env, policy, cfg.batch_size)
        costs = np.array([tr.cost for tr in trajs], dtype=np.float32)          # CPU for stats
        logps = torch.stack([tr.logp_sum for tr in trajs], dim=0).to(DEVICE)   # [N] DEVICE
        c_tensor = torch.tensor(costs, dtype=torch.float32, device=DEVICE)

        # --- choose t (VaR)
        if cfg.use_plugin_t:
            t_val = empirical_var(costs, cfg.alpha)
            t = torch.tensor(t_val, dtype=torch.float32, device=DEVICE)
        else:
            t = t_param

        # --- OCE-CVaR surrogate
        oce_val = oce(c_tensor, t)  # t + E[ 1/(1-a)*relu(c - t) ]

        # --- baseline
        base = c_tensor.mean() if cfg.baseline == "mean" else torch.tensor(0.0, device=DEVICE)

        if cfg.mode == "constraint":
            # weights: (C - baseline) + λ * phi(C - t)
            tail_pen = phi_cvar(cfg.alpha)(c_tensor - t)
            w = (c_tensor - base) + lam * tail_pen

            # entropy (optional)
            d = policy.dist(); entropy = d.entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            # update θ
            opt_theta.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt_theta.step()

            # update t (if learned)
            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                oce_val.backward()      # gradient w.r.t. t via autograd through phi
                opt_t.step()

            # update λ (ascent on violation); clamp ≥ 0
            lam = torch.clamp(lam + cfg.lr_lambda * (oce_val.detach() - cfg.kappa), min=0.0)

        else:  # "objective": minimize OCE-CVaR directly
            tail_pen = phi_cvar(cfg.alpha)(c_tensor - t)
            pen_base = tail_pen.mean() if cfg.baseline == "mean" else torch.tensor(0.0, device=DEVICE)
            w = (tail_pen - pen_base)  # score weights ~ d/dθ E[phi(C-t)]

            d = policy.dist(); entropy = d.entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            opt_theta.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt_theta.step()

            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                oce_val.backward()
                opt_t.step()

        # --- logging to CPU scalars
        mean_c = float(c_tensor.mean().detach().cpu().item())
        t_cpu  = float(t.detach().cpu().item())
        logs["mean_cost"].append(mean_c)
        logs["var"].append(float(np.var(costs)))
        logs["t"].append(t_cpu)
        logs["lambda"].append(float(lam.detach().cpu().item()))
        logs["cvar"].append(float(empirical_cvar(costs, cfg.alpha, t=t_cpu)))
        logs["risk_prob"].append(float((costs > t_cpu).mean()))

        if (step+1) % 100 == 0:
            print(f"[{step+1}] device={DEVICE} mean={mean_c:.3f}  "
                  f"OCE-CVaR(est)={logs['cvar'][-1]:.3f}  t={t_cpu:.3f}  λ={logs['lambda'][-1]:.3f}")

    return policy, logs

# ========================================================
# 7) Run small experiment
# ========================================================
if __name__ == "__main__":
    print("Using device:", DEVICE)
    env = RiskyBanditEnv(p=0.1, M=10.0, safe_cost=1.0, seed=42)

    cfg = TrainConfig(
        mode="constraint",      # or "objective"
        alpha=0.9,
        kappa=3.0,              # tighten/loosen to see λ react
        batch_size=1024,
        steps=600,
        lr_theta=0.05,
        lr_t=0.05,
        lr_lambda=0.05,
        use_plugin_t=True,
        entropy_coeff=0.0,
        baseline="mean",
        grad_clip=10.0
    )

    policy, logs = train_oce_cvar(env, cfg, seed=0)

    # Simple checks
    mean_cost = np.array(logs["mean_cost"])
    cvars     = np.array(logs["cvar"])
    ts        = np.array(logs["t"])
    assert not np.isnan(mean_cost).any()
    assert not np.isnan(cvars).any()
    assert np.all(cvars[-50:] + 1e-6 >= ts[-50:]), "CVaR should be >= VaR (up to noise)"

    # Plots to files
    x = np.arange(len(mean_cost))
    plt.figure()
    plt.plot(x, mean_cost, label="Mean cost")
    plt.plot(x, cvars,     label=f"OCE-CVaR@{cfg.alpha}")
    plt.plot(x, ts,        label=f"VaR t@{cfg.alpha}")
    plt.xlabel("Iteration"); plt.ylabel("Cost"); plt.legend()
    plt.title(f"Mean vs OCE-CVaR vs VaR ({DEVICE})")
    plt.tight_layout(); plt.savefig("oce_cvar_curves.png"); plt.close()

    plt.figure()
    plt.plot(x, np.array(logs["lambda"]), label="lambda")
    plt.xlabel("Iteration"); plt.ylabel("λ"); plt.legend()
    plt.title("Lagrange Multiplier (constraint mode)")
    plt.tight_layout(); plt.savefig("oce_cvar_lambda.png"); plt.close()

    print("Saved: oce_cvar_curves.png, oce_cvar_lambda.png")