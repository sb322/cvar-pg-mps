# cvar_pg_mps.py
# -----------------------------
# CVaR-constrained policy gradient on a tiny MDP with a multi-point heavy tail.
# Device auto-selects: MPS (Apple Silicon) -> CUDA -> CPU.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# -----------------------------
# Device selection (MPS -> CUDA -> CPU)
# -----------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
torch.set_default_dtype(torch.float32)

# -----------------------------
# 1) Toy MDP with a 3-point risky tail
# -----------------------------
class RiskyBanditEnv:
    """
    One-step episodic env with actions: SAFE (deterministic cost) vs RISKY (3-point mixture).
    RISKY costs:
        - cost0 with prob p0 (e.g., 0 with 0.7)
        - cost1 with prob p1 (e.g., 5 with 0.2)
        - cost2 with prob p2 (e.g., 30 with 0.1)
    """
    def __init__(
        self,
        p0: float = 0.7,
        p1: float = 0.2,
        p2: float = 0.1,
        cost0: float = 0.0,
        cost1: float = 5.0,
        cost2: float = 30.0,
        safe_cost: float = 1.0,
        seed: int = 0,
    ):
        assert abs(p0 + p1 + p2 - 1.0) < 1e-8, "probabilities must sum to 1"
        self.p0, self.p1, self.p2 = p0, p1, p2
        self.c0, self.c1, self.c2 = cost0, cost1, cost2
        self.safe_cost = safe_cost
        self.rng = np.random.default_rng(seed)
        self.state_dim = 1
        self.n_actions = 2  # 0=SAFE, 1=RISKY

    def reset(self) -> int:
        return 0

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        if action == 0:  # SAFE
            cost = self.safe_cost
        else:            # RISKY: 3-point mixture
            u = self.rng.random()
            if u < self.p0:
                cost = self.c0
            elif u < self.p0 + self.p1:
                cost = self.c1
            else:
                cost = self.c2
        done = True
        info = {"cost": cost}
        return 0, -cost, done, info  # reward = -cost for completeness

# -----------------------------
# 2) Policy (Categorical) with MPS-safe sampling
# -----------------------------
class TinyPolicy(nn.Module):
    """
    Two-action policy [SAFE, RISKY].
    Sample action on CPU (avoids rare MPS RNG stalls), compute log-prob on DEVICE.
    """
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, device=DEVICE))  # start uniform

    def forward(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)

    def act(self) -> Tuple[int, torch.Tensor]:
        # Sample on CPU for robustness
        with torch.no_grad():
            logits_cpu = self.logits.detach().cpu()
            a_cpu = torch.distributions.Categorical(logits=logits_cpu).sample()
        # Log-prob on DEVICE (grad flows to logits)
        a_dev = a_cpu.to(self.logits.device)
        logp = self.forward().log_prob(a_dev)
        return int(a_cpu.item()), logp

    def prob_risky(self) -> float:
        with torch.no_grad():
            probs = self.forward().probs
            return float(probs[1].detach().cpu().item())

# -----------------------------
# 3) Rollout utils
# -----------------------------
@dataclass
class Traj:
    cost: float
    logp_sum: torch.Tensor  # sum log π(a|s); on DEVICE

def collect_batch(env: RiskyBanditEnv, policy: TinyPolicy, batch_size: int) -> List[Traj]:
    out: List[Traj] = []
    for _ in range(batch_size):
        env.reset()
        a, logp = policy.act()
        _, _, done, info = env.step(a)
        assert done
        out.append(Traj(cost=float(info["cost"]), logp_sum=logp))
    return out

# -----------------------------
# 4) CVaR helpers
# -----------------------------
def empirical_var(c: np.ndarray, alpha: float) -> float:
    # Use 'nearest' to stabilize discrete tails
    return float(np.percentile(c, 100 * alpha, method="nearest"))

def empirical_cvar(c: np.ndarray, alpha: float, t: Optional[float] = None) -> float:
    if t is None:
        t = empirical_var(c, alpha)
    tail = c[c > t]
    if tail.size == 0:
        return t
    return float(tail.mean())

def ru_surrogate(c: torch.Tensor, alpha: float, t: torch.Tensor) -> torch.Tensor:
    hinge = torch.relu(c - t)
    return t + hinge.mean() / (1.0 - alpha)

# -----------------------------
# 5) Training loop
# -----------------------------
@dataclass
class TrainConfig:
    mode: str = "constraint"   # "constraint" or "objective"
    alpha: float = 0.8         # move cutoff into the body of risky mixture
    kappa: float = 8.0         # CVaR budget to discourage heavy tail
    batch_size: int = 1024
    steps: int = 600
    lr_theta: float = 0.05
    lr_t: float = 0.05
    lr_lambda: float = 0.05
    use_plugin_t: bool = True
    entropy_coeff: float = 0.0
    baseline: str = "mean"     # "none" or "mean"
    grad_clip: float = 10.0

def train(env: RiskyBanditEnv, cfg: TrainConfig, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = TinyPolicy().to(DEVICE)
    opt = optim.SGD(policy.parameters(), lr=cfg.lr_theta)

    t_param = nn.Parameter(torch.tensor(1.0, device=DEVICE))
    opt_t = optim.SGD([t_param], lr=cfg.lr_t)

    lam = torch.tensor(0.0, device=DEVICE)

    logs = {"mean_cost": [], "var": [], "cvar": [], "t": [], "lambda": [], "risk_prob": [], "pi_risky": []}

    for step in range(cfg.steps):
        if step % 10 == 0:
            print(f"step {step}...", flush=True)

        # 1) Rollout
        trajs = collect_batch(env, policy, cfg.batch_size)
        costs = np.array([tr.cost for tr in trajs], dtype=np.float32)
        logps = torch.stack([tr.logp_sum for tr in trajs], dim=0).to(DEVICE)

        # 2) t & statistics
        if cfg.use_plugin_t:
            t_value = empirical_var(costs, cfg.alpha)
            t_tensor = torch.tensor(t_value, dtype=torch.float32, device=DEVICE)
        else:
            t_tensor = t_param

        c_tensor = torch.tensor(costs, dtype=torch.float32, device=DEVICE)
        ru = ru_surrogate(c_tensor, cfg.alpha, t_tensor)
        c_mean = c_tensor.mean()
        c_var  = float(np.var(costs))
        c_cvar = empirical_cvar(costs, cfg.alpha, t=float(t_tensor.detach().cpu().item()))

        # 3) Baseline
        baseline = c_mean.detach() if cfg.baseline == "mean" else torch.tensor(0.0, device=DEVICE)

        # 4) Policy update
        if cfg.mode == "constraint":
            tail = torch.relu(c_tensor - t_tensor)
            w = (c_tensor - baseline) + (lam / (1.0 - cfg.alpha)) * tail

            entropy = policy.forward().entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            opt.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()

            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                ru.backward()
                opt_t.step()

            lam = torch.clamp(lam + cfg.lr_lambda * (ru.detach() - cfg.kappa), min=0.0)

        else:
            tail = torch.relu(c_tensor - t_tensor)
            base = tail.mean() if cfg.baseline != "none" else torch.tensor(0.0, device=DEVICE)
            w = (tail - base) / (1.0 - cfg.alpha)

            entropy = policy.forward().entropy().mean()
            loss_pg = (w.detach() * (-logps)).mean() - cfg.entropy_coeff * entropy

            opt.zero_grad(set_to_none=True)
            loss_pg.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            opt.step()

            if not cfg.use_plugin_t:
                opt_t.zero_grad(set_to_none=True)
                ru.backward()
                opt_t.step()

        # 5) Logging
        logs["mean_cost"].append(float(c_mean.detach().cpu().item()))
        logs["var"].append(float(c_var))
        logs["cvar"].append(float(c_cvar))
        logs["t"].append(float(t_tensor.detach().cpu().item()))
        logs["lambda"].append(float(lam.detach().cpu().item()))
        risk_prob = float((costs > float(t_tensor.detach().cpu().item())).mean())
        logs["risk_prob"].append(risk_prob)
        logs["pi_risky"].append(policy.prob_risky())

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        if (step + 1) % 100 == 0:
            print(f"[{step+1}] device={DEVICE} mean={logs['mean_cost'][-1]:.3f} "
                  f"CVaR={logs['cvar'][-1]:.3f} VaR t={logs['t'][-1]:.3f} "
                  f"λ={logs['lambda'][-1]:.3f}  π(risky)={logs['pi_risky'][-1]:.3f}",
                  flush=True)

    return policy, logs

# -----------------------------
# 6) Run experiment
# -----------------------------
if __name__ == "__main__":
    print("Using device:", DEVICE)

    # Heavier tail with multiple levels -> VaR and CVaR separate at alpha=0.8
    env = RiskyBanditEnv(
        p0=0.7, p1=0.2, p2=0.1,
        cost0=0.0, cost1=5.0, cost2=30.0,
        safe_cost=1.0,
        seed=42
    )

    cfg = TrainConfig(
        mode="constraint",
        alpha=0.8,        # puts VaR near 5 while CVaR averages {5,30} tail
        kappa=8.0,        # encourages avoiding the heavy tail
        batch_size=1024,
        steps=600,
        lr_theta=0.05,
        lr_t=0.05,
        lr_lambda=0.05,
        use_plugin_t=True,
        entropy_coeff=0.0,
        baseline="mean",
        grad_clip=10.0,
    )

    policy, logs = train(env, cfg, seed=0)

    # --- Checks ---
    mean_cost = np.array(logs["mean_cost"])
    cvars     = np.array(logs["cvar"])
    ts        = np.array(logs["t"])
    assert not np.isnan(mean_cost).any()
    assert not np.isnan(cvars).any()
    assert np.all(cvars[-50:] + 1e-6 >= ts[-50:]), "CVaR should be >= VaR"

    # --- Plots ---
    x = np.arange(len(mean_cost))

    plt.figure()
    plt.plot(x, mean_cost, label="Mean cost")
    plt.plot(x, cvars,     label=f"CVaR@{cfg.alpha}")
    plt.plot(x, ts,        label=f"VaR t@{cfg.alpha}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend()
    plt.title(f"Mean vs CVaR vs VaR ({DEVICE})")
    plt.show()

    plt.figure()
    plt.plot(x, np.array(logs["lambda"]), label="lambda")
    plt.xlabel("Iteration")
    plt.ylabel("λ")
    plt.legend()
    plt.title("Lagrange Multiplier")
    plt.show()

    plt.figure()
    plt.plot(x, np.array(logs["pi_risky"]), label="π(risky)")
    plt.xlabel("Iteration")
    plt.ylabel("Policy prob of risky")
    plt.legend()
    plt.title("Policy moves to satisfy CVaR constraint")
    plt.show()