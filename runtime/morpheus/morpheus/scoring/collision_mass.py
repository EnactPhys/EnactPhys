"""Mass estimator used by the collision *Physical Invariance* score.

Given the observed 1-D trajectories of two colliding bodies, an MLP trained on
elastic-collision simulations predicts the mass ratio, which feeds the collision
energy/momentum conservation checks in :mod:`morpheus.scoring.physical_score`.

This is distinct from ``CollisionPINN`` in :mod:`morpheus.scoring.dynamical_score`
(which handles the *Dynamical* collision score). The module-level seeds below are
load-bearing: ``predict_masses_from_directory`` trains a fresh MLP on random
simulations and relies on this seeding for deterministic scores.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import pickle
import pandas as pd

from ..device import get_device

device = str(get_device())
torch.manual_seed(0)
np.random.seed(0)


def finite_diff(y, t):
    # y: [N, 2], t: [N,1]
    dt = (t[1:] - t[:-1]).clamp_min(1e-8)  # [N-1,1]
    v = (y[1:] - y[:-1]) / dt              # [N-1,2]
    return v

def generate_observed_trajectory(m1_true, m2_true, v1_0=2.0, v2_0=0.0, T=2.0, N=400, R_dyn=0.2):
    base_radius = 0.08
    r1_true = base_radius * (m1_true ** (1/3))
    r2_true = base_radius * (m2_true ** (1/3))
    x1_0, x2_0 = -1.0, +1.0
    t = torch.linspace(0., T, N)
    rel_speed = (v1_0 - v2_0) + 1e-8
    tc_val = ((x2_0 - x1_0) - R_dyn) / rel_speed
    tc = t.new_tensor(tc_val)
    tc = torch.clamp(tc, min=t[1], max=t[-2])
    x1 = torch.empty_like(t); x2 = torch.empty_like(t)
    pre = t <= tc
    x1[pre] = x1_0 + v1_0 * t[pre]
    x2[pre] = x2_0 + v2_0 * t[pre]
    v1p = ((m1_true - m2_true)*v1_0 + 2*m2_true*v2_0) / (m1_true + m2_true)
    v2p = ((m2_true - m1_true)*v2_0 + 2*m1_true*v1_0) / (m1_true + m2_true)
    x1_tc = x1_0 + v1_0 * tc
    x2_tc = x2_0 + v2_0 * tc
    post = t > tc
    dt_post = t[post] - tc
    x1[post] = x1_tc + v1p * dt_post
    x2[post] = x2_tc + v2p * dt_post
    x_data = torch.stack([x1, x2], dim=1)  # [N,2]
    return t.unsqueeze(1), x_data, r1_true, r2_true, (x1_0, x2_0)


def plot_collision_true_vs_pred(t_vals, x_true, x_pred, out_path, title="Collision: true vs predicted"):

    import numpy as np
    import matplotlib.pyplot as plt

    # to numpy
    if hasattr(t_vals, "detach"): t = t_vals.detach().cpu().reshape(-1).numpy()
    else: t = np.asarray(t_vals).reshape(-1)

    if hasattr(x_true, "detach"): xt = x_true.detach().cpu().numpy()
    else: xt = np.asarray(x_true)

    if hasattr(x_pred, "detach"): xp = x_pred.detach().cpu().numpy()
    else: xp = np.asarray(x_pred)

    # metrics (per-object + combined)
    mse1 = np.mean((xp[:,0] - xt[:,0])**2)
    mse2 = np.mean((xp[:,1] - xt[:,1])**2)
    var_pool = np.var(np.concatenate([xt[:,0], xt[:,1]]))
    nmse = (mse1 + mse2)/2.0 / (var_pool + 1e-12)

    # plot
    plt.figure(figsize=(8, 5))
    # object 1
    plt.plot(t, xt[:,0], label="obj1 true", linewidth=2)
    plt.plot(t, xp[:,0], "--", label="obj1 pred")
    # object 2
    plt.plot(t, xt[:,1], label="obj2 true", linewidth=2)
    plt.plot(t, xp[:,1], "--", label="obj2 pred")

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title(f"{title}\nMSE1={mse1:.3f} | MSE2={mse2:.3f} | NMSE={nmse:.3f}")
    plt.grid(True)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, 64), nn.Tanh()]
        for _ in range(2):
            layers += [nn.Linear(64, 64), nn.Tanh()]
        layers += [nn.Linear(64, 2)]
        self.net = nn.Sequential(*layers)
    def forward(self, t): 
        return self.net(t)

def run_candidate(
    m1, m2, t, x_data, 
    steps=5000, lr=1e-3, return_artifacts=False,
    touch_threshold=2,   
    cross_w=1e4,            
    touch_w=1e4            
):
    t = t.to(device)
    x_data = x_data.to(device)

    net = MLP().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)

    loss_total_list, loss_data_list, loss_phys_list = [], [], []
    cross_list, min_gap_list, touch_list = [], [], []

    for it in range(steps):
        opt.zero_grad()
        x_pred = net(t)  

        loss_data = ((x_pred - x_data) ** 2).mean()


        gap = x_pred[:, 1] - x_pred[:, 0]          
        cross = torch.relu(-gap).max()              
        min_gap = gap.min()                         
        touch = torch.relu(min_gap - touch_threshold)  

        v_fd = finite_diff(x_pred, t)               
        tc_idx = int(torch.argmin(gap).item())
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]
        v_plus  = v_fd[post_i]
        mom_before = m1 * v_minus[0] + m2 * v_minus[1]
        mom_after  = m1 * v_plus[0]  + m2 * v_plus[1]
        en_before = 0.5 * m1 * (v_minus[0]**2) + 0.5 * m2 * (v_minus[1]**2)
        en_after  = 0.5 * m1 * (v_plus[0]**2)  + 0.5 * m2 * (v_plus[1]**2)
        loss_gov = (mom_after - mom_before).pow(2) + (en_after - en_before).pow(2)

        loss_rules = cross_w * cross + touch_w * touch
        loss = loss_data + loss_gov + loss_rules
        loss.backward()
        opt.step()

        loss_total_list.append(loss.item())
        loss_data_list.append(loss_data.item())
        loss_phys_list.append(loss_gov.item())
        cross_list.append(float(cross.item()))
        min_gap_list.append(float(min_gap.item()))
        touch_list.append(float(touch.item()))

    net.eval()
    with torch.no_grad():
        x_pred = net(t)
        mse = ((x_pred - x_data)**2).mean().item()
        var_pooled = torch.var(torch.cat([x_data[:, 0], x_data[:, 1]], dim=0), unbiased=False).item()
        nmse = mse / (var_pooled + 1e-12)

        gap = x_pred[:, 1] - x_pred[:, 0]
        v_fd = finite_diff(x_pred, t)
        tc_idx = int(torch.argmin(gap).item())
        pre_i  = max(0, min(v_fd.shape[0]-1, tc_idx-1))
        post_i = max(0, min(v_fd.shape[0]-1, tc_idx))
        v_minus = v_fd[pre_i]; v_plus = v_fd[post_i]
        mom_resid = (m1 * v_plus[0] + m2 * v_plus[1]) - (m1 * v_minus[0] + m2 * v_minus[1])
        en_resid  = (0.5*m1*v_plus[0]**2 + 0.5*m2*v_plus[1]**2) - (0.5*m1*v_minus[0]**2 + 0.5*m2*v_minus[1]**2)
        loss_gov = (mom_resid**2 + en_resid**2).item()

        cross_eval = torch.relu(-gap).max().item()
        min_gap_eval = gap.min().item()
        touch_eval = torch.relu(gap.min() - touch_threshold).item()

        total = mse + loss_gov + cross_w * cross_eval + touch_w * touch_eval

    artifacts = {
        "x_pred": x_pred.detach().cpu().numpy(),
        "x_true": x_data.detach().cpu().numpy(),
        "losses": {
            "total": loss_total_list,
            "data": loss_data_list,
            "gov": loss_phys_list,
            "cross": cross_list,
            "min_gap": min_gap_list,
            "touch": touch_list
        },
        "t": t.detach().cpu().squeeze().numpy(),
        "rank_score": total,
        "mse": mse,
        "nmse": nmse,
        "gov": loss_gov,
        "rules_eval": {
            "cross": cross_eval,
            "min_gap": min_gap_eval,
            "touch": touch_eval,
            "touch_threshold": float(touch_threshold)
        }
    }
    return total, net, artifacts



@torch.no_grad()
def extract_vel_features(t, x, k=10):
    v_fd = finite_diff(x, t)           # [N-1,2]
    gap = (x[:,1] - x[:,0])            # [N]
    tc_idx = int(torch.argmin(gap).item())
    tc_idx = max(k, min(tc_idx, v_fd.shape[0]-k))
    v_minus = v_fd[tc_idx-k:tc_idx].mean(dim=0)   # [2]
    v_plus  = v_fd[tc_idx:tc_idx+k].mean(dim=0)   # [2]
    feat = torch.stack([v_minus[0], v_minus[1], v_plus[0], v_plus[1]], dim=0)
    return feat

class MassMLP(nn.Module):
    def __init__(self, m_min=1.0, m_max=10.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
        self.m_min = m_min
        self.m_max = m_max

    def forward(self, feat_n):
        s = self.net(feat_n) 
        m = self.m_min + (self.m_max - self.m_min) * torch.sigmoid(s)
        return m


def train_mass_mlp(num_sims=4000, epochs=200, lr=1e-3, 
                   T=2.0, N=400, v1_range=(0.5,5.0), v2_range=(0.0,10.0),
                   m_range=(1.0,10.0), device=None, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = MassMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    feats, targets = [], []
    for _ in range(num_sims):
        m1 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        m2 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
        v1_0 = float(np.random.uniform(*v1_range))
        v2_0 = float(np.random.uniform(*v2_range))

        t, x_data, _, _, _ = generate_observed_trajectory(m1, m2, v1_0=v1_0, v2_0=v2_0, T=T, N=N)
        t = t.to(device); x_data = x_data.to(device)
        feat = extract_vel_features(t, x_data)     # [4]
        feats.append(feat)
        targets.append(torch.tensor([m1, m2], device=device))

    X = torch.stack(feats, dim=0).to(device)       # [M,4]
    Y = torch.stack(targets, dim=0).to(device)     # [M,2]

    mu = X.mean(dim=0, keepdim=True)
    sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sigma
    model.feat_mu = mu.detach()
    model.feat_sigma = sigma.detach()

    model.train()
    bs = 256
    for ep in range(epochs):
        idx = torch.randperm(Xn.shape[0], device=device)
        tot = 0.0
        for i in range(0, Xn.shape[0], bs):
            xb = Xn[idx[i:i+bs]]
            yb = Y[idx[i:i+bs]]
            pred = model(xb)
            loss = ((pred - yb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        if verbose and (ep % 20 == 0 or ep == epochs-1):
            print(f"[MassMLP] epoch {ep+1:03d}/{epochs} | MSE={tot/Xn.shape[0]:.4e}")
    model.eval()
    return model

@torch.no_grad()
def mlp_predict_masses(model, t, x):
    """Predict (m1, m2) from positions x(t) using MassMLP."""
    dev = next(model.parameters()).device
    t = t.to(dev); x = x.to(dev)
    feat = extract_vel_features(t, x)                                 
    feat_n = (feat - model.feat_mu.squeeze(0)) / model.feat_sigma.squeeze(0)
    m = model(feat_n.unsqueeze(0)).squeeze(0)                          
    return float(m[0].item()), float(m[1].item())

def load_trajectories(base_directory, file_name, kernel_size=25):
    all_trajectories = []
    for trajectory_number in range(9):
        video_dir = os.path.join(base_directory, f"video_{trajectory_number}_fps30")
        file_path = os.path.join(video_dir, file_name)
        
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            
            data_len = len(data)
            t = np.arange(data_len)
            x = np.array([item[0] if item is not None else np.nan for item in data])
            y = np.array([item[1] if item is not None else np.nan for item in data])
            
            valid_start_idx = np.where(~np.isnan(x))[0][0]
            x = x[valid_start_idx:]
            y = y[valid_start_idx:]
            t = t[valid_start_idx:]
            
            df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
            traj_x = df['x'].to_numpy() / 100.0
            traj_y = df['y'].to_numpy() / 100.0
            traj_t = df['t'].to_numpy()
            
            traj_t_shifted = np.linspace(0, 6, len(traj_x)) 
            t_data = torch.tensor(traj_t_shifted, dtype=torch.float32).reshape(-1, 1)
            x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
            y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)

            all_trajectories.append((t_data, x_data, y_data))
    return all_trajectories


def make_common_time_grid(T=2.0, N=400):
    return np.linspace(0.0, T, N, dtype=np.float32)

def resample_series_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    t_src = t_src.flatten().astype(np.float32)
    y_src = y_src.flatten().astype(np.float32)
    return np.interp(t_grid, t_src, y_src)

def build_real_observation_from_pkls(traj_A, traj_B, T_match=2.0, N_match=400):
    tA, _, yA = traj_A
    tB, _, yB = traj_B

    tA_np = tA.squeeze(1).cpu().numpy().astype(np.float32)
    tB_np = tB.squeeze(1).cpu().numpy().astype(np.float32)
    yA_np = yA.squeeze(1).cpu().numpy().astype(np.float32)
    yB_np = yB.squeeze(1).cpu().numpy().astype(np.float32)

    t_min = max(tA_np.min(), tB_np.min())
    t_max = min(tA_np.max(), tB_np.max())
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        t_min = min(tA_np.min(), tB_np.min())
        t_max = max(tA_np.max(), tB_np.max())

    t_obs = np.linspace(t_min, t_max, N_match, dtype=np.float32)

    y1_obs = np.interp(t_obs, tA_np, yA_np)
    y2_obs = np.interp(t_obs, tB_np, yB_np)

    T_obs = max(t_max - t_min, 1e-6)
    t_match = (t_obs - t_min) * (T_match / T_obs)

    t_torch = torch.tensor(t_match, dtype=torch.float32).reshape(-1, 1)
    x_torch = torch.tensor(np.stack([y1_obs, y2_obs], axis=1), dtype=torch.float32) 
    return t_torch, x_torch

@torch.no_grad()
def mlp_predict_masses_on_real(mass_mlp: nn.Module, t_real: torch.Tensor, x_real: torch.Tensor):
    return mlp_predict_masses(mass_mlp, t_real, x_real)

def run_pinn_on_real_with_mlp(mass_mlp: nn.Module,
                              traj_A, traj_B,
                              steps=10000, T_match=2.0, N_match=400,
                              tag="real"):
    t_real, x_real = build_real_observation_from_pkls(traj_A, traj_B,
                                                      T_match=T_match, N_match=N_match)
    t_real = t_real.to(device)
    x_real = x_real.to(device)

    m1_hat, m2_hat = mlp_predict_masses_on_real(mass_mlp, t_real, x_real)
    print(f"[{tag}] MassMLP on REAL y(t): (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f})")

    score, net, art = run_candidate(m1_hat, m2_hat, t_real, x_real, steps=steps, return_artifacts=True)
    print(f"[{tag}] PINN total={score:.6e} | data={art['mse']:.6e} | gov={art['gov']:.6e}")
    print(f"[{tag}] NMSE={art['nmse']:.6f}")

    os.makedirs("collision_gifs2", exist_ok=True)
    os.makedirs("collision_gifs3", exist_ok=True)
    os.makedirs("collision_gifs4", exist_ok=True)

    # base_radius = 0.08
    # r1_cand = base_radius * (m1_hat ** (1/3))
    # r2_cand = base_radius * (m2_hat ** (1/3))

    # x_pred = art["x_pred"]   
    # t_vals = art["t"]        
    tag_best = f"REAL_predMLP_{m1_hat:.3f}_{m2_hat:.3f}"
    
    plot_collision_true_vs_pred(
    t_vals=art["t"],
    x_true=x_real.detach().cpu().numpy(),   # [N,2]
    x_pred=art["x_pred"],                   # [N,2]
    out_path=f"collision_gifs4/real_vs_pred_{tag_best}.png",
    title=f"Real vs Predicted (m̂1={m1_hat:.2f}, m̂2={m2_hat:.2f})"
)

    print(f"\n=== REAL MASS RESULT for {tag} ===")
    print(f"Pred masses: (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f}) | TOTAL={score:.6e}")
    return {"m1_hat": m1_hat, "m2_hat": m2_hat, "score": score, "art": art}


def predict_masses_from_directory(base_directory, 
                                    file_A="centres3d_obj_1.pkl", 
                                    file_B="centres3d_obj_2.pkl",
                                    mass_mlp=None,
                                    epochs=1000, lr=9e-5,
                                    steps=10000, T_match=2.0, N_match=400,
                                    output_dir="collision_gifs4",
                                    tag="directory_call"
                                ):
    if mass_mlp is None:
        print("Training new MassMLP...")
        mass_mlp = train_mass_mlp(
            num_sims=4000, epochs=1000, lr=9e-5,
            T=2.0, N=400,
            v1_range=(0.5,5.0), v2_range=(-0.0,10.0),
            m_range=(1.0,10.0),  
            device=device, verbose=True
        )
    
    traj_A = load_trajectories(base_directory, file_A)
    traj_B = load_trajectories(base_directory, file_B)

    n_traj = min(len(traj_A), len(traj_B))
    results = []
    for idx in range(n_traj):
        print(f"\n=== REAL RUN: video_{idx} ===")
        result = run_pinn_on_real_with_mlp(
            mass_mlp,
            traj_A[idx], traj_B[idx],
            steps=steps,
            T_match=T_match,
            N_match=N_match,
            tag=f"real_video_{idx}"
        )
        results.append(result)
    return results
