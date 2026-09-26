"""Dynamical Score -- physics-informed neural networks (PINNs) per phenomenon.

This is the paper's "Dynamical Score" (code name ``statistical_score``): a small PINN
is fit to the observed trajectory under the governing equation of motion, and the score
is ``1 - min(NMSE, 1)``. One model per phenomenon: Pendulum, SpringMass, FreeFall,
Projectile, BouncingBall, Collision, Slide, DoublePendulum. ``run_pin_framework`` is
the dispatcher used by :mod:`morpheus.scoring.combined`.

Device handling is CUDA/MPS/CPU-agnostic throughout (only small nets are trained).
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.ndimage import median_filter

from ..device import get_device

#: The trained collision mass-estimator ships alongside this module.
_MASSMLP_PATH = str(Path(__file__).resolve().parent / "collision_mass_mlp.pt")


def apply_filter(t_data, x_data, y_data=None, kernel_size=25):
    """
    Apply median filtering. If y_data is provided, both x and y are filtered.
    Otherwise only x_data is filtered.
    """
    if y_data is None:
        x_np = x_data.numpy().flatten()
        x_filtered_np = medfilt(x_np, kernel_size=kernel_size)
        x_filtered = torch.tensor(x_filtered_np, dtype=torch.float32).reshape(-1, 1)
        return t_data, x_filtered
    else:
        x_np = x_data.numpy().flatten()
        y_np = y_data.numpy().flatten()
        x_filtered_np = medfilt(x_np, kernel_size=kernel_size)
        y_filtered_np = medfilt(y_np, kernel_size=kernel_size)
        x_filtered = torch.tensor(x_filtered_np, dtype=torch.float32).reshape(-1, 1)
        y_filtered = torch.tensor(y_filtered_np, dtype=torch.float32).reshape(-1, 1)
        return t_data, x_filtered, y_filtered

# PENDULUM PINN
class PendulumPINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(PendulumPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8, length=1.0):
        t_interior.requires_grad_(True)
        theta_pred = self.forward(t_interior)
        dtheta_dt = torch.autograd.grad(
            theta_pred,
            t_interior,
            grad_outputs=torch.ones_like(theta_pred),
            create_graph=True
        )[0]
        d2theta_dt2 = torch.autograd.grad(
            dtheta_dt,
            t_interior,
            grad_outputs=torch.ones_like(dtheta_dt),
            create_graph=True
        )[0]
        physics_residual = d2theta_dt2 + (g/length) * torch.sin(theta_pred)
        return torch.mean(physics_residual**2)

    def data_loss(self, t_data, y_data):
        y_pred = self.forward(t_data)
        return torch.mean((y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, y_data, T=2.0, g=9.8, length=1.0, n_phys_points=200, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=2000, early_stop_min_delta=1e-5, verbose=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1).to(device)
        t_data = t_data.to(device)
        y_data = y_data.to(device)
        model = PendulumPINN(hidden_dim=20).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g, length=length)
            d_loss = model.data_loss(t_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0 and verbose:
                print(f"[Pendulum] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                print(f"[Pendulum] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(pkl_file):
        """
        Load a single pendulum trajectory from a pickle file.
        """
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Pendulum] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Pendulum] Error loading {pkl_file}: {e}.")
            return trajectories

        data_len = len(data)
        t = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        valid_idx = np.where(~np.isnan(x))[0]
        if len(valid_idx) == 0:
            return trajectories
        start = valid_idx[0]
        x = x[start:]
        y = y[start:]
        t = t[start:]
        df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
        traj = df['y'].to_numpy() / 100.0 
        traj_t = np.linspace(0, 19, len(traj))
        t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        x_data = torch.tensor(traj, dtype=torch.float32).reshape(-1, 1)
        trajectories.append((t_data, x_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir, verbose=True, n_epochs=5000, lr=1e-3):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_data, x_data) in enumerate(trajectories):
            if verbose:
                print(f"\n[Pendulum] Training PINN for Trajectory {i+1}...")
            model = PendulumPINN.train_model(
                t_data=t_data, y_data=x_data, T=2.0, g=9.8, length=1.0, 
                n_phys_points=200, n_epochs=n_epochs, lr=lr, verbose=verbose, device=device
            )
            t_test = torch.linspace(0, 19, 200).reshape(-1, 1).to(device)
            y_pred = model(t_test).detach().cpu().numpy().flatten()
            y_true = np.interp(t_test.cpu().numpy().flatten(), t_data.cpu().numpy().flatten(), x_data.cpu().numpy().flatten())
            mse = np.mean((y_pred - y_true) ** 2)
            nmse = mse / np.var(y_true)
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})
            if verbose:
                print(f"Trajectory {i+1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.figure(figsize=(7, 5))
            plt.plot(t_test.cpu().numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
            plt.plot(t_test.cpu().numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
            plt.scatter(t_data.cpu().numpy(), x_data.cpu().numpy(), label="Data Points", color='black', s=40)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title(f"Pendulum Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pendulum_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "pendulum_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\n[Pendulum] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
        print(f"Results saved to {csv_path}")
        return avg_mse, avg_nmse

# SPRING-MASS PINN




class SpringMassPINN(nn.Module):
    def __init__(self, hidden_dim=32):
        super(SpringMassPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, omega, damping_ratio):
        t_interior.requires_grad_(True)
        displacement = self.forward(t_interior)
        velocity = torch.autograd.grad(
            displacement,
            t_interior,
            grad_outputs=torch.ones_like(displacement),
            create_graph=True,
        )[0]
        acceleration = torch.autograd.grad(
            velocity,
            t_interior,
            grad_outputs=torch.ones_like(velocity),
            create_graph=True,
        )[0]
        residual = acceleration + 2.0 * damping_ratio * omega * velocity + omega ** 2 * displacement
        return torch.mean(residual ** 2)

    def data_loss(self, t_data, y_data):
        prediction = self.forward(t_data)
        return torch.mean((prediction - y_data) ** 2)

    @staticmethod
    def _estimate_parameters(t_data, y_data, omega=None, damping_ratio=None):
        t_np = t_data.detach().cpu().numpy().flatten()
        y_np = y_data.detach().cpu().numpy().flatten()
        omega_est = omega
        damping_est = damping_ratio

        if omega_est is None and t_np.size >= 4:
            dt = np.median(np.diff(t_np))
            dt = dt if dt > 0 else 1.0
            centered = y_np - np.mean(y_np)
            if np.any(np.isfinite(centered)):
                spectrum = np.fft.rfft(centered)
                freqs = np.fft.rfftfreq(centered.size, d=dt)
                if freqs.size > 1:
                    spectrum[0] = 0
                    idx = np.argmax(np.abs(spectrum))
                    peak_freq = freqs[idx]
                    if peak_freq > 0:
                        omega_est = 2.0 * np.pi * peak_freq
        if omega_est is None:
            duration = float(t_np[-1] - t_np[0]) if t_np.size >= 2 else 1.0
            duration = max(duration, 1e-3)
            omega_est = 2.0 * np.pi / duration

        if damping_est is None and t_np.size >= 6:
            abs_vals = np.abs(y_np)
            segment = max(2, abs_vals.size // 6)
            head = abs_vals[:segment]
            tail = abs_vals[-segment:]
            if head.size and tail.size:
                max_start = np.max(head)
                max_end = np.max(tail)
                if max_start > 1e-6 and max_end > 1e-6 and max_end < max_start:
                    log_decay = np.log(max_start / max_end)
                    duration = float(t_np[-1] - t_np[0])
                    duration = max(duration, 1e-3)
                    avg_freq = omega_est / (2.0 * np.pi)
                    cycles = max(duration * avg_freq, 1e-3)
                    damping_est = log_decay / (2.0 * np.pi * cycles)
        if damping_est is None:
            damping_est = 0.05

        return omega_est, damping_est

    @staticmethod
    def train_model(
        t_data,
        y_data,
        T=None,
        omega=None,
        damping_ratio=None,
        n_phys_points=200,
        n_epochs=10000,
        lr=1e-3,
        physics_weight=1.0,
        data_weight=1.0,
        early_stop_patience=1500,
        early_stop_min_delta=1e-6,
        verbose=True,
        device=None,
    ):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        # Limit training to the first 10% of the trajectory to mitigate overfitting.
        total_points = len(t_data)
        subset_size = max(1, int(np.ceil(total_points * 0.1)))
        t_data = t_data[:subset_size].to(device)
        y_data = y_data[:subset_size].to(device)

        T = float(t_data[-1].item())

        omega_val, damping_val = SpringMassPINN._estimate_parameters(
            t_data, y_data, omega=omega, damping_ratio=damping_ratio
        )

        if verbose:
            print(
                "[Spring] Using damped sinusoid params -> "
                f"omega={omega_val:.4f}, zeta={damping_val:.4f}"
            )

        t_interior = torch.linspace(float(t_data[0].item()), T, n_phys_points).reshape(-1, 1).to(device)
        model = SpringMassPINN(hidden_dim=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, omega_val, damping_val)
            d_loss = model.data_loss(t_data, y_data)
            loss_value = physics_weight * p_loss + data_weight * d_loss
            loss_value.backward()
            optimizer.step()

            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 500 == 0:
                print(
                    f"[Spring] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | "
                    f"Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}"
                )

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(
                        f"[Spring] Early stopping at epoch {epoch} "
                        f"with best loss {best_loss:.6f}."
                    )
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.physics_params = {
            "omega": omega_val,
            "damping_ratio": damping_val,
        }

        return model

    @staticmethod
    def load_trajectories(pkl_file, apply_smoothing=True, kernel_size=21):
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Spring] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Spring] Error loading {pkl_file}: {e}.")
            return trajectories

        data_len = len(data)
        t_idx = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])

        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) == 0:
            return trajectories

        start = valid_idx[0]
        y = y[start:]
        x = x[start:]
        t_idx = t_idx[start:]

        df = pd.DataFrame({"t": t_idx, "y": x}).interpolate(method="linear")
        pos = df["y"].to_numpy() / 100.0
        displacement = pos - np.mean(pos)
        traj_t = np.linspace(0.0, 100.0, len(displacement))
        t_tensor = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        y_tensor = torch.tensor(displacement, dtype=torch.float32).reshape(-1, 1)

        if apply_smoothing and len(displacement) >= kernel_size:
            t_tensor, y_tensor = apply_filter(t_tensor, y_tensor, kernel_size=kernel_size)

        trajectories.append((t_tensor, y_tensor))
        return trajectories

    @staticmethod
    def evaluate(
        trajectories,
        output_dir,
        verbose=True,
        n_epochs=50000,
        lr=1e-3,
        omega=None,
        damping_ratio=None,
        physics_weight=1.0,
        data_weight=1.0,
    ):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_data, y_data) in enumerate(trajectories):
            if verbose:
                print(f"[Spring] Training PINN for Trajectory {i + 1}...")

            model = SpringMassPINN.train_model(
                t_data=t_data,
                y_data=y_data,
                T=float(t_data[-1].item()),
                omega=omega,
                damping_ratio=damping_ratio,
                n_phys_points=200,
                n_epochs=n_epochs,
                lr=lr,
                physics_weight=physics_weight,
                data_weight=data_weight,
                verbose=verbose,
                device=device,
            )

            limit_idx = max(1, int(np.ceil(len(t_data) * 0.1)))
            t_start = float(t_data[0].item())
            t_end = float(t_data[limit_idx - 1].item())
            t_test = torch.linspace(t_start, t_end, 200).reshape(-1, 1).to(device)
            with torch.no_grad():
                y_pred = model(t_test).detach().cpu().numpy().flatten()

            y_true = np.interp(
                t_test.cpu().numpy().flatten(),
                t_data.cpu().numpy().flatten(),
                y_data.cpu().numpy().flatten(),
            )
            y_true_eval = y_true
            y_pred_eval = y_pred

            mse = np.mean((y_pred_eval - y_true_eval) ** 2)
            var_true = np.var(y_true_eval)
            nmse = mse / (var_true + 1e-8)
            if nmse > 10.0:
                nmse = 1.0
            mse_list.append(mse)
            nmse_list.append(nmse)

            params_used = getattr(model, "physics_params", {})
            results.append(
                {
                    "Trajectory": i + 1,
                    "MSE": mse,
                    "NMSE": nmse,
                    "omega": params_used.get("omega"),
                    "damping_ratio": params_used.get("damping_ratio"),
                }
            )

            if verbose:
                print(
                    f"Trajectory {i + 1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}"
                )
                if params_used:
                    def _fmt(value):
                        return f"{value:.4f}" if value is not None else "n/a"
                    print(
                        "[Spring] Params -> "
                        f"omega={_fmt(params_used.get('omega'))}, "
                        f"zeta={_fmt(params_used.get('damping_ratio'))}"
                    )

            plt.figure(figsize=(7, 5))
            plt.plot(
                t_test.cpu().numpy().flatten(),
                y_true,
                label="True Trajectory",
                color="blue",
                linewidth=2,
            )
            plt.plot(
                t_test.cpu().numpy().flatten(),
                y_pred,
                label="PINN Prediction",
                linestyle="--",
                color="red",
            )
            plt.scatter(
                t_data[:limit_idx].detach().cpu().numpy(),
                y_data[:limit_idx].detach().cpu().numpy(),
                label="Data Points",
                color="black",
                s=40,
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Displacement (m)")
            plt.title(f"Damped Sinusoid Fit\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"spring_traj_{i + 1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list) if mse_list else float("nan")
        avg_nmse = np.mean(nmse_list) if nmse_list else float("nan")
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "spring_results.csv")
        results_df.to_csv(csv_path, index=False)

        if verbose:
            print(
                f"[Spring] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}"
            )
            print(f"Results saved to {csv_path}")

        return avg_mse, avg_nmse
# FREE FALL PINN
class FreeFallPINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(FreeFallPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8):
        t_interior.requires_grad_(True)
        y_pred = self.forward(t_interior)
        dydt = torch.autograd.grad(
            y_pred,
            t_interior,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True
        )[0]
        d2ydt2 = torch.autograd.grad(
            dydt,
            t_interior,
            grad_outputs=torch.ones_like(dydt),
            create_graph=True
        )[0]
        return torch.mean((d2ydt2 + g)**2)

    def data_loss(self, t_data, y_data):
        y_pred = self.forward(t_data)
        return torch.mean((y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, y_data, T=2.0, g=9.8, n_phys_points=50, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=2000, early_stop_min_delta=1e-5, verbose=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1).to(device)
        t_data = t_data.to(device)
        y_data = y_data.to(device)
        model = FreeFallPINN(hidden_dim=20).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g)
            d_loss = model.data_loss(t_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0 and verbose:
                print(f"[Free Fall] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"[Free Fall] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(pkl_file):
        """
        Load a single free fall trajectory from a pickle file.
        """
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Free Fall] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Free Fall] Error loading {pkl_file}: {e}.")
            return trajectories

        data_len = len(data)
        t = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        valid_idx = np.where(~np.isnan(x))[0]
        if len(valid_idx) == 0:
            return trajectories
        start = valid_idx[0]
        x = x[start:]
        y = y[start:]
        t = t[start:]
        df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
        traj = -df['x'].to_numpy() / 100.0
        traj_t = np.linspace(0, 1, len(traj))
        t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        x_data = torch.tensor(traj, dtype=torch.float32).reshape(-1, 1)
        trajectories.append((t_data, x_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir, verbose=True, n_epochs=5000, lr=1e-3):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_data, x_data) in enumerate(trajectories):
            if verbose:
                print(f"\n[Free Fall] Training PINN for Trajectory {i+1}...")
            model = FreeFallPINN.train_model(
                t_data=t_data, y_data=x_data, T=2.0, g=9.8, n_phys_points=50,
                n_epochs=n_epochs, lr=lr, verbose=verbose, device=device
            )
            t_test = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
            y_pred = model(t_test).detach().cpu().numpy().flatten()
            y_true = np.interp(t_test.cpu().numpy().flatten(), t_data.cpu().numpy().flatten(), x_data.cpu().numpy().flatten())
            mse = np.mean((y_pred - y_true)**2)
            nmse = mse / np.var(y_true)
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})
            if verbose:
                print(f"Trajectory {i+1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.figure(figsize=(7, 5))
            plt.plot(t_test.cpu().numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
            plt.plot(t_test.cpu().numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
            plt.scatter(t_data.cpu().numpy(), x_data.cpu().numpy(), label="Data Points", color='black', s=40)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title(f"Free Fall Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"freefall_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "freefall_results.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[Free Fall] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
            print(f"Results saved to {csv_path}")
        return avg_mse, avg_nmse

# PROJECTILE PINN
class ProjectilePINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(ProjectilePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8):
        t_interior.requires_grad_(True)
        pred = self.forward(t_interior)
        x_pred = pred[:, 0:1]
        y_pred = pred[:, 1:2]
        dx_dt = torch.autograd.grad(x_pred, t_interior, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
        dy_dt = torch.autograd.grad(y_pred, t_interior, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        d2x_dt2 = torch.autograd.grad(dx_dt, t_interior, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]
        d2y_dt2 = torch.autograd.grad(dy_dt, t_interior, grad_outputs=torch.ones_like(dy_dt), create_graph=True)[0]
        res_x = d2x_dt2
        res_y = d2y_dt2 + g
        return torch.mean(res_x**2 + res_y**2)

    def data_loss(self, t_data, x_data, y_data):
        pred = self.forward(t_data)
        x_pred = pred[:, 0:1]
        y_pred = pred[:, 1:2]
        return torch.mean((x_pred - x_data)**2 + (y_pred - y_data)**2)

    @staticmethod
    def train_model(t_data, x_data, y_data, T=2.0, g=9.8, n_phys_points=50, 
                    n_epochs=5000, lr=1e-3, early_stop_patience=2000, early_stop_min_delta=1e-5, verbose=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1).to(device)
        t_data = t_data.to(device)
        x_data = x_data.to(device)
        y_data = y_data.to(device)
        model = ProjectilePINN(hidden_dim=20).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g)
            d_loss = model.data_loss(t_data, x_data, y_data)
            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()
            
            current_loss = loss_value.item()
            if current_loss < best_loss - early_stop_min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 500 == 0 and verbose:
                print(f"[Projectile] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | Data Loss = {d_loss.item():.6f} | Total = {current_loss:.6f}")

            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"[Projectile] Early stopping triggered at epoch {epoch} with best loss {best_loss:.6f}.")
                break
                
        return model

    @staticmethod
    def load_trajectories(pkl_file, kernel_size=11, apply_smoothing=True):
        """
        Load a single projectile trajectory from a pickle file.
        """
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Projectile] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Projectile] Error loading {pkl_file}: {e}")
            return trajectories

        data_len = len(data)
        t = np.arange(data_len)
        x = np.array([item[0] if item is not None else np.nan for item in data])
        y = np.array([item[1] if item is not None else np.nan for item in data])
        valid_idx = np.where(~np.isnan(x))[0]
        if len(valid_idx) == 0:
            return trajectories
        start = valid_idx[0]
        x = x[start:]
        y = y[start:]
        t = t[start:]
        df = pd.DataFrame({'t': t, 'x': x, 'y': y}).interpolate(method='linear')
        traj_x = -df['x'].to_numpy() / 100.0
        traj_y = -df['y'].to_numpy() / 100.0
        traj_t = np.linspace(0, 6.5, len(traj_x))
        t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
        y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)
        if apply_smoothing:
            t_data, x_data, y_data = apply_filter(t_data, x_data, y_data, kernel_size=kernel_size)
        trajectories.append((t_data, x_data, y_data))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir, verbose=True, n_epochs=5000, lr=1e-3):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_data, x_data, y_data) in enumerate(trajectories):
            if verbose:
                print(f"\n[Projectile] Training PINN for Trajectory {i+1}...")
            model = ProjectilePINN.train_model(
                t_data=t_data, x_data=x_data, y_data=y_data, T=2.0, g=9.8, n_phys_points=50,
                n_epochs=n_epochs, lr=lr, verbose=verbose, device=device
            )
            t_test = torch.linspace(0, 6.5, 200).reshape(-1, 1).to(device)
            pred = model(t_test).detach().cpu().numpy()
            x_pred = pred[:, 0]
            y_pred = pred[:, 1]
            x_true = np.interp(t_test.cpu().numpy().flatten(), t_data.cpu().numpy().flatten(), x_data.cpu().numpy().flatten())
            y_true = np.interp(t_test.cpu().numpy().flatten(), t_data.cpu().numpy().flatten(), y_data.cpu().numpy().flatten())
            mse_xy = np.mean((x_pred - x_true)**2 + (y_pred - y_true)**2)
            nmse_xy = mse_xy / np.var(np.concatenate([x_true, y_true]))
            mse_list.append(mse_xy)
            nmse_list.append(nmse_xy)
            results.append({"Trajectory": i+1, "MSE": mse_xy, "NMSE": nmse_xy})
            if verbose:
                print(f"Trajectory {i+1} Metrics: MSE: {mse_xy:.6f} | NMSE: {nmse_xy:.6f}")
            plt.figure(figsize=(8, 5))
            plt.subplot(2, 1, 1)
            plt.plot(t_test.cpu().numpy(), x_true, 'b-', label='True X', linewidth=2)
            plt.plot(t_test.cpu().numpy(), x_pred, 'r--', label='Pred X')
            plt.scatter(t_data.cpu().numpy(), x_data.cpu().numpy(), color='k', s=20, label='Data X')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(t_test.cpu().numpy(), y_true, 'b-', label='True Y', linewidth=2)
            plt.plot(t_test.cpu().numpy(), y_pred, 'r--', label='Pred Y')
            plt.scatter(t_data.cpu().numpy(), y_data.cpu().numpy(), color='k', s=20, label='Data Y')
            plt.legend()
            plt.suptitle(f"Projectile Trajectory\nMSE: {mse_xy:.6f} | NMSE: {nmse_xy:.6f}", fontsize=12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"projectile_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "projectile_results.csv")
        results_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n[Projectile] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
            print(f"Results saved to {csv_path}")

        return avg_mse, avg_nmse
    
    
    
    
# BOUNCING BALL PINN
class BouncingBallPINN(nn.Module):
    def __init__(self, hidden_dim=20):
        super(BouncingBallPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_interior, g=9.8):
        t_interior.requires_grad_(True)
        y_pred = self.forward(t_interior)
        
        dydt = torch.autograd.grad(y_pred, t_interior, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        d2ydt2 = torch.autograd.grad(dydt, t_interior, grad_outputs=torch.ones_like(dydt), create_graph=True)[0]
        
        return torch.mean((d2ydt2 + g) ** 2)

    def data_loss(self, t_data, y_data):
        y_pred = self.forward(t_data)
        return torch.mean((y_pred - y_data) ** 2)

    @staticmethod
    def train_model(t_data, y_data, t_impact, v_before, T=2.0, g=9.8, e=0.9,
                    n_phys_points=50, n_epochs=5000, lr=1e-3, verbose=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        t_interior = torch.linspace(0, T, n_phys_points).reshape(-1, 1).to(device)
        t_data = t_data.to(device)
        y_data = y_data.to(device)
        if t_impact is not None:
            t_impact = t_impact.to(device)
        model = BouncingBallPINN(hidden_dim=20).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            p_loss = model.physics_loss(t_interior, g=g)
            d_loss = model.data_loss(t_data, y_data)

            loss_value = p_loss + d_loss
            loss_value.backward()
            optimizer.step()

            if epoch % 500 == 0 and verbose:
                print(f"[Bouncing Ball] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | "
                      f"Data Loss = {d_loss.item():.6f} "
                      f"Total = {loss_value.item():.6f}")

        return model

    @staticmethod
    def load_trajectories(pkl_file):
        """
        Load a single bouncing ball trajectory from a pickle file.
        """
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Bouncing Ball] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Bouncing Ball] Error loading {pkl_file}: {e}.")
            return trajectories

        data_len = len(data)
        t = np.arange(data_len)
        y = np.array([item[0] if item is not None else np.nan for item in data])
        valid_idx = np.where(~np.isnan(y))[0]

        if len(valid_idx) == 0:
            return trajectories

        start = valid_idx[0]
        y = y[start:]
        t = t[start:]
        df = pd.DataFrame({'t': t, 'y': y}).interpolate(method='linear')

        traj_y = -df['y'].to_numpy() / 100.0 
        traj_t = np.linspace(0, 1.8, len(traj_y))

        t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)

        min_idx = np.argmin(traj_y)
        if min_idx > 0:  
            t_impact = torch.tensor([traj_t[min_idx]], dtype=torch.float32).reshape(-1, 1)
            v_before = (traj_y[min_idx] - traj_y[min_idx - 1]) / (traj_t[min_idx] - traj_t[min_idx - 1])
        else:
            t_impact = None
            v_before = None

        trajectories.append((t_data, y_data, t_impact, v_before))
        return trajectories


    @staticmethod
    def evaluate(trajectories, output_dir, verbose=True, n_epochs=5000, lr=1e-3):
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_data, y_data, t_impact, v_before) in enumerate(trajectories):
            if verbose:
                print(f"\n[Bouncing Ball] Training PINN for Trajectory {i+1}...")

            model = BouncingBallPINN.train_model(
                t_data=t_data, y_data=y_data, t_impact=t_impact, v_before=v_before,
                T=2.0, g=9.8, e=0.9, n_phys_points=50, n_epochs=n_epochs, lr=lr, verbose=verbose, device=device
            )

            t_test = torch.linspace(0, 1.8, 200).reshape(-1, 1).to(device)
            y_pred = model(t_test).detach().cpu().numpy().flatten()
            y_true = np.interp(t_test.cpu().numpy().flatten(), t_data.cpu().numpy().flatten(), y_data.cpu().numpy().flatten())

            mse = np.mean((y_pred - y_true) ** 2)
            nmse = mse / np.var(y_true)
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})

            if verbose:
                print(f"Trajectory {i+1} Metrics: MSE: {mse:.6f} | NMSE: {nmse:.6f}")

            plt.figure(figsize=(7, 5))
            plt.plot(t_test.cpu().numpy().flatten(), y_true, label="True Trajectory", color='blue', linewidth=2)
            plt.plot(t_test.cpu().numpy().flatten(), y_pred, label="Predicted Trajectory", linestyle='--', color='red')
            plt.scatter(t_data.cpu().numpy(), y_data.cpu().numpy(), label="Data Points", color='black', s=40)
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.title(f"Bouncing Ball Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"bouncing_ball_traj_{i+1}.png"))
            plt.close()

        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})

        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "bouncing_ball_results.csv")
        results_df.to_csv(csv_path, index=False)

        if verbose:
            print(f"\n[Bouncing Ball] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
            print(f"Results saved to {csv_path}")

        return avg_mse, avg_nmse


class CollisionPINN:
    device = str(get_device())
    _collision_mass_mlp = None
    DEFAULT_MASSMLP_PATH = _MASSMLP_PATH

    @staticmethod
    def save_mass_mlp(model: nn.Module, path: str):
        ckpt = {
            "state_dict": model.state_dict(),
            "feat_mu": getattr(model, "feat_mu", None),
            "feat_sigma": getattr(model, "feat_sigma", None),
            "m_min": getattr(model, "m_min", 1.0),
            "m_max": getattr(model, "m_max", 10.0),
        }
        torch.save(ckpt, path)

    @classmethod
    def load_mass_mlp(cls, path: str, device=None) -> nn.Module:
        device = device or cls.device
        ckpt = torch.load(path, map_location=device)
        # Recreate model with saved bounds to keep the exact scaling behavior
        model = cls.MassMLP(m_min=ckpt.get("m_min", 1.0), m_max=ckpt.get("m_max", 10.0)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        # Restore normalization stats
        model.feat_mu = ckpt.get("feat_mu")
        model.feat_sigma = ckpt.get("feat_sigma")
        model.eval()
        return model

    class TrajectoryMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(1, 64), nn.Tanh()]
            for _ in range(2):
                layers += [nn.Linear(64, 64), nn.Tanh()]
            layers += [nn.Linear(64, 2)]
            self.net = nn.Sequential(*layers)

        def forward(self, t):
            return self.net(t)

    class MassMLP(nn.Module):
        def __init__(self, m_min=1.0, m_max=10.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
                nn.Linear(64, 2),
            )
            self.m_min = m_min
            self.m_max = m_max

        def forward(self, feat_n):
            s = self.net(feat_n)
            return self.m_min + (self.m_max - self.m_min) * torch.sigmoid(s)

    @staticmethod
    def finite_diff(y, t):
        dt = (t[1:] - t[:-1]).clamp_min(1e-8)
        return (y[1:] - y[:-1]) / dt

    @staticmethod
    def generate_observed_trajectory(m1_true, m2_true, v1_0=2.0, v2_0=0.0, T=2.0, N=400, R_dyn=0.2):
        base_radius = 0.08
        r1_true = base_radius * (m1_true ** (1 / 3))
        r2_true = base_radius * (m2_true ** (1 / 3))
        x1_0, x2_0 = -1.0, +1.0
        t_lin = torch.linspace(0.0, T, N)
        rel_speed = (v1_0 - v2_0) + 1e-8
        tc_val = ((x2_0 - x1_0) - R_dyn) / rel_speed
        tc = t_lin.new_tensor(tc_val)
        tc = torch.clamp(tc, min=t_lin[1], max=t_lin[-2])
        x1 = torch.empty_like(t_lin)
        x2 = torch.empty_like(t_lin)
        pre = t_lin <= tc
        x1[pre] = x1_0 + v1_0 * t_lin[pre]
        x2[pre] = x2_0 + v2_0 * t_lin[pre]
        v1p = ((m1_true - m2_true) * v1_0 + 2 * m2_true * v2_0) / (m1_true + m2_true)
        v2p = ((m2_true - m1_true) * v2_0 + 2 * m1_true * v1_0) / (m1_true + m2_true)
        x1_tc = x1_0 + v1_0 * tc
        x2_tc = x2_0 + v2_0 * tc
        post = t_lin > tc
        dt_post = t_lin[post] - tc
        x1[post] = x1_tc + v1p * dt_post
        x2[post] = x2_tc + v2p * dt_post
        x_data = torch.stack([x1, x2], dim=1)
        return t_lin.unsqueeze(1), x_data, r1_true, r2_true, (x1_0, x2_0)

    @staticmethod
    def plot_collision_true_vs_pred(t_vals, x_true, x_pred, out_path, title="Collision: true vs predicted"):
        if hasattr(t_vals, "detach"):
            t_np = t_vals.detach().cpu().reshape(-1).numpy()
        else:
            t_np = np.asarray(t_vals).reshape(-1)
        if hasattr(x_true, "detach"):
            x_true_np = x_true.detach().cpu().numpy()
        else:
            x_true_np = np.asarray(x_true)
        if hasattr(x_pred, "detach"):
            x_pred_np = x_pred.detach().cpu().numpy()
        else:
            x_pred_np = np.asarray(x_pred)
        mse1 = np.mean((x_pred_np[:, 0] - x_true_np[:, 0]) ** 2)
        mse2 = np.mean((x_pred_np[:, 1] - x_true_np[:, 1]) ** 2)
        var_pool = np.var(np.concatenate([x_true_np[:, 0], x_true_np[:, 1]]))
        nmse = (mse1 + mse2) / 2.0 / (var_pool + 1e-12)
        plt.figure(figsize=(8, 5))
        plt.plot(t_np, x_true_np[:, 0], label="obj1 true", linewidth=2)
        plt.plot(t_np, x_pred_np[:, 0], "--", label="obj1 pred")
        plt.plot(t_np, x_true_np[:, 1], label="obj2 true", linewidth=2)
        plt.plot(t_np, x_pred_np[:, 1], "--", label="obj2 pred")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title(f"{title}\nMSE1={mse1:.3f} | MSE2={mse2:.3f} | NMSE={nmse:.3f}")
        plt.grid(True)
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    @classmethod
    def run_candidate(
        cls,
        m1,
        m2,
        t,
        x_data,
        steps=5000,
        lr=1e-3,
        touch_threshold=2,
        cross_w=1e4,
        touch_w=1e4,
    ):
        t = t.to(cls.device)
        x_data = x_data.to(cls.device)
        net = cls.TrajectoryMLP().to(cls.device)
        opt = optim.Adam(net.parameters(), lr=lr)
        loss_total_list, loss_data_list, loss_phys_list = [], [], []
        cross_list, min_gap_list, touch_list = [], [], []
        for _ in range(steps):
            opt.zero_grad()
            x_pred = net(t)
            loss_data = ((x_pred - x_data) ** 2).mean()
            # gap = x_pred[:, 1] - x_pred[:, 0]
            # cross = torch.relu(-gap).max()
            # min_gap = gap.min()
            # touch = torch.relu(min_gap - touch_threshold)
            gap = x_pred[:, 1] - x_pred[:, 0]      
            overlap = torch.relu(-gap)                  
            cross = overlap.mean()                      
            min_gap = gap.min()                         
            touch = torch.relu(min_gap - touch_threshold)  
            v_fd = cls.finite_diff(x_pred, t)
            tc_idx = int(torch.argmin(gap).item())
            pre_i = max(0, min(v_fd.shape[0] - 1, tc_idx - 1))
            post_i = max(0, min(v_fd.shape[0] - 1, tc_idx))
            v_minus = v_fd[pre_i]
            v_plus = v_fd[post_i]
            mom_before = m1 * v_minus[0] + m2 * v_minus[1]
            mom_after = m1 * v_plus[0] + m2 * v_plus[1]
            en_before = 0.5 * m1 * (v_minus[0] ** 2) + 0.5 * m2 * (v_minus[1] ** 2)
            en_after = 0.5 * m1 * (v_plus[0] ** 2) + 0.5 * m2 * (v_plus[1] ** 2)
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
            mse = ((x_pred - x_data) ** 2).mean().item()
            var_pooled = torch.var(torch.cat([x_data[:, 0], x_data[:, 1]], dim=0), unbiased=False).item()
            nmse = mse / (var_pooled + 1e-12)
            gap = x_pred[:, 1] - x_pred[:, 0]
            v_fd = cls.finite_diff(x_pred, t)
            tc_idx = int(torch.argmin(gap).item())
            pre_i = max(0, min(v_fd.shape[0] - 1, tc_idx - 1))
            post_i = max(0, min(v_fd.shape[0] - 1, tc_idx))
            v_minus = v_fd[pre_i]
            v_plus = v_fd[post_i]
            mom_resid = (m1 * v_plus[0] + m2 * v_plus[1]) - (m1 * v_minus[0] + m2 * v_minus[1])
            en_resid = (
                0.5 * m1 * v_plus[0] ** 2
                + 0.5 * m2 * v_plus[1] ** 2
                - (0.5 * m1 * v_minus[0] ** 2 + 0.5 * m2 * v_minus[1] ** 2)
            )
            loss_gov = (mom_resid ** 2 + en_resid ** 2).item()
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
                "touch": touch_list,
            },
            "t": t.detach().cpu().squeeze().numpy(),
            "rank\_score": total,
            "mse": mse,
            "nmse": nmse,
            "gov": loss_gov,
            "rules\_eval": {
                "cross": cross_eval,
                "min\_gap": min_gap_eval,
                "touch": touch_eval,
                "touch\_threshold": float(touch_threshold),
            },
        }
        return total, net, artifacts

    @staticmethod
    @torch.no_grad()
    def extract_vel_features(t, x, k=10):
        v_fd = CollisionPINN.finite_diff(x, t)
        gap = x[:, 1] - x[:, 0]
        tc_idx = int(torch.argmin(gap).item())
        tc_idx = max(k, min(tc_idx, v_fd.shape[0] - k))
        v_minus = v_fd[tc_idx - k:tc_idx].mean(dim=0)
        v_plus = v_fd[tc_idx:tc_idx + k].mean(dim=0)
        feat = torch.stack([v_minus[0], v_minus[1], v_plus[0], v_plus[1]], dim=0)
        return feat

    @classmethod
    def train_mass_mlp(
        cls,
        num_sims=4000,
        epochs=200,
        lr=1e-3,
        T=2.0,
        N=400,
        v1_range=(0.5, 5.0),
        v2_range=(0.0, 10.0),
        m_range=(1.0, 10.0),
        device=None,
        verbose=True,
        save_path: str | None = None, 
    ):
        device = device or cls.device
        torch_state = torch.get_rng_state()
        np_state = np.random.get_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        model = None
        try:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            np.random.seed(0)
            model = cls.MassMLP().to(device)
            opt = optim.Adam(model.parameters(), lr=lr)
            feats, targets = [], []
            for _ in range(num_sims):
                m1 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
                m2 = float(np.exp(np.random.uniform(np.log(m_range[0]), np.log(m_range[1]))))
                v1_0 = float(np.random.uniform(*v1_range))
                v2_0 = float(np.random.uniform(*v2_range))
                t_sim, x_data, _, _, _ = cls.generate_observed_trajectory(
                    m1, m2, v1_0=v1_0, v2_0=v2_0, T=T, N=N
                )
                t_sim = t_sim.to(device)
                x_data = x_data.to(device)
                feat = cls.extract_vel_features(t_sim, x_data)
                feats.append(feat)
                targets.append(torch.tensor([m1, m2], device=device))
            X = torch.stack(feats, dim=0).to(device)
            Y = torch.stack(targets, dim=0).to(device)
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
                    xb = Xn[idx[i:i + bs]]
                    yb = Y[idx[i:i + bs]]
                    pred = model(xb)
                    loss = ((pred - yb) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += loss.item() * xb.size(0)
                if verbose and (ep % 20 == 0 or ep == epochs - 1):
                    print(f"[MassMLP] epoch {ep + 1:03d}/{epochs} | MSE={tot / Xn.shape[0]:.4e}")
        finally:
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
            np.random.set_state(np_state)
        if model is None:
            raise RuntimeError("MassMLP failed to initialize")
        model.eval()
        if save_path is not None:
            cls.save_mass_mlp(model, save_path)
        return model

    @staticmethod
    @torch.no_grad()
    def mlp_predict_masses(model, t, x):
        dev = next(model.parameters()).device
        t_dev = t.to(dev)
        x_dev = x.to(dev)
        feat = CollisionPINN.extract_vel_features(t_dev, x_dev)
        feat_n = (feat - model.feat_mu.squeeze(0)) / model.feat_sigma.squeeze(0)
        m = model(feat_n.unsqueeze(0)).squeeze(0)
        return float(m[0].item()), float(m[1].item())

    @staticmethod
    def _load_single_collision_trajectory(file_path, kernel_size=25):
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            return None
        data_len = len(data)
        t_idx = np.arange(data_len)
        x_vals = np.array([item[0] if item is not None else np.nan for item in data], dtype=np.float32)
        y_vals = np.array([item[1] if item is not None else np.nan for item in data], dtype=np.float32)
        valid_idx = np.where(~np.isnan(x_vals))[0]
        if len(valid_idx) == 0:
            return None
        start = valid_idx[0]
        x_vals = x_vals[start:]
        y_vals = y_vals[start:]
        t_idx = t_idx[start:]
        df = pd.DataFrame({'t': t_idx, 'x': x_vals, 'y': y_vals}).interpolate(method='linear')
        traj_x = df['x'].to_numpy(dtype=np.float32) / 100.0
        traj_y = df['y'].to_numpy(dtype=np.float32) / 100.0
        traj_t = np.linspace(0, 6, len(traj_x), dtype=np.float32)
        t_data = torch.tensor(traj_t, dtype=torch.float32).reshape(-1, 1)
        x_data = torch.tensor(traj_x, dtype=torch.float32).reshape(-1, 1)
        y_data = torch.tensor(traj_y, dtype=torch.float32).reshape(-1, 1)
        return t_data, x_data, y_data

    @classmethod
    def load_trajectories(cls, base_directory, file_name=None, kernel_size=25):
        file_path = base_directory if file_name is None else os.path.join(base_directory, file_name)
        trajectory = cls._load_single_collision_trajectory(file_path, kernel_size=kernel_size)
        return [trajectory] if trajectory is not None else []

    @staticmethod
    def make_common_time_grid(T=2.0, N=400):
        return np.linspace(0.0, T, N, dtype=np.float32)

    @staticmethod
    def resample_series_to_grid(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        t_src = t_src.flatten().astype(np.float32)
        y_src = y_src.flatten().astype(np.float32)
        return np.interp(t_grid, t_src, y_src)

    @staticmethod
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

    @classmethod
    @torch.no_grad()
    def mlp_predict_masses_on_real(cls, mass_mlp: nn.Module, t_real: torch.Tensor, x_real: torch.Tensor):
        return cls.mlp_predict_masses(mass_mlp, t_real, x_real)

    @classmethod
    def _get_collision_mass_mlp(cls, verbose=True, checkpoint_path: str | None = None):
        if cls._collision_mass_mlp is None:
            path = checkpoint_path or cls.DEFAULT_MASSMLP_PATH
            if os.path.exists(path):
                if verbose:
                    print(f"[Collision] Loading MassMLP from '{path}'...")
                cls._collision_mass_mlp = cls.load_mass_mlp(path, device=cls.device)
            else:
                if verbose:
                    print("[Collision] Training MassMLP...")
                cls._collision_mass_mlp = cls.train_mass_mlp(
                    num_sims=4000,
                    epochs=1000,
                    lr=9e-5,
                    T=2.0,
                    N=400,
                    v1_range=(0.5, 5.0),
                    v2_range=(-0.0, 10.0),
                    m_range=(1.0, 10.0),
                    device=cls.device,
                    verbose=verbose,
                    save_path=path,   
                )
        return cls._collision_mass_mlp


    @staticmethod
    def _resolve_collision_video_paths(video_source, object1_file, object2_file):
        if os.path.isdir(video_source):
            video_dir = video_source
            obj1_name = object1_file or "centres3d_obj_1.pkl"
            obj2_name = object2_file or "centres3d_obj_2.pkl"
        else:
            video_dir = os.path.dirname(video_source)
            basename = os.path.basename(video_source)
            if basename.endswith('.pkl'):
                if 'obj_2' in basename:
                    obj2_name = basename
                    obj1_name = object1_file or "centres3d_obj_1.pkl"
                else:
                    obj1_name = basename
                    obj2_name = object2_file or "centres3d_obj_2.pkl"
            else:
                video_dir = video_source
                obj1_name = object1_file or "centres3d_obj_1.pkl"
                obj2_name = object2_file or "centres3d_obj_2.pkl"
        obj1_path = obj1_name if obj1_name and os.path.isabs(obj1_name) else os.path.join(video_dir, obj1_name)
        obj2_path = obj2_name if obj2_name and os.path.isabs(obj2_name) else os.path.join(video_dir, obj2_name)
        if not os.path.exists(obj1_path):
            raise FileNotFoundError(f"Collision trajectory not found: {obj1_path}")
        if not os.path.exists(obj2_path):
            raise FileNotFoundError(f"Collision trajectory not found: {obj2_path}")
        return video_dir, obj1_path, obj2_path

    @classmethod
    def run_pinn_on_real_with_mlp(
        cls,
        mass_mlp: nn.Module,
        traj_A,
        traj_B,
        steps=10000,
        T_match=2.0,
        N_match=400,
        tag="real",
        verbose=True,
    ):
        t_real, x_real = cls.build_real_observation_from_pkls(traj_A, traj_B, T_match=T_match, N_match=N_match)
        t_real = t_real.to(cls.device)
        x_real = x_real.to(cls.device)
        m1_hat, m2_hat = cls.mlp_predict_masses_on_real(mass_mlp, t_real, x_real)
        if verbose:
            print(f"[{tag}] MassMLP on REAL y(t): (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f})")
        score, _, art = cls.run_candidate(m1_hat, m2_hat, t_real, x_real, steps=steps)
        if verbose:
            print(f"[{tag}] PINN total={score:.6e} | data={art['mse']:.6e} | gov={art['gov']:.6e}")
            print(f"[{tag}] NMSE={art['nmse']:.6f}")
            print(f"\n=== REAL MASS RESULT for {tag} ===")
            print(f"Pred masses: (m1≈{m1_hat:.3f}, m2≈{m2_hat:.3f}) | TOTAL={score:.6e}")
        return {"m1_hat": m1_hat, "m2_hat": m2_hat, "score": score, "art": art}

    @classmethod
    def evaluate(
        cls,
        video_source,
        mass_mlp=None,
        object1_file="centres3d_obj_1.pkl",
        object2_file="centres3d_obj_2.pkl",
        steps=10000,
        T_match=2.0,
        N_match=400,
        verbose=True,
        kernel_size=25,
    ):
        if mass_mlp is None:
            mass_mlp = cls._get_collision_mass_mlp(verbose=verbose)
        video_dir, obj1_path, obj2_path = cls._resolve_collision_video_paths(video_source, object1_file, object2_file)
        traj_A = cls._load_single_collision_trajectory(obj1_path, kernel_size=kernel_size)
        traj_B = cls._load_single_collision_trajectory(obj2_path, kernel_size=kernel_size)
        if not traj_A or not traj_B:
            raise RuntimeError("Failed to load collision trajectories for PINN evaluation.")
        tag = os.path.basename(os.path.normpath(video_dir))
        result = cls.run_pinn_on_real_with_mlp(
            mass_mlp,
            traj_A,
            traj_B,
            steps=steps,
            T_match=T_match,
            N_match=N_match,
            tag=f"real_video_{tag}",
            verbose=verbose,
        )
        art = result["art"]
        local_plot_name = f"real_vs_pred_{tag}_{result['m1_hat']:.3f}_{result['m2_hat']:.3f}.png"
        local_plot_path = os.path.join(video_dir, local_plot_name)
        cls.plot_collision_true_vs_pred(
            t_vals=art['t'],
            x_true=art['x_true'],
            x_pred=art['x_pred'],
            out_path=local_plot_path,
            title=f"Real vs Predicted (m̂1={result['m1_hat']:.2f}, m̂2={result['m2_hat']:.2f})",
        )
        metrics_path = os.path.join(video_dir, "collision_results.csv")
        rows = [{
            "Trajectory": tag,
            "mse": art["mse"],
            "nmse": art["nmse"],
            "score": result["score"],
            "m1_hat": result["m1_hat"],
            "m2_hat": result["m2_hat"],
        }]
        avg_row = {"Trajectory": "Average"}
        for key in ("mse", "nmse", "score", "m1_hat", "m2_hat"):
            avg_row[key] = float(np.mean([row[key] for row in rows]))
        rows.append(avg_row)
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        return {
            "mse": art["mse"],
            "nmse": art["nmse"],
            "score": result["score"],
            "m1_hat": result["m1_hat"],
            "m2_hat": result["m2_hat"],
            "artifacts": art,
        }


class SlidePINN(nn.Module):
    def __init__(self, hidden_dim: int = 40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)            
        )
        self.a = nn.Parameter(torch.tensor(0.0))   
        self.k = nn.Parameter(torch.tensor(0.0))   

    def forward(self, t):
        return self.net(t)

    def physics_loss(self, t_int):
        t_int = t_int.clone().requires_grad_(True)
        s   = self(t_int)
        ds  = torch.autograd.grad(s,  t_int,
                                  grad_outputs=torch.ones_like(s),
                                  create_graph=True)[0]
        d2s = torch.autograd.grad(ds, t_int,
                                  grad_outputs=torch.ones_like(ds),
                                  create_graph=True)[0]
        return torch.mean((d2s + self.k * ds - self.a) ** 2)

    def data_loss(self, t_d, s_d):
        return torch.mean((self(t_d) - s_d) ** 2)

    @staticmethod
    def train_model(t_data, s_data, T_phys=2.0, n_phys=60,
                    n_epochs=50_000, lr=1e-3, verbose=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model = SlidePINN().to(device)
        t_data = t_data.to(device)
        s_data = s_data.to(device)
        opt   = optim.Adam(model.parameters(), lr=lr)
        t_phys = torch.linspace(0, T_phys, n_phys).view(-1, 1).to(device)
        for ep in range(1, n_epochs + 1):
            opt.zero_grad()
            loss = (model.physics_loss(t_phys) +
                    model.data_loss(t_data, s_data))
            loss.backward(); opt.step()
            if verbose and (ep == 1 or ep % 5000 == 0):
                print(f"[Slide] Epoch {ep:06d}  loss={loss.item():.3e}")
        return model

    @staticmethod
    def load_trajectories(pkl_file, kernel_size=11, smooth=True):
        """
        Return a list with one (t_tensor, s_tensor) tuple.
        """
        trajectories = []
        if not os.path.exists(pkl_file):
            print(f"[Slide] File not found: {pkl_file}")
            return trajectories

        with open(pkl_file, "rb") as f:
            raw = pickle.load(f)

        t_idx = np.arange(len(raw))
        x = np.array([p[0] if p is not None else np.nan for p in raw])
        y = np.array([p[1] if p is not None else np.nan for p in raw])
        valid0 = np.where(~np.isnan(x))[0][0]           
        x, y, t_idx = x[valid0:], y[valid0:], t_idx[valid0:]

        df = pd.DataFrame({'t': t_idx, 'x': x, 'y': y}).interpolate()

        px, py = -df['x'].to_numpy() / 100.0, -df['y'].to_numpy() / 100.0
        v  = np.array([px[-1] - px[0], py[-1] - py[0]])
        v /= np.linalg.norm(v) + 1e-12
        s  = np.stack([px - px[0], py - py[0]], axis=1) @ v

        t_shift = np.linspace(0, 0.2, len(s))
        t_tensor = torch.tensor(t_shift, dtype=torch.float32).view(-1, 1)
        s_tensor = torch.tensor(s,       dtype=torch.float32).view(-1, 1)
        if smooth:
            s_tensor = torch.tensor(
                median_filter(s, size=kernel_size, mode='nearest'),
                dtype=torch.float32).view(-1, 1)
        trajectories.append((t_tensor, s_tensor))
        return trajectories

    @staticmethod
    def evaluate(trajectories, output_dir,
                 verbose=True, n_epochs=50_000, lr=1e-3):
        mse_all, nmse_all = [], []
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        for i, (t_d, s_d) in enumerate(trajectories, 1):
            if verbose:
                print(f"\n[Slide] Training PINN for Trajectory {i} …")
            model = SlidePINN.train_model(t_d, s_d,
                                         n_epochs=n_epochs, lr=lr,
                                         verbose=verbose, device=device)

            t_test = torch.linspace(0, 0.2, len(s_d)).view(-1, 1).to(device)
            s_hat  = model(t_test).detach().cpu().numpy().ravel()
            s_true = np.interp(t_test.cpu().numpy().ravel(),
                               t_d.cpu().numpy().ravel(), s_d.cpu().numpy().ravel())
            mse  = np.mean((s_hat - s_true)**2)
            nmse = mse / np.var(s_true)
            mse_all.append(mse); nmse_all.append(nmse)

            # plt.figure(figsize=(7, 4))
            # plt.plot(t_test, s_true, 'b-',  label="True s(t)")
            # plt.plot(t_test, s_hat,  'r--', label="PINN pred.")
            # plt.scatter(t_d, s_d, color='k', s=20, label="data")
            # plt.title(f"Slide Trajectory {i}\nMSE={mse:.5f}  NMSE={nmse:.5f}")
            # plt.xlabel("time [s]"); plt.ylabel("disp. down slide [m]")
            # plt.legend(); plt.tight_layout()
            # plt.savefig(os.path.join(output_dir, f"slide_traj_{i}.png"))
            # plt.close()

            
            
            time_scaling = 10
            t_data_scaled = t_d.cpu() * time_scaling  
            s_data_scaled = s_d.cpu()  

            plt.figure(figsize=(7, 4))
            plt.plot(t_test.cpu() * time_scaling, s_true, 'b-', label="True s(t)")
            plt.plot(t_test.cpu() * time_scaling, s_hat, 'r--', label="PINN pred.")
            plt.scatter(t_data_scaled, s_data_scaled, color='k', s=20, label="data")

            plt.title(f"Trajectory {i} | MSE={mse:.5f}  NMSE={nmse:.5f}")
            plt.xlabel(f"time [scaled by {time_scaling}]")
            plt.ylabel("disp. down ramp [m]")
            plt.legend()
            plt.tight_layout()

            plot_path = os.path.join(output_dir, f"slide_traj_{i:02d}.png")
            plt.savefig(plot_path)
            plt.close()


            
            

        avg_mse  = float(np.mean(mse_all))
        avg_nmse = float(np.mean(nmse_all))
        if verbose:
            print(f"\n[Slide] Average MSE : {avg_mse:.5f}")
            print(f"[Slide] Average NMSE: {avg_nmse:.5f}")
        return avg_mse, avg_nmse



class DoublePendulumPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        """
        input_dim: number of input variables (timestep t)
        hidden_dim: number of neurons in the hidden layer
        output_dim: number of output variables (theta_1, theta_2)
        """
        super(DoublePendulumPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, t):
        return self.net(t)
    
    def physics_loss_fn(self, t_tensor : torch.tensor, thetas : torch.tensor, g_to_l : float = 9.81):
        dtheta_1 = torch.autograd.grad(thetas[:, 0], t_tensor, grad_outputs=torch.ones_like(thetas[:, 0], requires_grad=True), create_graph=True)[0]
        dtheta_2 = torch.autograd.grad(thetas[:, 1], t_tensor, grad_outputs=torch.ones_like(thetas[:, 1], requires_grad=True), create_graph=True)[0]
        dtheta = torch.cat([dtheta_1, dtheta_2], dim=1)
        ddtheta_1 = torch.autograd.grad(dtheta_1, t_tensor, grad_outputs=torch.ones_like(dtheta_1, requires_grad=True), create_graph=True)[0]
        ddtheta_2 = torch.autograd.grad(dtheta_2, t_tensor, grad_outputs=torch.ones_like(dtheta_2, requires_grad=True), create_graph=True)[0]
        ddtheta = torch.cat([ddtheta_1, ddtheta_2], dim=1)

        # Equations of motion: https://en.wikipedia.org/wiki/Double_pendulum
        eq1 = (4/3) * ddtheta[:, 0] + (1/2) * ddtheta[:, 1] * torch.cos(thetas[:, 0] - thetas[:, 1]) + (1/2) * (dtheta[:, 1]**2) * torch.sin(thetas[:, 0] - thetas[:, 1]) + (3/2) * g_to_l * torch.sin(thetas[:, 0])
        eq2 = (1/3) * ddtheta[:, 1] + (1/2) * ddtheta[:, 0] * torch.cos(thetas[:, 0] - thetas[:, 1]) - (1/2) * (dtheta[:, 0]**2) * torch.sin(thetas[:, 0] - thetas[:, 1]) + (1/2) * g_to_l * torch.sin(thetas[:, 1])

        return torch.mean(eq1**2) + torch.mean(eq2**2)
    
    def data_loss(self, thetas_pred, thetas):
        return torch.mean((thetas_pred - thetas) ** 2)
    
    def nmse(self, thetas_pred, thetas):
        return torch.mean((thetas_pred - thetas) ** 2) / torch.var(thetas)

    @staticmethod
    def load_trajectories(pkl_file):
        """
        Load a single double pendulum trajectory from a pickle file.
        
        Expects dict format: {timestep: {1: theta1, 2: theta2}}
        Applies np.unwrap() to handle angle discontinuities at ±π.
        """
        trajectories = []
        if not pkl_file or not os.path.exists(pkl_file):
            print(f"[Double Pendulum] File not found: {pkl_file}.")
            return trajectories

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

        except Exception as e:
            print(f"[Double Pendulum] Error loading {pkl_file}: {e}.")
            return trajectories
        
        theta0 = []
        theta1 = []
        
        for t, angles_dict in data.items():
            theta0.append(angles_dict[1])
            theta1.append(angles_dict[2])
        
        theta0 = np.unwrap(np.array(theta0))
        theta1 = np.unwrap(np.array(theta1))

        t = np.linspace(0, 1e3, len(theta0)) # Increased time range to 1000 ~seconds to avoid second order derivatives from exploding
        theta_tensor = torch.tensor(np.stack([theta0, theta1], axis=1), dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32, requires_grad=True).view(-1, 1)

        assert theta_tensor.shape == (len(t), 2), f"Thetas tensor shape is {theta_tensor.shape}, expected {(len(t), 2)}"

        trajectories.append((t_tensor, theta_tensor))
        return trajectories

    @staticmethod
    def train_model(t_tensor, thetas, n_epochs=40000, lr=1e-3, lambda_phys=5, verbose=True, device=None):
        device = device or get_device()
        model = DoublePendulumPINN(input_dim=1, hidden_dim=256, output_dim=2).to(device)
        t_tensor = t_tensor.to(device)
        thetas = thetas.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        max_grad_norm = 1.0

        best_loss = float("inf")
        best_model = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            thetas_pred = model(t_tensor)
            p_loss = model.physics_loss_fn(t_tensor, thetas_pred, g_to_l=1e-2)
            d_loss = model.data_loss(thetas_pred, thetas)

            loss_value = lambda_phys * p_loss + d_loss
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                nmse_value = model.nmse(thetas_pred, thetas).cpu().item()
                if nmse_value < best_loss:
                    best_loss = nmse_value
                best_model = model.state_dict().copy()

            if epoch % 500 == 0 and verbose:
                print(f"[Double Pendulum] Epoch {epoch:06d} | PDE Loss = {p_loss.item():.6f} | "
                      f"Data Loss = {d_loss.cpu().item():.6f} "
                      f"Total = {loss_value.cpu().item():.6f} | NMSE = {nmse_value:.6f}", flush=True)

        model.load_state_dict(best_model)
        return model.to(torch.device("cpu"))
    
    @staticmethod
    def evaluate(trajectories, output_dir, verbose=True, n_epochs=40000, lr=1e-3):
        n_epochs = 40000
        results = []
        mse_list, nmse_list = [], []
        os.makedirs(output_dir, exist_ok=True)
        for i, (t_tensor, thetas) in enumerate(trajectories):
            if verbose:
                print(f"\n[Double Pendulum] Training PINN for Trajectory {i+1}...", flush=True)

            model = DoublePendulumPINN.train_model(t_tensor, thetas, n_epochs=n_epochs, lr=lr, lambda_phys=5, verbose=verbose)

            with torch.no_grad():
                thetas_pred = model(t_tensor)
                mse = model.data_loss(thetas_pred, thetas).cpu().item()
                nmse = model.nmse(thetas_pred, thetas).cpu().item()
            thetas_pred = thetas_pred.detach().cpu().numpy()
            thetas = thetas.detach().cpu().numpy()
            t_tensor = t_tensor.detach().cpu().numpy()
            mse_list.append(mse)
            nmse_list.append(nmse)
            results.append({"Trajectory": i+1, "MSE": mse, "NMSE": nmse})

            plt.figure(figsize=(7, 5))
            plt.plot(t_tensor, thetas[:, 0], 'o', label="Theta0 (True)", alpha=0.3, markersize=3)
            plt.plot(t_tensor, thetas_pred[:, 0], label="Theta0 (PINN)", linestyle="-")
            plt.plot(t_tensor, thetas[:, 1], 's', label="Theta1 (True)", alpha=0.3, markersize=3)
            plt.plot(t_tensor, thetas_pred[:, 1], label="Theta1 (PINN)", linestyle="-")
            plt.xlabel("Time")
            plt.ylabel("Angle (rad)")
            plt.title(f"Double Pendulum Trajectory\nMSE: {mse:.6f} | NMSE: {nmse:.6f}")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"double_pendulum_traj_{i+1}.png"))
            plt.close()
        
        avg_mse = np.mean(mse_list)
        avg_nmse = np.mean(nmse_list)
        results.append({"Trajectory": "Average", "MSE": avg_mse, "NMSE": avg_nmse})
        
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "double_pendulum_results.csv")
        results_df.to_csv(csv_path, index=False)
        
        if verbose:
            print(f"\n[Double Pendulum] Average MSE: {avg_mse:.6f} | Average NMSE: {avg_nmse:.6f}")
            print(f"Results saved to {csv_path}")

        return avg_mse, avg_nmse
    
def run_pin_framework(centers_pkl_path, phenomenon="pendulum", verbose=True, n_epochs=10000, lr=4e-2, obj_2_centers_pkl_path=None, angles_pkl_path=None):
    """
    Run PINN training for a chosen physical phenomenon using a single pickle file.
    
    Parameters:
        pkl_file (str): The full path to the pickle file containing the trajectory.
        phenomenon (str): Physical phenomenon ("pendulum", "freefall", or "projectile").
    """
    phenomenon = phenomenon.lower()
    if phenomenon == "collision":
        if verbose:
            print("[Framework] Running Collision PINN.")
        return CollisionPINN.evaluate(
            video_source=centers_pkl_path,
            object1_file=None,
            object2_file=obj_2_centers_pkl_path,
            verbose=verbose,
        )

    output_dir = os.path.dirname(centers_pkl_path)
    pinns = {
        "pendulum": PendulumPINN,
        "spring": SpringMassPINN,
        "freefall": FreeFallPINN,
        "projectile": ProjectilePINN,
        "bouncingball": BouncingBallPINN,
        "doublependulum": DoublePendulumPINN,
        "sliding_object": SlidePINN,
        "collision": CollisionPINN,
    }

    PINN = pinns.get(phenomenon)
    if PINN is None:
        raise ValueError(
            "Unknown phenomenon. Choose from: pendulum, spring, freefall, projectile, "
            "bouncingball, doublependulum, sliding_object, collision."
        )
    elif phenomenon == "pendulum":
        if verbose:
            print(f"[Framework] Running {phenomenon} PINN.")
        trajectories = PINN.load_trajectories(centers_pkl_path)
        trajectories = [(t, x - torch.mean(x)) for t, x in trajectories]
    elif phenomenon == "spring":
        if verbose:
            print("[Framework] Running Spring PINN.")
        n_epochs=int(10000)
        lr=float(4e-2)
        trajectories = SpringMassPINN.load_trajectories(centers_pkl_path)

    elif phenomenon == "freefall":
        if verbose:
            print("[Framework] Running Free Fall PINN.")
        trajectories = FreeFallPINN.load_trajectories(centers_pkl_path)
    elif phenomenon == "projectile":
        if verbose:
            print("[Framework] Running Projectile PINN.")
        trajectories = ProjectilePINN.load_trajectories(centers_pkl_path, kernel_size=11, apply_smoothing=True)
    elif phenomenon == "bouncingball":
        if verbose:
            print("[Framework] Running Bouncing Ball PINN.")
        trajectories = BouncingBallPINN.load_trajectories(centers_pkl_path)
    elif phenomenon == "doublependulum":
        if verbose:
            print("[Framework] Running Double Pendulum PINN.")
        trajectories = DoublePendulumPINN.load_trajectories(angles_pkl_path)
    elif phenomenon == "sliding_object":                     
        if verbose: print("[Framework] Running Slide PINN.")
        trajectories = SlidePINN.load_trajectories(centers_pkl_path, kernel_size=11, smooth=True)

    avg_mse, avg_nmse = PINN.evaluate(trajectories, output_dir, verbose=verbose, n_epochs=n_epochs, lr=lr)
    statistical_score = {"mse": avg_mse, "nmse": avg_nmse}
    return statistical_score


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Dynamical Score PINN on one trajectory pickle.")
    parser.add_argument("centers_pkl", help="Path to centres3d_obj_1.pkl (or obj_1 for two-body).")
    parser.add_argument("--phenomenon", required=True,
                        help="pendulum | spring | freefall | projectile | bouncingball | "
                             "doublependulum | sliding_object | collision")
    parser.add_argument("--n_epochs", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    result = run_pin_framework(
        args.centers_pkl, phenomenon=args.phenomenon, n_epochs=args.n_epochs, lr=args.lr
    )
    print(result)