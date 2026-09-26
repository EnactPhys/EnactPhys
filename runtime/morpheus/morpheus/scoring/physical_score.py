import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import json
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class PhenomenonSpec:
    """Per-experiment recipe for the physical-invariance score.

    ``calculate_one_video_score`` used to re-test ``exp_name`` with hardcoded string
    comparisons at every one of its ~7 stages (pkl loading, hyper-parameters, energy,
    depth, stillness, acceleration, momentum/period/distance, aggregation). That
    branching is now data-driven: each canonical experiment maps to one
    ``PhenomenonSpec`` and every stage consults the spec instead of the raw name.

    The keys of :data:`PHYSICAL_SPECS` are the *canonical* experiment names produced by
    ``taxonomy.EXP_NAME_MAP`` (applied in ``combined.calculate_combined_score`` before
    this function runs) -- e.g. every ``rolling_*`` collapses to ``sliding_book`` and
    ``non_holonomic`` becomes the hyphenated ``non-holonomic_pendulum``. Do not rename
    these keys without updating ``EXP_NAME_MAP`` in lockstep -- a mismatch raises a
    ``KeyError`` here rather than silently mis-scoring.
    """

    n_obj: int                      # number of tracked objects (1 or 2)
    needs_distance_pkl: bool        # load the max-distance pickle (holonomic, double)
    needs_angle_pkl: bool           # load the per-frame angle pickle (double pendulum)
    scale: int                      # energy/velocity scale factor
    energy_mode: Optional[str]      # 'default' | 'collision' | 'double_pendulum' | None (skip energy)
    has_accel: bool                 # run acceleration-conservation score
    momentum_mode: Optional[str]    # 'single' | 'collision' | None
    has_period: bool                # run period-conservation score
    distance_mode: Optional[str]    # 'single' | 'two' | None
    rw_window_div: Optional[int]    # real-world time_window divisor (falls back to 4)
    rw_min_period: Optional[int]    # real-world min_period override (falls back to 10)
    aggregate: Tuple[str, ...]      # individual-score keys averaged into physical_score


#: Canonical experiment name -> scoring recipe. See :class:`PhenomenonSpec`.
#: The recipes below reproduce the original per-``exp_name`` branching exactly; the
#: only real-world-vs-generated divergence lives in ``rw_window_div`` / ``rw_min_period``.
PHYSICAL_SPECS = {
    # Ballistic single-object motion: energy + acceleration + horizontal momentum.
    "falling_ball": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=400,
        energy_mode="default", has_accel=True, momentum_mode="single",
        has_period=False, distance_mode=None, rw_window_div=10, rw_min_period=None,
        aggregate=("energy_conservation", "acceleration_conservation",
                   "horizontal_momentum_conservation"),
    ),
    "projectile": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=400,
        energy_mode="default", has_accel=True, momentum_mode="single",
        has_period=False, distance_mode=None, rw_window_div=10, rw_min_period=None,
        aggregate=("energy_conservation", "acceleration_conservation",
                   "horizontal_momentum_conservation"),
    ),
    "bouncing_ball": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=400,
        energy_mode="default", has_accel=True, momentum_mode="single",
        has_period=False, distance_mode=None, rw_window_div=10, rw_min_period=None,
        aggregate=("energy_conservation", "acceleration_conservation",
                   "horizontal_momentum_conservation"),
    ),
    # Sliding/rolling: acceleration only (energy is skipped). Absorbs all rolling_*.
    "sliding_book": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=1000,
        energy_mode=None, has_accel=True, momentum_mode=None,
        has_period=False, distance_mode=None, rw_window_div=None, rw_min_period=None,
        aggregate=("acceleration_conservation",),
    ),
    # Single pendulum: energy + period + string-length (distance) conservation.
    "holonomic_pendulum": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=True, needs_angle_pkl=False, scale=1000,
        energy_mode="default", has_accel=False, momentum_mode=None,
        has_period=True, distance_mode="single", rw_window_div=50, rw_min_period=35,
        aggregate=("energy_conservation", "period_conservation", "distance_conservation"),
    ),
    "non-holonomic_pendulum": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=1000,
        energy_mode="default", has_accel=False, momentum_mode=None,
        has_period=False, distance_mode=None, rw_window_div=50, rw_min_period=35,
        aggregate=("energy_conservation",),
    ),
    # Spring: energy runs to populate velocities, but only momentum + period are scored.
    "spring": PhenomenonSpec(
        n_obj=1, needs_distance_pkl=False, needs_angle_pkl=False, scale=1000,
        energy_mode="default", has_accel=False, momentum_mode="single",
        has_period=True, distance_mode=None, rw_window_div=None, rw_min_period=40,
        aggregate=("horizontal_momentum_conservation", "period_conservation"),
    ),
    # Two-object experiments.
    "double_pendulum": PhenomenonSpec(
        n_obj=2, needs_distance_pkl=True, needs_angle_pkl=True, scale=1000,
        energy_mode="double_pendulum", has_accel=False, momentum_mode=None,
        has_period=False, distance_mode="two", rw_window_div=50, rw_min_period=None,
        aggregate=("energy_conservation", "distance_conservation"),
    ),
    "collision": PhenomenonSpec(
        n_obj=2, needs_distance_pkl=False, needs_angle_pkl=False, scale=2000,
        energy_mode="collision", has_accel=False, momentum_mode="collision",
        has_period=False, distance_mode=None, rw_window_div=50, rw_min_period=None,
        aggregate=("energy_conservation", "horizontal_momentum_conservation"),
    ),
}


def process_centers(data):
    """
    Process and interpolate center data (x, y, z, and t) from the given list.
    Returns the arrays for x, y, z, t and the first valid index (first_value_idx)
    computed from the center data.
    """
    data_len = len(data)
    t = np.zeros((data_len, 1))
    x = np.zeros(data_len)
    y = np.zeros(data_len)
    z = np.zeros(data_len)
    first_value_flag = False
    first_value_idx = 0

    for i, item in enumerate(data):
        t[i] = i + 1
        # Check if the center data is missing
        if (isinstance(item[0], str) and item[0] == 'nan') or np.isnan(item[0]):
            # Find the next valid center value
            j = i
            while j < data_len:
                if (isinstance(data[j][0], str) and data[j][0] == 'nan') or np.isnan(data[j][0]):
                    j += 1
                else:
                    break
            # Find the previous valid center value
            k = i
            while k >= 0:
                if (isinstance(data[k][0], str) and data[k][0] == 'nan') or np.isnan(data[k][0]):
                    k -= 1
                else:
                    break

            if k >= 0 and j < data_len:
                prev_val = data[k]
                next_val = data[j]
                x[i] = float(prev_val[0]) + (float(next_val[0]) - float(prev_val[0])) / (j - k) * (i - k)
                y[i] = float(prev_val[1]) + (float(next_val[1]) - float(prev_val[1])) / (j - k) * (i - k)
                z[i] = float(prev_val[2]) + (float(next_val[2]) - float(prev_val[2])) / (j - k) * (i - k)
            else:
                if k >= 0:
                    x[i], y[i], z[i] = map(float, data[k])
                elif j < data_len:
                    x[i], y[i], z[i] = map(float, data[j])
        else:
            x[i] = float(item[0])
            y[i] = float(item[1])
            z[i] = float(item[2])
            if not first_value_flag:
                first_value_flag = True
                first_value_idx = i

    return x[first_value_idx:], y[first_value_idx:], z[first_value_idx:], t[first_value_idx:], first_value_idx

def process_distance(distance_data):
    """
    Process and interpolate distance data independently.
    Returns the distances array with missing values filled by linear interpolation.
    Missing values are represented by None.
    """
    data_len = len(distance_data)
    distances = np.zeros(data_len)

    for i in range(data_len):
        val = distance_data[i]
        # Check if this distance value is missing (None)
        if np.isnan(val):
            # Find the next valid distance value
            j = i
            while j < data_len:
                next_val = distance_data[j]
                if np.isnan(next_val):
                    j += 1
                else:
                    break
            # Find the previous valid distance value
            k = i
            while k >= 0:
                prev_val = distance_data[k]
                if np.isnan(prev_val):
                    k -= 1
                else:
                    break

            if k >= 0 and j < data_len:
                distances[i] = float(distance_data[k]) + (float(distance_data[j]) - float(distance_data[k])) / (j - k) * (i - k)
            elif k >= 0:
                distances[i] = float(distance_data[k])
            elif j < data_len:
                distances[i] = float(distance_data[j])
            else:
                distances[i] = np.nan
        else:
            try:
                distances[i] = float(val)
            except:
                distances[i] = np.nan

    return distances


def data_processing(data, distance_data=None, plot=False):
    """
    Process the given data by interpolating missing center values and, if provided,
    missing distance values separately.
    
    Returns:
      (x, y, z, t) if distance_data is None, otherwise (x, y, z, t, distances)
    
    The final arrays are sliced from the first valid center value, keeping the behavior
    of the original code.
    """
    x, y, z, t, first_value_idx = process_centers(data)
    
    if distance_data is not None:
        # Ensure data and distance_data have the same length
        assert len(data) == len(distance_data), f"Data and distance data lengths do not match: {len(data)} != {len(distance_data)}"
        distances = process_distance(distance_data)
        # Slice using the center data's first valid index to maintain consistent lengths
        distances = distances[first_value_idx:]
    
    if plot:
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        print("z shape:", z.shape)
        print("t shape:", t.shape)
        print("x:", x)

        # Plot x versus t
        plt.scatter(t, x, alpha=0.2)
        plt.xlabel("$t$")
        plt.ylabel("$X$")
    
    if distance_data is not None:
        return x, y, z, t, distances
    else:
        return x, y, z, t
    
def calculate_velocity(positions, times, window=5):
    """
    Enhanced velocity calculation using multiple techniques and adaptive windowing
    
    Args:
        positions: Position data array
        times: Time data array
        window: Base window size for calculations
    """
    # Ensure window is odd
    window = window + 1 if window % 2 == 0 else window

    velocities = np.zeros_like(positions)
    
    # 1. Initial noise reduction on position data
    # from scipy.signal import savgol_filter
    # positions_filtered = savgol_filter(positions, window, 3)
    
    # 2. Central difference for bulk calculation
    dt = np.diff(times)
    v_central = np.zeros_like(positions)
    v_central[1:-1] = (positions[2:] - positions[:-2]) / (times[2:] - times[:-2])
    # 3. Forward/backward difference for endpoints
    v_central[0] = (positions[1] - positions[0]) / dt[0]
    v_central[-1] = (positions[-1] - positions[-2]) / dt[-1]
    
    # 4. slope of the linear regression
    half_window = window // 2
    
    for i in range(len(positions)):
        
        start = max(0, i - half_window)
        end = min(len(positions), i + half_window + 1)
        
        # do linear regression
        t_window = times[start:end] - times[i]  # relative time
        p_window = positions[start:end]
        
        # least squares solution to a linear matrix equation
        A = np.vstack([t_window, np.ones_like(t_window)]).T
        slope, _ = np.linalg.lstsq(A, p_window, rcond=None)[0]
        
        velocities[i] = slope
    
    # # 5. Combine methods with weighted average
    alpha = 0.7  # weight for polynomial method
    velocities = alpha * velocities + (1-alpha) * v_central
    
    # 6. Final smoothing to remove any remaining noise
    velocities = savgol_filter(velocities, window, 3)
    
    return velocities

def calculate_angular_velocity(angles, times, window=5):
    """
    Calculate the angular velocity using central difference method.
    """
    window = window + 1 if window % 2 == 0 else window
    angular_velocities = np.zeros_like(angles)
    dt = np.diff(times)
    angular_velocities[1:-1] = (angles[2:] - angles[:-2]) / (times[2:] - times[:-2])
    angular_velocities[0] = (angles[1] - angles[0]) / dt[0]
    angular_velocities[-1] = (angles[-1] - angles[-2]) / dt[-1]

    half_window = window // 2
    for i in range(len(angles)):

        start = max(0, i - half_window)
        end = min(len(angles), i + half_window + 1)

        t_window = times[start:end] - times[i]  # relative time
        p_window = angles[start:end]

        A = np.vstack([t_window, np.ones_like(t_window)]).T
        slope, _ = np.linalg.lstsq(A, p_window, rcond=None)[0]

        angular_velocities[i] = slope
    
    alpha = 0.7  # weight for polynomial method
    angular_velocities = alpha * angular_velocities + (1-alpha) * angular_velocities

    angular_velocities = savgol_filter(angular_velocities, window, 3)
    
    return angular_velocities

def check_energy_conservation(trajectory, mass=1.0, g=9.81, window=21, scale=500):
    """
    Improved energy conservation checking with enhanced velocity calculation
    """
    times = trajectory[:, -1] / 100  # ms to s
    positions = trajectory[:, :3] / scale  # mm to m

    # Calculate velocities for each dimension
    vx = calculate_velocity(positions[:, 0], times, window)
    vy = calculate_velocity(positions[:, 1], times, window)
    vz = calculate_velocity(positions[:, 2], times, window)
    
    # Calculate energies
    v_squared = vx**2 + vy**2
    
    KE = 0.5 * mass * v_squared
    
    # Use maximum height as reference for potential energy
    h_ref = np.min(positions[:, 0])  # y-coordinate for height
    PE = - mass * g * (positions[:, 0] - h_ref)
    
    E_total = KE + PE
    
    # Calculate uncertainties
    ke_uncertainty = 0.5 * mass * np.std(v_squared) / np.sqrt(window)
    pe_uncertainty = mass * g * np.std(positions[:, 0]) / np.sqrt(window)
    
    return {
        'E_total': E_total,
        'KE': KE,
        'PE': PE,
        'velocities': {'vx': vx, 'vy': vy},
        'uncertainties': {
            'KE': ke_uncertainty,
            'PE': pe_uncertainty,
            'total': np.sqrt(ke_uncertainty**2 + pe_uncertainty**2)
        }
    }

def analyze_energy_conservation(trajectory, t_start, t_end, window=21, scale=500, plot=False, figure_name=""):
    """
    Enhanced analysis with uncertainty visualization and additional metrics
    """
    mask = (trajectory[:, -1] >= t_start) & (trajectory[:, -1] <= t_end)
    segment = trajectory[mask]
    
    results = check_energy_conservation(segment, window=window, scale=scale)
    
    # Plot with uncertainty bands
    # plt.figure(figsize=(15, 12))
    
    times = segment[:, -1]
    results['times'] = times
    uncertainty = results['uncertainties']['total']
    
    if plot:
        plt.subplot(311)
        plt.plot(times, results['E_total'], 'b-', label='Total Energy')
        plt.fill_between(times, 
                        results['E_total'] - 2*uncertainty,
                        results['E_total'] + 2*uncertainty,
                        alpha=0.2, color='b')
        plt.plot(times, results['KE'], 'g--', label='Kinetic Energy')
        plt.plot(times, results['PE'], 'r--', label='Potential Energy')
        plt.ylabel('Energy (J)')
        # plt.title('Energy Conservation Over Time'+ '\n' + figure_name)
        plt.legend()
        plt.grid(True)

    
    return results

def calculate_conservation_score_std(times, energy, t_start=30, t_end=80, normalization_factor=1.0):
    """
    Calculate energy conservation score using standard deviation method.
    
    Parameters:
    -----------
    time : array-like
        Time points
    energy : array-like
        Total energy values
    t_start : float
        Start time for analysis
    t_end : float
        End time for analysis
    normalization_factor : float
        Factor to adjust the sensitivity of the score to standard deviation
        Larger values make the score more sensitive to variations
        
    Returns:
    --------
    float
        Conservation score between 0 and 1
    dict
        Additional statistics
    """
    # Select data in time range
    mask = (times >= t_start) & (times <= t_end)
    energy_section = energy[mask]
    
    # Calculate statistics
    energy_mean = np.mean(energy_section)
    energy_std = np.std(energy_section)
    energy_max = np.max(energy_section)
    energy_min = np.min(energy_section)
    
    # Calculate relative standard deviation (coefficient of variation)
    # Adding a small number to avoid division by zero
    relative_std = energy_std / (abs(energy_mean) + 1e-10)
    
    # Calculate score
    # Using relative standard deviation ensures the score is scale-invariant
    if abs(energy_mean) >= 10 * abs(energy_std) or abs(energy_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * energy_std)
    
    # Additional statistics
    stats = {
        'mean': energy_mean,
        'std': energy_std,
        'relative_std': relative_std,
        'max': energy_max,
        'min': energy_min,
        'max_deviation': max(abs(energy_max - energy_mean), abs(energy_min - energy_mean))
    }
    
    return score, stats

def calculate_acceleration(v, times):
    """
    Calculate the acceleration in the x-direction using central difference method.
    
    Args:
        vz: Velocity data array in the z-direction
        times: Time data array
    
    Returns:
        Acceleration data array in the z-direction
    """
    # Initialize acceleration array
    a = np.zeros_like(v)
    
    # Calculate time differences
    dt = np.diff(times)
    
    # Central difference for bulk calculation
    a[1:-1] = (v[2:] - v[:-2]) / (times[2:] - times[:-2])
    
    # Forward/backward difference for endpoints
    a[0] = (v[1] - v[0]) / dt[0]
    a[-1] = (v[-1] - v[-2]) / dt[-1]
    
    return a

# Example usage:
# Assuming you have already calculated vz and times
# vz = calculate_velocity_improved_v3(positions[:, 2], times, window)
# az = calculate_acceleration_z(vz, times)

def check_acceleration(trajectory, window=21, scale=500):
    """
    Check the acceleration of the trajectory
    """
    times = trajectory[:, -1] / 100  # ms to s
    positions = trajectory[:, :3] / scale  # mm to m

    vx = calculate_velocity(positions[:, 0], times, window)
    vy = calculate_velocity(positions[:, 1], times, window)
    # vz = calculate_velocity(positions[:, 2], times, window)

    ax = calculate_acceleration(vx, times)
    ay = calculate_acceleration(vy, times)
    # az = calculate_acceleration(vz, times)

    return ax, ay, vx, vy

def analyze_acceleration(trajectory, 
                         t_start, 
                         t_end, 
                         window=21, 
                         scale=500, 
                         plot=False, 
                         figure_name=""):
    """
    Analyze the acceleration of the trajectory
    """
    mask = (trajectory[:, -1] >= t_start) & (trajectory[:, -1] <= t_end)
    segment = trajectory[mask]
    
    ax, ay, vx, vy = check_acceleration(segment, window=window, scale=scale)
    times = segment[:, -1]
    results = {
        'ax': ax,
        'ay': ay,
        'vx': vx,
        'vy': vy,
        'times': times
    }

    if plot:
        plt.subplot(312)
        plt.plot(times, ay, 'g-', label='Acceleration in x-direction')
        plt.plot(times, ax, 'b-', label='Acceleration in y-direction')
        # plt.plot(times, az, 'r-', label='Acceleration in z-direction')
        plt.ylabel('Acceleration (m/s^2)')
        # plt.title('Acceleration Over Time'+ '\n' + figure_name)
        plt.legend(loc='center left', bbox_to_anchor=(0, 0.8))
        plt.grid(True)
    
    return results

def calculate_acceleration_score(times, 
                                 acceleration, 
                                 t_start=30, 
                                 t_end=80, 
                                 normalization_factor=1.0):
    """
    Calculate the acceleration score using standard deviation method.
    """
    # Select data in time range
    mask = (times >= t_start) & (times <= t_end)
    acceleration_section = acceleration[mask]

    # Calculate statistics
    acceleration_mean = np.mean(acceleration_section)
    acceleration_std = np.std(acceleration_section)
    acceleration_max = np.max(acceleration_section)
    acceleration_min = np.min(acceleration_section)
    
    # Calculate relative standard deviation (coefficient of variation)
    # Adding a small number to avoid division by zero
    relative_std = acceleration_std / (abs(acceleration_mean) + 1e-10)
    
    # Calculate score
    # Using relative standard deviation ensures the score is scale-invariant
    if abs(acceleration_mean) >= 10 * abs(acceleration_std) or abs(acceleration_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * acceleration_std)

    # Additional statistics
    stats = {
        'mean': acceleration_mean,
        'std': acceleration_std,
        'relative_std': relative_std,
        'max': acceleration_max,
        'min': acceleration_min,
        'max_deviation': max(abs(acceleration_max - acceleration_mean), abs(acceleration_min - acceleration_mean))
    }
    
    return score, stats

def check_stillness(trajectory, window=5, v_threshold=0.01, t_start=30, t_end=80):
    """
    Check if the object is still within a certain range of time
    """
    mask = (trajectory[:, -1] >= t_start) & (trajectory[:, -1] <= t_end)
    segment = trajectory[mask]
    # Calculate velocity
    vx = calculate_velocity(segment[:, 0], segment[:, -1], window)
    vy = calculate_velocity(segment[:, 1], segment[:, -1], window)

    # Calculate the percentage of time the object is still
    stillness_percentage = np.mean((vx <= v_threshold) & (vy <= v_threshold)) * 100

    return stillness_percentage

    

def calculate_horizontal_momentum_score(velocity_y, 
                                         velocity_times, 
                                         t_start=30, 
                                         t_end=80, 
                                         normalization_factor=1.0):
    """
    Calculate the horizontal momentum score using standard deviation method.
    """

    time_mask = (velocity_times >= t_start) & (velocity_times <= t_end)

    segment = velocity_y[time_mask]

    if segment.shape[0] < 2:
        return -1, {}

    # Calculate statistics
    horizontal_momentum_mean = np.mean(segment)
    horizontal_momentum_std = np.std(segment)
    horizontal_momentum_max = np.max(segment)
    horizontal_momentum_min = np.min(segment)

    relative_std = horizontal_momentum_std / (abs(horizontal_momentum_mean) + 1e-10)    

    if abs(horizontal_momentum_mean) >= 10 * abs(horizontal_momentum_std) or abs(horizontal_momentum_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * horizontal_momentum_std)
    
    stats = {
        'mean': horizontal_momentum_mean,
        'std': horizontal_momentum_std,
        'relative_std': relative_std,
        'max': horizontal_momentum_max,
        'min': horizontal_momentum_min,
        'max_deviation': max(abs(horizontal_momentum_max - horizontal_momentum_mean), abs(horizontal_momentum_min - horizontal_momentum_mean))
    }
    
    return score, stats

import numpy as np
from scipy.signal import find_peaks, savgol_filter

def extract_periods(trajectory, 
                    window=5, 
                    times=None,
                    normalization_factor=1.0,
                    min_period=20):
    """
    Extract periods from trajectory data and calculate period consistency.
    
    Args:
        trajectory: The trajectory data array (e.g., x, y, or z component).
        window: Window size for smoothing the data.
    
    Returns:
        periods: List of detected periods.
        period_consistency: Standard deviation of the periods.
    """
    # Smooth the trajectory data
    smoothed_trajectory = savgol_filter(trajectory, window, 3)
    
    # Find peaks in the smoothed data
    peaks, _ = find_peaks(smoothed_trajectory)
    
    # Calculate periods as differences between consecutive peaks
    periods = np.diff(peaks)

    # filter out the periods that are too small
    periods = periods[periods > min_period]
    
    # Calculate period consistency as the standard deviation of the periods
    period_mean = np.mean(periods)
    period_std = np.std(periods)
    period_max = np.max(periods)
    period_min = np.min(periods)

    relative_std = period_std / (abs(period_mean) + 1e-10)
    
    if abs(period_mean) >= 10 * abs(period_std) or abs(period_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * period_std)


    stats = {
        'periods': periods,
        'times': times,
        'mean': period_mean,
        'std': period_std,
        'relative_std': relative_std,
        'max': period_max,
        'min': period_min,
        'max_deviation': max(abs(period_max - period_mean), abs(period_min - period_mean))
    }
    
    return score, stats

def calculate_distance_score(distances, 
                            times, 
                            t_start=30, 
                            t_end=80, 
                            normalization_factor=1.0):
    """
    Calculate the horizontal momentum score using standard deviation method.
    """
    time_mask = (times >= t_start) & (times <= t_end)
    segment = distances[time_mask]
    
    if segment.shape[0] < 2:
        return -1, {}

    # Calculate statistics
    distance_mean = np.mean(segment)
    distance_std = np.std(segment)
    distance_max = np.max(segment)
    distance_min = np.min(segment)

    relative_std = distance_std / (abs(distance_mean) + 1e-10)    

    if abs(distance_mean) >= 10 * abs(distance_std) or abs(distance_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * distance_std)
    
    stats = {
        'mean': distance_mean,
        'std': distance_std,
        'relative_std': relative_std,
        'max': distance_max,
        'min': distance_min,
        'max_deviation': max(abs(distance_max - distance_mean), abs(distance_min - distance_mean))
    }
    
    return score, stats

def calculate_depth_consistency_score(depths, 
                            times, 
                            scale=1000,
                            t_start=30, 
                            t_end=80, 
                            normalization_factor=1.0):
    """
    Calculate the depth consistency score using standard deviation method.
    """
    time_mask = (times >= t_start) & (times <= t_end)
    segment = depths[time_mask]
    segment = segment / scale
    depth_mean = np.mean(segment)
    depth_std = np.std(segment)
    depth_max = np.max(segment)
    depth_min = np.min(segment)

    relative_std = depth_std / (abs(depth_mean) + 1e-10)
    if abs(depth_mean) >= 10 * abs(depth_std) or abs(depth_mean) >= 10:
        score = 1 / (1 + normalization_factor * relative_std)
    else:
        score = 1 / (1 + normalization_factor * depth_std)

    stats = {
        'mean': depth_mean,
        'std': depth_std,
        'relative_std': relative_std,
        'max': depth_max,
        'min': depth_min,
        'max_deviation': max(abs(depth_max - depth_mean), abs(depth_min - depth_mean))
    }
    return score, stats

def check_double_pendulum_energy_conservation(trajectory1, 
                                               trajectory2, 
                                               angles1, 
                                               angles2, 
                                               distances1, 
                                               distances2, 
                                               mass=1.0, 
                                               g=9.81, 
                                               window=21, 
                                               scale=500):
    """
    Improved energy conservation checking with enhanced velocity calculation
    """
    times = trajectory1[:, -1] / 100  # ms to s
    positions1 = trajectory1[:, :3] / scale  # mm to m
    positions2 = trajectory2[:, :3] / scale  # mm to m
    
    # Calculate velocities for each dimension
    vx1 = calculate_velocity(positions1[:, 0], times, window)
    vy1 = calculate_velocity(positions1[:, 1], times, window)
    # vz1 = calculate_velocity(positions1[:, 2], times, window)
    
    vx2 = calculate_velocity(positions2[:, 0], times, window)
    vy2 = calculate_velocity(positions2[:, 1], times, window)
    # vz2 = calculate_velocity(positions2[:, 2], times, window)

    # Calculate angular velocities
    omega1 = calculate_angular_velocity(angles1, times, window)
    omega2 = calculate_angular_velocity(angles2, times, window)

    I1 = 1/12 * mass * distances1**2
    I2 = 1/12 * mass * distances2**2
    
    # Calculate energies
    v1_squared = vx1**2 + vy1**2 # + vz1**2
    v2_squared = vx2**2 + vy2**2 # + vz2**2
    
    KE1 = 0.5 * mass * v1_squared + 0.5 * I1 * omega1**2
    KE2 = 0.5 * mass * v2_squared + 0.5 * I2 * omega2**2
    
    # Use maximum height as reference for potential energy
    h_ref1 = np.min(positions1[:, 0])  # y-coordinate for height
    h_ref2 = np.min(positions2[:, 0])  # y-coordinate for height
    PE1 = - mass * g * (positions1[:, 0] - h_ref1)
    PE2 = - mass * g * (positions2[:, 0] - h_ref2)
    
    E_total = KE1 + PE1 + KE2 + PE2
    
    # Calculate uncertainties
    ke_uncertainty = 0.5 * mass * (np.std(v1_squared) / np.sqrt(window) + np.std(v2_squared) / np.sqrt(window))
    pe_uncertainty = mass * g * (np.std(positions1[:, 0]) / np.sqrt(window) + np.std(positions2[:, 0]) / np.sqrt(window))
    
    return {
        'E_total': E_total,
        'KE1': KE1,
        'PE1': PE1,
        'KE2': KE2,
        'PE2': PE2,
        'velocities': {'vx1': vx1, 'vy1': vy1, 'vx2': vx2, 'vy2': vy2},
        'uncertainties': {
            'KE': ke_uncertainty,
            'PE': pe_uncertainty,
            'total': np.sqrt(ke_uncertainty**2 + pe_uncertainty**2)
        }
    }

def analyze_double_pendulum_energy_conservation(trajectory1, 
                                                  trajectory2, 
                                                  distances1,
                                                  distances2,
                                                  angles1,
                                                  angles2,
                                                  t_start, 
                                                  t_end, 
                                                  window=21, 
                                                  scale=500, 
                                                  plot=False, 
                                                  figure_name=""):
    """
    Enhanced analysis with uncertainty visualization and additional metrics
    """
    
    results = check_double_pendulum_energy_conservation(trajectory1, 
                                                        trajectory2, 
                                                        angles1, 
                                                        angles2, 
                                                        distances1, 
                                                        distances2, 
                                                        window=window, 
                                                        scale=scale)
    
    
    times = trajectory1[:, -1]
    results['times'] = times
    uncertainty = results['uncertainties']['total']
    
    if plot:
        plt.subplot(311)
        plt.plot(times, results['E_total'], 'b-', label='Total Energy')
        plt.fill_between(times, 
                        results['E_total'] - 2*uncertainty,
                        results['E_total'] + 2*uncertainty,
                        alpha=0.2, color='b')
        plt.plot(times, results['KE1'], 'g--', label='Kinetic Energy 1')
        plt.plot(times, results['PE1'], 'r--', label='Potential Energy 1')
        plt.plot(times, results['KE2'], 'g--', label='Kinetic Energy 2')
        plt.plot(times, results['PE2'], 'r--', label='Potential Energy 2')
        plt.ylabel('Energy (J)')
        # plt.title('Energy Conservation Over Time'+ '\n' + figure_name)
        plt.legend()
        plt.grid(True)

    
    return results

def check_collision_energy_conservation(trajectory1, trajectory2, mass1=1.0, mass2=1.0, g=9.81, window=21, scale=500):
    """
    Improved energy conservation checking with enhanced velocity calculation
    """
    times = trajectory1[:, -1] / 100  # ms to s
    positions1 = trajectory1[:, :3] / scale  # mm to m
    positions2 = trajectory2[:, :3] / scale  # mm to m
    # print("position length:", len(positions))
    # print("time length:", len(times))
    
    # Calculate velocities for each dimension
    vx1 = calculate_velocity(positions1[:, 0], times, window)
    vy1 = calculate_velocity(positions1[:, 1], times, window)
    # vz1 = calculate_velocity(positions1[:, 2], times, window)
    
    vx2 = calculate_velocity(positions2[:, 0], times, window)
    vy2 = calculate_velocity(positions2[:, 1], times, window)
    # vz2 = calculate_velocity(positions2[:, 2], times, window)
    
    # Calculate energies
    v1_squared = vx1**2 + vy1**2 # + vz1**2
    v2_squared = vx2**2 + vy2**2 # + vz2**2
    
    KE1 = 0.5 * mass1 * v1_squared
    KE2 = 0.5 * mass2 * v2_squared
    
    # Use maximum height as reference for potential energy
    h_ref1 = np.min(positions1[:, 0])  # y-coordinate for height
    h_ref2 = np.min(positions2[:, 0])  # y-coordinate for height
    PE1 = - mass1 * g * (positions1[:, 0] - h_ref1)
    PE2 = - mass2 * g * (positions2[:, 0] - h_ref2)
    
    E_total = KE1 + PE1 + KE2 + PE2
    
    # Calculate uncertainties
    ke_uncertainty = 0.5 * mass1 * (np.std(v1_squared) / np.sqrt(window)) + 0.5 * mass2 * (np.std(v2_squared) / np.sqrt(window))
    pe_uncertainty = mass1 * g * (np.std(positions1[:, 0]) / np.sqrt(window)) + mass2 * g * (np.std(positions2[:, 0]) / np.sqrt(window))
    
    return {
        'E_total': E_total,
        'times': times,
        'KE1': KE1,
        'PE1': PE1,
        'KE2': KE2,
        'PE2': PE2,
        'velocities': {'vx1': vx1, 'vy1': vy1, 'vx2': vx2, 'vy2': vy2},
        'uncertainties': {
            'KE': ke_uncertainty,
            'PE': pe_uncertainty,
            'total': np.sqrt(ke_uncertainty**2 + pe_uncertainty**2)
        }
    }

def analyze_collision_energy_conservation(trajectory1, trajectory2, 
                                           mass1, mass2,
                                           t_start, t_end, 
                                           window=21, 
                                           scale=500, 
                                           plot=False, 
                                           figure_name=""):
    """
    Enhanced analysis with uncertainty visualization and additional metrics
    """
    mask = (trajectory1[:, -1] >= t_start) & (trajectory1[:, -1] <= t_end)
    segment1 = trajectory1[mask]
    segment2 = trajectory2[mask]
    
    results = check_collision_energy_conservation(segment1, segment2, mass1=1.0, mass2=1.0, window=window, scale=scale)
    
    # Plot with uncertainty bands
    # plt.figure(figsize=(15, 12))
    
    times = segment1[:, -1]
    results['times'] = times
    uncertainty = results['uncertainties']['total']
    
    if plot:
        plt.subplot(311)
        plt.plot(times, results['E_total'], 'b-', label='Total Energy')
        plt.fill_between(times, 
                        results['E_total'] - 2*uncertainty,
                        results['E_total'] + 2*uncertainty,
                        alpha=0.2, color='b')
        plt.plot(times, results['KE1'], 'g--', label='Kinetic Energy 1')
        plt.plot(times, results['PE1'], 'r--', label='Potential Energy 1')
        plt.plot(times, results['KE2'], 'g--', label='Kinetic Energy 2')
        plt.plot(times, results['PE2'], 'r--', label='Potential Energy 2')
        plt.ylabel('Energy (J)')
        # plt.title('Energy Conservation Over Time'+ '\n' + figure_name)
        plt.legend()
        plt.grid(True)

    
    return results

def calculate_horizontal_momentum_score_collision(velocity_y1, 
                                                  velocity_y2, 
                                                  mass1, 
                                                  mass2, 
                                                  velocity_times, 
                                                  t_start=30, 
                                                  t_end=80, 
                                                  normalization_factor=1.0):
    """
    Calculate the horizontal momentum score for collision experiments.
    """
    time_mask = (velocity_times >= t_start) & (velocity_times <= t_end)
    segment1 = velocity_y1[time_mask]
    segment2 = velocity_y2[time_mask]

    if segment1.shape[0] < 2 or segment2.shape[0] < 2:
        return -1, {}
    
    momentum1 = mass1 * segment1
    momentum2 = mass2 * segment2
    
    total_momentum = momentum1 + momentum2

    # Calculate statistics
    momentum_mean = np.mean(total_momentum)
    momentum_std = np.std(total_momentum)
    momentum_max = np.max(total_momentum)
    momentum_min = np.min(total_momentum)
    
    # Calculate score
    
    if abs(momentum_mean) >= 10 * abs(momentum_std) or abs(momentum_mean) >= 10:
        score = 1 / (1 + normalization_factor * momentum_std)
    else:
        score = 1 / (1 + normalization_factor * momentum_std)
    
    stats = {
        'momentum_mean': momentum_mean,
        'momentum_std': momentum_std,
        'momentum_max': momentum_max,
        'momentum_min': momentum_min
    }
    
    return score, stats
    

def predict_masses(base_dir, mass_mlp=None, T_match=2.0, N_match=400):
    """Predict the two object masses for a collision video from its trajectory pickles.

    ``base_dir`` must contain ``centres3d_obj_1.pkl`` and ``centres3d_obj_2.pkl``. The
    committed collision mass estimator ``collision_mass_mlp.pt`` (via ``CollisionPINN``,
    the same network the Dynamical score uses) then predicts ``(m1, m2)`` in one forward
    pass -- fast and deterministic, no per-video training. Returns a one-element list
    ``[{'m1_hat', 'm2_hat'}]``.

    If the pickles are not present in ``base_dir``, this returns equal masses ``(1.0,
    1.0)``. Note this is what the collision *Physical Invariance* numbers reported in the
    paper use: the pipeline passes the *input* video directory here, which holds the
    frames but not the trajectory pickles (those are written to the tracking-*output*
    tree), so the collision energy check is evaluated with equal masses. Pass a directory
    that contains the pickles to use estimated masses instead.
    """
    obj1_path = os.path.join(base_dir, "centres3d_obj_1.pkl")
    obj2_path = os.path.join(base_dir, "centres3d_obj_2.pkl")
    if not (os.path.exists(obj1_path) and os.path.exists(obj2_path)):
        return [{'m1_hat': 1.0, 'm2_hat': 1.0}]
    try:
        from .dynamical_score import CollisionPINN

        mlp = mass_mlp or CollisionPINN._get_collision_mass_mlp(verbose=False)
        traj_A = CollisionPINN._load_single_collision_trajectory(obj1_path)
        traj_B = CollisionPINN._load_single_collision_trajectory(obj2_path)
        if not traj_A or not traj_B:
            return [{'m1_hat': 1.0, 'm2_hat': 1.0}]
        t_real, x_real = CollisionPINN.build_real_observation_from_pkls(
            traj_A, traj_B, T_match=T_match, N_match=N_match
        )
        m1, m2 = CollisionPINN.mlp_predict_masses_on_real(mlp, t_real, x_real)
        return [{'m1_hat': m1, 'm2_hat': m2}]
    except Exception as e:  # noqa: BLE001 - fall back to equal masses if anything fails
        print(f"Note: collision mass estimation unavailable ({e}); using equal masses.")
        return [{'m1_hat': 1.0, 'm2_hat': 1.0}]


def calculate_one_video_score(centers_pkl_path, 
                              exp_name, 
                              category, 
                              figure_name="", 
                              stillness_penalty=False, 
                              obj_2_centers_pkl_path=None, 
                              distance_pkl_path=None,
                              angle_pkl_path=None,
                              videos_dir=None,
                              mass_mlp=None):
    
    spec = PHYSICAL_SPECS[exp_name]
    # Per-key physical sub-scores; the aggregate physical_score is the mean over
    # spec.aggregate (see the end of the function).
    computed = {}

    with open(centers_pkl_path, 'rb') as f_centers:
        data = pickle.load(f_centers)
        data_len = len(data)
    
    if spec.n_obj == 2 and spec.needs_angle_pkl:  # double_pendulum
        if obj_2_centers_pkl_path is not None:
            with open(obj_2_centers_pkl_path, 'rb') as f_obj2_centers:
                data_obj2 = pickle.load(f_obj2_centers)
        else:
            print("--------------------------------")
            print("obj_2_centers_pkl_path is None for double pendulum, skipping this trajectory")
            print("--------------------------------")
            results = {
                "individual_physical_scores": {},
                "physical_score": 0.0,
            }
            return results
        
        if angle_pkl_path is not None:
            with open(angle_pkl_path, 'rb') as f_angle:
                data_angle = pickle.load(f_angle)
        if distance_pkl_path is not None:
            with open(distance_pkl_path, 'rb') as f_distance:
                data_distance = pickle.load(f_distance)
            distance1 = data_distance[1]
            distance2 = data_distance[2]
        else:
            print("--------------------------------")
            print("distance_pkl_path is None for double pendulum, skipping this trajectory")
            print("--------------------------------")
            results = {
                "individual_physical_scores": {},
                "physical_score": 0.0,
            }
            return results

        x1, y1, z1, t1, distance1 = data_processing(data, distance1, plot=False)
        x2, y2, z2, t2, distance2 = data_processing(data_obj2, distance2, plot=False)
        t = t1
    elif spec.n_obj == 2:  # collision
        if obj_2_centers_pkl_path is not None:
            with open(obj_2_centers_pkl_path, 'rb') as f_obj2_centers:
                data_obj2 = pickle.load(f_obj2_centers)
        else:
            print("--------------------------------")
            print("obj_2_centers_pkl_path is None for collision, skipping this trajectory")
            print("--------------------------------")
            results = {
                "individual_physical_scores": {},
                "physical_score": 0.0,
            }
            return results
        x1, y1, z1, t1 = data_processing(data, plot=False)
        x2, y2, z2, t2 = data_processing(data_obj2, plot=False)
        t = t1
    elif spec.needs_distance_pkl:  # holonomic_pendulum
        if distance_pkl_path is not None:
            with open(distance_pkl_path, 'rb') as f_distance:
                distance_data = pickle.load(f_distance)
            distance = distance_data[2]
        else:
            print("--------------------------------")
            print("distance_pkl_path is None for holonomic pendulum, skipping this trajectory")
            print("--------------------------------")
            results = {
                "individual_physical_scores": {},
                "physical_score": 0.0,
            }
            return results
    
        # Data processing for the centers and distances
        x, y, z, t, distances = data_processing(data, distance_data=distance, plot=False)
    else:
        x, y, z, t = data_processing(data, plot=False)  

    if spec.n_obj == 2:
        t1 = t1.reshape(-1)
        t2 = t2.reshape(-1)
        # Keep the shared time vector one-dimensional for the generic window code
        # below. NumPy 2.x no longer allows int() on a one-element ndarray.
        t = t1
        trajectory1 = np.column_stack((x1, y1, z1, t1))
        trajectory2 = np.column_stack((x2, y2, z2, t2))
    else:
        t = t.reshape(-1)
        trajectory = np.column_stack((x, y, z, t))

    # if category == 'real_world_trajectories':
    #     if trajectory.shape[0] < 50:
    #         continue


    fps = 100
    dt = 1 / fps
    
    # t_end is different for each experiment

    ## Set hyperparameters
    # Andrii: Remove for new projectile videos
    t_start = int(t[0])
    t_end = int(t[-1])

    v_threshold = 0.01
    
    # Real-world trajectories use tighter scoring windows than generated ones -- this is
    # the ONLY real-world-vs-generated divergence in the scoring. ``rw_window_div`` encodes
    # the original per-experiment divisor (10 for ballistic, 50 for pendulum/collision);
    # everything else (incl. every generated video) uses //4.
    if category == 'real_world_trajectories' and spec.rw_window_div is not None:
        time_window = (t_end - t_start) // spec.rw_window_div
    else:
        time_window = (t_end - t_start) // 4
    time_window = max(time_window, 5)
    if time_window < 2:
        print("--------------------------------")
        print("time_window is too small, skipping this trajectory")
        print(f"trajectory length: {t.shape[0]}")
        # print(f"figure_name: {figure_name}")
        print(f"t_start: {t_start}")
        print(f"t_end: {t_end}")
        print(f"time_window: {time_window}")
        print("--------------------------------")
        results = {
            "individual_physical_scores": {},
            "physical_score": 0.0,
        }
        return results

    
    scale = spec.scale
    
    
    if category == 'real_world_trajectories' and spec.rw_min_period is not None:
        min_period = spec.rw_min_period
    else:
        min_period = 10
    min_distance = 0  # NOTE: computed but unused downstream (dead); preserved from original.
    
    
    ### Energy Conservation Score
    if spec.energy_mode is not None:  # energy is skipped for sliding_book / rolling
        if spec.energy_mode == 'double_pendulum':
            
            # Angles are stored as {timestep/frame_idx: {obj_id: angle}}.
            # These angles have already been cropped to the time range where both objects are tracked.
            # Therefore, we can use these timesteps to crop the trajectories and max distances accordingly,
            # as those are still in their raw (uncropped) form.
            common_timesteps = list(data_angle.keys())
            t_start = min(common_timesteps)
            t_end = max(common_timesteps) + 1
            
            mask = (trajectory1[:, -1] >= t_start) & (trajectory1[:, -1] <= t_end)
            trajectory1 = trajectory1[mask]
            trajectory2 = trajectory2[mask]
            distance1 = distance1[mask]
            distance2 = distance2[mask]
            
            angles1 = []
            angles2 = []
            
            for t, angles_dict in data_angle.items():
                angles1.append(angles_dict[1])
                angles2.append(angles_dict[2])
            
            angles1 = np.array(angles1)
            angles2 = np.array(angles2)
        
            energy_results = analyze_double_pendulum_energy_conservation(trajectory1, 
                                        trajectory2, 
                                        angles1=angles1,
                                        angles2=angles2,
                                        distances1=distance1,
                                        distances2=distance2,
                                        t_start=t_start, 
                                        t_end=t_end, 
                                        window=5, 
                                        scale=scale, 
                                        plot=False, 
                                        figure_name=figure_name)
        elif spec.energy_mode == 'collision':
            # NOTE: mass2_ratio is computed here but DISCARDED downstream --
            # analyze_collision_energy_conservation hardcodes mass1=mass2=1.0 internally,
            # so collision energy is scored with equal masses. This reproduces the paper's
            # reported collision numbers (see MORPHEUS_MIGRATION_NOTES.md 2.6); do not
            # "fix" it without re-baselining the collision scores.
            mass_results = predict_masses(base_dir=videos_dir, mass_mlp=mass_mlp)
            mass1 = mass_results[0]['m1_hat']
            mass2 = mass_results[0]['m2_hat']
            mass2_ratio = mass2 / mass1 if mass1 else 1.0

            energy_results = analyze_collision_energy_conservation(trajectory1, 
                                                                   trajectory2, 
                                                                   mass1=1.0, 
                                                                   mass2=mass2_ratio,
                                                                   t_start=t_start, 
                                                                   t_end=t_end, 
                                                                   window=5, 
                                                                   scale=scale, 
                                                                   plot=False, 
                                                                   figure_name=figure_name)
            
        else:   
            energy_results = analyze_energy_conservation(trajectory, 
                                        t_start=t_start, 
                                        t_end=t_end, 
                                        window=5, 
                                        scale=scale, 
                                        plot=False, 
                                        figure_name=figure_name)

        total_energy = energy_results['E_total']
        energy_times = energy_results['times']
        energy_score_list = []
        energy_stats_list = []
        for t in range(t_start, t_end - time_window, time_window//2):
            energy_score_t, energy_stats_t = calculate_conservation_score_std(energy_times, 
                                                                            total_energy, 
                                                                            t_start=t, 
                                                                            t_end=t+time_window)
            energy_score_list.append(energy_score_t)
            energy_stats_list.append(energy_stats_t)


        if len(energy_score_list) == 0:
            energy_score = 0.0
            energy_stats = {}
        else:
            energy_score = np.max(energy_score_list)
            max_index = np.argmax(energy_score_list)
            energy_stats = energy_stats_list[max_index]

        computed['energy_conservation'] = energy_score

    ### Depth Consistency Score
    if spec.n_obj == 2:
        depth_consistency_score_t1, depth_consistency_stats_t1 = calculate_depth_consistency_score(z1,
                                                                                    t,
                                                                                    scale=scale,
                                                                                    t_start=t_start,
                                                                                    t_end=t_end)
        depth_consistency_score_t2, depth_consistency_stats_t2 = calculate_depth_consistency_score(z2,
                                                                                    t,
                                                                                    scale=scale,
                                                                                    t_start=t_start,
                                                                                    t_end=t_end)
        depth_consistency_score_t = (depth_consistency_score_t1 + depth_consistency_score_t2) / 2
        depth_consistency_stats_t = {
            "depth_consistency_stats_t1": depth_consistency_stats_t1,
            "depth_consistency_stats_t2": depth_consistency_stats_t2,
        }
    else:   
        depth_consistency_score_t, depth_consistency_stats_t = calculate_depth_consistency_score(z,
                                                                                    t,
                                                                                    scale=scale,
                                                                                    t_start=t_start,
                                                                                    t_end=t_end)
    depth_consistency_score = depth_consistency_score_t
    depth_consistency_stats = depth_consistency_stats_t

    ### Check Stillness (return a percentage number)
    if spec.n_obj == 2:
        stillness_percentage1 = check_stillness(trajectory1,
                                           window=5,
                                           v_threshold=v_threshold, 
                                           t_start=t_start, 
                                           t_end=t_end)
        stillness_percentage2 = check_stillness(trajectory2, 
                                           window=5,
                                           v_threshold=v_threshold, 
                                           t_start=t_start, 
                                           t_end=t_end)
        stillness_percentage = (stillness_percentage1 + stillness_percentage2) / 2
    else:
        stillness_percentage = check_stillness(trajectory, 
                                           window=5,
                                           v_threshold=v_threshold, 
                                           t_start=t_start, 
                                           t_end=t_end)
    
    
    if spec.has_accel:
        ### Acceleration Conservation Score
        acceleration_results = analyze_acceleration(trajectory,
                                        t_start=t_start, 
                                        t_end=t_end, 
                                        window=5, 
                                        scale=scale, 
                                        plot=False, 
                                        figure_name=figure_name)
        
        ax = acceleration_results['ax']
        ay = acceleration_results['ay']
        # az = acceleration_results['az']
        vx_for_acceleration = acceleration_results['vx']
        vy_for_acceleration = acceleration_results['vy']
        # vz_for_acceleration = acceleration_results['vz']
        v_for_acceleration = np.sqrt(vx_for_acceleration**2 + vy_for_acceleration**2)
        acceleration_times = acceleration_results['times']
        acceleration_score_list = []
        acceleration_stats_list = []
        stillness_count = 0
        for t in range(t_start, t_end - time_window, time_window//2):
            acceleration_score_t, acceleration_stats_t = calculate_acceleration_score(acceleration_times, 
                                                                                      ax, 
                                                                                      t_start=t, 
                                                                                      t_end=t+time_window)
            acceleration_score_list.append(acceleration_score_t)
            acceleration_stats_list.append(acceleration_stats_t)
        if len(acceleration_score_list) == 0:
            acceleration_score = 0.0
            acceleration_stats = {}
        else:
            acceleration_score = np.max(acceleration_score_list)
            max_index = np.argmax(acceleration_score_list)
            acceleration_stats = acceleration_stats_list[max_index]
        computed['acceleration_conservation'] = acceleration_score


    # Horizontal momentum conservation. Ballistic + spring use the single-object score;
    # collision uses the two-body variant. These are independent (non-exclusive) blocks
    # now -- spring runs both this and the period block below, matching the original.
    if spec.momentum_mode == 'single':
        ### Horizontal Momentum Conservation Score
        velocity_y = energy_results['velocities']['vy']
        velocity_times = energy_results['times']
        # velocity_y_with_time = np.column_stack((velocity_y, velocity_times))
        horizontal_momentum_score_list = []
        horizontal_momentum_stats_list = []
        for t in range(t_start, t_end - time_window, time_window//2):
            horizontal_momentum_score_t, horizontal_momentum_stats_t = calculate_horizontal_momentum_score(velocity_y,
                                                                                                           velocity_times,
                                                                                                           t_start=t,
                                                                                                           t_end=t+time_window)
            # print("horizontal_momentum_score_t:", horizontal_momentum_score_t)
            if horizontal_momentum_score_t == -1:
                continue
            horizontal_momentum_score_list.append(horizontal_momentum_score_t)
            horizontal_momentum_stats_list.append(horizontal_momentum_stats_t)
        try:
            horizontal_momentum_score = np.max(horizontal_momentum_score_list)
            max_index = np.argmax(horizontal_momentum_score_list)
            horizontal_momentum_stats = horizontal_momentum_stats_list[max_index]
        except:
            horizontal_momentum_score = 0
            horizontal_momentum_stats = {}
        computed['horizontal_momentum_conservation'] = horizontal_momentum_score
    elif spec.momentum_mode == 'collision':
        ### Horizontal Momentum Conservation Score
        velocity_y1 = energy_results['velocities']['vy1']
        velocity_y2 = energy_results['velocities']['vy2']
        velocity_times = energy_results['times']
        horizontal_momentum_score_list = []
        horizontal_momentum_stats_list = []
        # NOTE: the two append() lines below are intentionally dedented OUT of the for-loop,
        # and the score call uses t_start/t_end (not the loop variable t), so only a single
        # window is ever scored. This quirk is preserved to reproduce the paper's collision
        # numbers; do not "fix" it without re-baselining.
        for t in range(t_start, t_end - time_window, time_window//2):
            horizontal_momentum_score_t, horizontal_momentum_stats_t = calculate_horizontal_momentum_score_collision(velocity_y1,
                                                                                                                     velocity_y2,
                                                                                                                     mass1=1.0,
                                                                                                                     mass2=1.0,
                                                                                                                     velocity_times=velocity_times,
                                                                                                                     t_start=t_start,
                                                                                                                     t_end=t_end)
        horizontal_momentum_score_list.append(horizontal_momentum_score_t)
        horizontal_momentum_stats_list.append(horizontal_momentum_stats_t)
        try:
            horizontal_momentum_score = np.max(horizontal_momentum_score_list)
            max_index = np.argmax(horizontal_momentum_score_list)
            horizontal_momentum_stats = horizontal_momentum_stats_list[max_index]
        except:
            horizontal_momentum_score = 0
            horizontal_momentum_stats = {}
        computed['horizontal_momentum_conservation'] = horizontal_momentum_score

    # Period conservation (spring + holonomic pendulum).
    if spec.has_period:
        ### Period Conservation Score
        try:
            trajectory_x = trajectory[:, 0]
            period_score, period_stats = extract_periods(trajectory_x,
                                                        window=5,
                                                        normalization_factor=1.0,
                                                        min_period=min_period)
        except:
            period_score = 0
            period_stats = {}
        computed['period_conservation'] = period_score

    # Distance (string-length) conservation: holonomic pendulum (1 object) and
    # double pendulum (mean over the 2 objects).
    if spec.distance_mode == 'single':
        distance_times =  energy_results['times']
        ### Distance Conservation Score
        distance_score, distance_stats = calculate_distance_score(distances,
                                                                 distance_times,
                                                                 t_start=t_start,
                                                                 t_end=t_end)
        computed['distance_conservation'] = distance_score
    elif spec.distance_mode == 'two':
        distance_times =  energy_results['times']
        distance_score1, distance_stats1 = calculate_distance_score(distance1,
                                                            distance_times,
                                                            t_start=t_start,
                                                            t_end=t_end)
        distance_score2, distance_stats2 = calculate_distance_score(distance2,
                                                            distance_times,
                                                            t_start=t_start,
                                                            t_end=t_end)

        total_distance_score = (distance_score1 + distance_score2) / 2
        total_distance_stats = {
            "distance1_stats": distance_stats1,
            "distance2_stats": distance_stats2,
        }
        computed['distance_conservation'] = total_distance_score


    results = {
        "individual_physical_scores": {},
        "physical_score": 0.0,
    }

    # depth_consistency and stillness are recorded for every experiment but excluded from
    # the aggregate physical_score (preserved from the original behaviour).
    results["individual_physical_scores"]["depth_consistency"] = depth_consistency_score
    results["individual_physical_scores"]["stillness"] = stillness_percentage

    # The aggregate is the mean of the sub-scores listed in spec.aggregate. This reproduces
    # every original per-experiment formula exactly, e.g. falling -> (E+A+H)/3,
    # sliding_book -> A, holonomic -> (E+P+D)/3, collision -> (E+H)/2.
    for score_key in spec.aggregate:
        results["individual_physical_scores"][score_key] = computed[score_key]
    aggregate_values = [computed[score_key] for score_key in spec.aggregate]
    results["physical_score"] = sum(aggregate_values) / len(aggregate_values)

    return results
