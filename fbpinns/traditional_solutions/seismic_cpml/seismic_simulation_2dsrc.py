
import numpy as np
import time
from fbpinns.traditional_solutions.seismic_cpml.seismic_CPML_helper import get_dampening_profiles

def run_seismic_simulation_src(
        grid_x, grid_y, time_steps, dx, dy, dt, pml_points,
        vel, rho, initial_pressure, dtype=np.float32,
        save_wavefields=True, gather_points=None, source=None, source_location=None):
    """
    Run 2D Seismic Simulation using Convolutional Perfectly Matched Layer (CPML)

    Parameters:
        grid_x, grid_y: Grid size in x and y directions
        time_steps: Number of time steps
        dx, dy: Grid spacing in x and y directions
        dt: Time step size
        pml_points: Number of points in the PML region
        vel: Velocity field
        rho: Density field
        initial_pressure: Tuple of initial pressures (past, present)
        dtype: Data type to use
        save_wavefields: Whether to save wavefields during simulation
        gather_points: Indices to gather time series from specific locations
        source: Function that provides source term (should accept time step and return scalar or 2D array)
                or a pre-calculated 3D array of shape (time_steps, grid_x, grid_y)
        source_location: Tuple (x, y) specifying the location of the source
    """

    # Convert input arrays to desired data type
    vel = vel.astype(dtype)
    rho = rho.astype(dtype)

    if gather_points is not None:
        gather_output = True
    else:
        gather_output = False

    # Constants for PML damping
    max_k = 1.0
    alpha_max = 2.0 * np.pi * (20.0 / 2.0)  # From Festa and Vilotte, using dominant frequency of 20 Hz
    power = 2.0
    reflection_coeff = 0.001
    stability_limit = 1e25

    # Stability Check
    courant = np.max(vel) * dt * np.sqrt(1 / (dx ** 2) + 1 / (dy ** 2))
    if courant > 1.0:
        raise Exception(f"ERROR: Unstable time step, Courant number is {courant:.2f}")

    # Initialize damping profiles
    damping_profiles = get_dampening_profiles(vel, pml_points, reflection_coeff, max_k, alpha_max, power, dt, (dx, dy), dtype)
    [a_x, a_x_half, b_x, b_x_half, k_x, k_x_half], [a_y, a_y_half, b_y, b_y_half, k_y, k_y_half] = damping_profiles

    # Initialize arrays for the simulation
    bulk_modulus = rho * (vel ** 2)
    pressure_curr = initial_pressure[1].astype(dtype)
    pressure_prev = initial_pressure[0].astype(dtype)

    # Memory arrays for derivatives in PML
    mem_dpressure_dx = np.zeros((grid_x, grid_y), dtype=dtype)
    mem_dpressure_dy = np.zeros((grid_x, grid_y), dtype=dtype)
    mem_dpressure_xx = np.zeros((grid_x, grid_y), dtype=dtype)
    mem_dpressure_yy = np.zeros((grid_x, grid_y), dtype=dtype)

    # Output arrays
    if save_wavefields:
        wavefields = np.zeros((time_steps, grid_x, grid_y), dtype=dtype)
    if gather_output:
        gather = np.zeros((gather_points.shape[0], time_steps), dtype=dtype)

    # Density at half-grid points
    rho_half_x = np.pad(0.5 * (rho[1:grid_x, :] + rho[:grid_x - 1, :]), [[0, 1], [0, 0]], mode="edge")
    rho_half_y = np.pad(0.5 * (rho[:, 1:grid_y] + rho[:, :grid_y - 1]), [[0, 0], [0, 1]], mode="edge")

    # Run simulation
    start_time = time.time()
    if source_location is None:
        source_x, source_y = grid_x // 2, grid_y // 2  # Default source location at the center of the grid
    else:
        source_x, source_y = source_location

    for t in range(time_steps):
        # Compute first-order spatial derivatives
        dpressure_dx = np.pad((pressure_curr[1:grid_x, :] - pressure_curr[:grid_x - 1, :]) / dx, [[0, 1], [0, 0]], mode="constant")
        dpressure_dy = np.pad((pressure_curr[:, 1:grid_y] - pressure_curr[:, :grid_y - 1]) / dy, [[0, 0], [0, 1]], mode="constant")

        # Apply PML damping
        mem_dpressure_dx = b_x_half * mem_dpressure_dx + a_x_half * dpressure_dx
        mem_dpressure_dy = b_y_half * mem_dpressure_dy + a_y_half * dpressure_dy

        dpressure_dx = dpressure_dx / k_x_half + mem_dpressure_dx
        dpressure_dy = dpressure_dy / k_y_half + mem_dpressure_dy

        # Divide by density at half points
        pressure_xx = dpressure_dx / rho_half_x
        pressure_yy = dpressure_dy / rho_half_y

        # Compute second-order spatial derivatives
        dpressure_xx_dx = np.pad((pressure_xx[1:grid_x, :] - pressure_xx[:grid_x - 1, :]) / dx, [[1, 0], [0, 0]], mode="constant")
        dpressure_yy_dy = np.pad((pressure_yy[:, 1:grid_y] - pressure_yy[:, :grid_y - 1]) / dy, [[0, 0], [1, 0]], mode="constant")

        # Apply PML damping to second derivatives
        mem_dpressure_xx = b_x * mem_dpressure_xx + a_x * dpressure_xx_dx
        mem_dpressure_yy = b_y * mem_dpressure_yy + a_y * dpressure_yy_dy

        dpressure_xx_dx = dpressure_xx_dx / k_x + mem_dpressure_xx
        dpressure_yy_dy = dpressure_yy_dy / k_y + mem_dpressure_yy

        # Time evolution scheme
        pressure_next = -pressure_prev + 2 * pressure_curr + dt ** 2 * (dpressure_xx_dx + dpressure_yy_dy) * bulk_modulus

        # Inject source term if provided
        if source is not None:
            if callable(source):
                # If source is a function, use it
                pressure_next[source_x,source_y] += source(t) /(dx * dy) * dt ** 2
            else:
                # If source is an array, use the precomputed value for this time step
                pressure_next += source[t] /(dx * dy) * dt ** 2
    
        # Apply Dirichlet boundary conditions
        pressure_next[0, :] = pressure_next[-1, :] = 0.0
        pressure_next[:, 0] = pressure_next[:, -1] = 0.0

        # Save outputs if required
        if save_wavefields:
            wavefields[t, :, :] = pressure_curr
        if gather_output:
            gather[:, t] = pressure_curr[gather_points[:, 0], gather_points[:, 1]]

        # Stability check
        if np.max(np.abs(pressure_curr)) > stability_limit:
            raise Exception('Simulation became unstable')

        # Update time steps
        pressure_prev, pressure_curr = pressure_curr, pressure_next

        # Print progress every 1000 iterations
        if t % 1000 == 0 and t != 0:
            elapsed_time = (time.time() - start_time) / 1000.0
            print(f"[Step {t}/{time_steps}] {elapsed_time:.2f} s per step")
            start_time = time.time()

    return wavefields if save_wavefields else None, gather if gather_output else None
