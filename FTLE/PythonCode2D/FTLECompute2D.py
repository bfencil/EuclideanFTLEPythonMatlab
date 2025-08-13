from numba import njit
import numpy as np
import math

@njit
def FTLE_2d_compute(x_initial, y_initial, x_final, y_final, initial_time, final_time):
    """
    Compute FTLE field on a uniform 2D grid using finite differences.

    Parameters:
        x_initial, y_initial: 2D arrays of initial grid positions.
        x_final, y_final: 2D arrays of advected grid positions.
        time (float): Total advection time.

    Returns:
        FTLE (2D ndarray): Finite-time Lyapunov exponent values.
    """
    nx, ny = x_initial.shape
    FTLE = np.full((nx, ny), np.nan)
    isotropy = np.full((nx,ny),np.nan)
    F = np.zeros((2, 2))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):

            # Skip NaNs in initial positions
            if math.isnan(x_initial[i, j]) or math.isnan(y_initial[i, j]):
                continue

            # Local grid spacing
            dx = x_initial[i + 1, j] - x_initial[i - 1, j]
            dy = y_initial[i, j + 1] - y_initial[i, j - 1]


            if dx == 0 or dy == 0:
                continue

            # Compute finite difference deformation gradient ∂Xf/∂X0
            F[0, 0] = (x_final[i + 1, j] - x_final[i - 1, j]) / (2 * dx)
            F[0, 1] = (x_final[i, j + 1] - x_final[i, j - 1]) / (2 * dy)
            F[1, 0] = (y_final[i + 1, j] - y_final[i - 1, j]) / (2 * dx)
            F[1, 1] = (y_final[i, j + 1] - y_final[i, j - 1]) / (2 * dy)

            # Cauchy-Green strain tensor: C = Fᵀ F
            C = F.T @ F

            if np.isnan(C).any() or np.isinf(C).any():
                continue

            # Maximum eigenvalue of C
            eigenvalues = np.linalg.eigvalsh(C)
            max_eigenvalue = np.max(eigenvalues)

            if max_eigenvalue <= 0:
                continue
            time = np.abs(initial_time - final_time)
            FTLE[i, j] = (1 / (2 * time)) * np.log(np.sqrt(max_eigenvalue))
            isotropy[i,j] = (1 / (2 * time)) * np.log(np.linalg.det(C))

    return FTLE.flatten(), isotropy.flatten()

