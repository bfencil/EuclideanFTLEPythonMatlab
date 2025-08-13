from numba import njit
import numpy as np
import math

@njit
def FTLE_3d_compute(x_initial, y_initial, z_initial, x_final, y_final, z_final, initial_time, final_time):
    nx, ny, nz = x_initial.shape
    FTLE = np.full((nx, ny, nz), np.nan)
    isotropy = np.full((nx,ny,nz), np.nan)
    F_right = np.zeros((3, 3))

    for z_index in range(1, nz - 1):
        for x_index in range(1, nx - 1):
            for y_index in range(1, ny - 1):

                if math.isnan(x_initial[x_index, y_index, z_index]) or math.isnan(y_initial[x_index, y_index, z_index]):
                    continue

                dx = x_initial[x_index + 1, y_index, z_index] - x_initial[x_index - 1, y_index, z_index]
                dy = y_initial[x_index, y_index + 1, z_index] - y_initial[x_index, y_index - 1, z_index]
                dz = z_initial[x_index, y_index, z_index + 1] - z_initial[x_index, y_index, z_index - 1]

                if dx == 0 or dy == 0 or dz == 0:
                    continue

                # ∂Xf/∂X0 (deformation gradient matrix, F_right)
                F_right[0, 0] = (x_final[x_index + 1, y_index, z_index] - x_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[0, 1] = (x_final[x_index, y_index + 1, z_index] - x_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[0, 2] = (x_final[x_index, y_index, z_index + 1] - x_final[x_index, y_index, z_index - 1]) / (2 * dz)

                F_right[1, 0] = (y_final[x_index + 1, y_index, z_index] - y_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[1, 1] = (y_final[x_index, y_index + 1, z_index] - y_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[1, 2] = (y_final[x_index, y_index, z_index + 1] - y_final[x_index, y_index, z_index - 1]) / (2 * dz)

                F_right[2, 0] = (z_final[x_index + 1, y_index, z_index] - z_final[x_index - 1, y_index, z_index]) / (2 * dx)
                F_right[2, 1] = (z_final[x_index, y_index + 1, z_index] - z_final[x_index, y_index - 1, z_index]) / (2 * dy)
                F_right[2, 2] = (z_final[x_index, y_index, z_index + 1] - z_final[x_index, y_index, z_index - 1]) / (2 * dz)

                # Cauchy-Green strain tensor
                C = F_right.T @ F_right

                if np.isnan(C).any() or np.isinf(C).any():
                    continue

                eigenvalues = np.linalg.eigh(C)[0]
                max_eigen = np.max(eigenvalues)

                if max_eigen <= 0:
                    continue

                time = np.abs(initial_time - final_time)
                FTLE[x_index, y_index, z_index] = (1 / (2 * time)) * np.log(np.sqrt(max_eigen))
                isotropy[x_index, y_index, z_index] = (1 / (2 * time)) * np.log(np.linalg.det(C))

    return FTLE.flatten(), isotropy.flatten()
