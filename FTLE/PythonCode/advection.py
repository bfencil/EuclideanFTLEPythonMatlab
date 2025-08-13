import numpy as np
from numba import njit

# ---------------------------
# Internal helpers (Numba OK)
# ---------------------------

@njit(cache=True)
def _knn_idw(point, points, k, eps=1e-12):
    """
    Brute-force kNN + inverse-distance weights.
    Returns:
      idx: (k,) int32 indices of neighbors
      w:   (k,) float64 normalized weights
    """
    m = points.shape[0]
    d2 = np.empty(m, dtype=np.float64)
    # squared distances
    for i in range(m):
        acc = 0.0
        for j in range(points.shape[1]):
            diff = points[i, j] - point[j]
            acc += diff * diff
        d2[i] = acc

    order = np.argsort(d2)
    idx = order[:k]

    w = np.empty(k, dtype=np.float64)
    wsum = 0.0
    for j in range(k):
        wj = 1.0 / (np.sqrt(d2[idx[j]]) + eps)
        w[j] = wj
        wsum += wj

    inv_wsum = 1.0 / (wsum + eps)
    for j in range(k):
        w[j] *= inv_wsum

    return idx, w


@njit(cache=True)
def _interp_vec_time_dependent(point, points, vec_floor, vec_ceil, frac, k):
    """
    Interpolate vector at 'point' using IDW over 'vec_floor' and 'vec_ceil'
    with linear-in-time blending by 'frac' in [0,1].
      points:   (M, dpos)
      vec_floor:(M, dvec)
      vec_ceil: (M, dvec)
    returns: (dvec,)
    """
    idx, w = _knn_idw(point, points, k)
    dvec = vec_floor.shape[1]
    out = np.zeros(dvec, dtype=np.float64)
    for j in range(k):
        ii = idx[j]
        for a in range(dvec):
            val = (1.0 - frac) * vec_floor[ii, a] + frac * vec_ceil[ii, a]
            out[a] += w[j] * val
    return out


@njit(cache=True)
def _rk4_step(pos, dt, points, vec_floor, vec_ceil, frac, k):
    """
    One RK4 step for a single particle (works for 2D or 3D).
      pos:      (d,) current position
      points:   (M, d) velocity sample locations
      vec_*:    (M, d) velocity samples at floor/ceil times
      frac:     float in [0,1]
    returns next position (d,)
    """
    k1 = _interp_vec_time_dependent(pos, points, vec_floor, vec_ceil, frac, k)
    k2 = _interp_vec_time_dependent(pos + 0.5 * dt * k1, points, vec_floor, vec_ceil, frac, k)
    k3 = _interp_vec_time_dependent(pos + 0.5 * dt * k2, points, vec_floor, vec_ceil, frac, k)
    k4 = _interp_vec_time_dependent(pos + dt * k3, points, vec_floor, vec_ceil, frac, k)
    return pos + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------
# Public APIs (drop-in replacements)
# ---------------------------------

@njit(cache=True)
def RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, k=4):
    """
    Time-dependent 2D advection using Numba + IDW interpolation.

    Parameters
    ----------
    velocity_points : (M, 2) float64
    velocity_vectors: (M, 2, T) float64
    trajectories    : (N, 2, S) float64, preallocated; [:,:,0] are initial positions
    dt              : float
    fine_time       : (S-1,) float64 array of fractional "times" (e.g., 0, 0.25, 0.5, ...)
    k               : int, number of neighbors for IDW

    Returns
    -------
    trajectories : (N, 2, S) float64 (filled in-place)
    """
    N = trajectories.shape[0]
    S = trajectories.shape[2]

    for t_index in range(S - 1):
        t = fine_time[t_index]
        t_floor = int(np.floor(t))
        t_ceil  = int(np.ceil(t))
        frac = t - t_floor

        vec_floor = velocity_vectors[:, :, t_floor]  # (M,2)
        vec_ceil  = velocity_vectors[:, :, t_ceil]   # (M,2)

        for i in range(N):
            pos = trajectories[i, :, t_index]
            nxt = _rk4_step(pos, dt, velocity_points, vec_floor, vec_ceil, frac, k)
            trajectories[i, 0, t_index + 1] = nxt[0]
            trajectories[i, 1, t_index + 1] = nxt[1]

    return trajectories


@njit(cache=True)
def RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, k=16):
    """
    Time-dependent 3D advection using Numba + IDW interpolation.

    Parameters
    ----------
    velocity_points : (M, 3) float64
    velocity_vectors: (M, 3, T) float64
    trajectories    : (N, 3, S) float64, preallocated; [:,:,0] are initial positions
    dt              : float
    fine_time       : (S-1,) float64
    k               : int, number of neighbors for IDW

    Returns
    -------
    trajectories : (N, 3, S) float64 (filled in-place)
    """
    N = trajectories.shape[0]
    S = trajectories.shape[2]

    for t_index in range(S - 1):
        t = fine_time[t_index]
        t_floor = int(np.floor(t))
        t_ceil  = int(np.ceil(t))
        frac = t - t_floor

        vec_floor = velocity_vectors[:, :, t_floor]  # (M,3)
        vec_ceil  = velocity_vectors[:, :, t_ceil]   # (M,3)

        for i in range(N):
            pos = trajectories[i, :, t_index]
            nxt = _rk4_step(pos, dt, velocity_points, vec_floor, vec_ceil, frac, k)
            trajectories[i, 0, t_index + 1] = nxt[0]
            trajectories[i, 1, t_index + 1] = nxt[1]
            trajectories[i, 2, t_index + 1] = nxt[2]

    return trajectories

