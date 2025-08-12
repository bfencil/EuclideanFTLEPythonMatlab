import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))  # Adjust for relative import

from FTLE.PythonCode.FlatSurfaceMain import run_FTLE_2d # Primary FTLE computation
import h5py
import numpy as np

# --- Load Bickley flow data ---
file_path = os.path.join(os.path.dirname(__file__), rf'Data/bickley_flow_data.h5')
with h5py.File(file_path, 'r') as f:
    velocity_points = f['points'][:]                 # shape (M, 2)
    velocity_vectors = f['vectors'][:]               # shape (M, 2, T)
    time_steps = f['time_steps'][:]                  # shape (T,)

# --- Define 2D grid for particle seeding (must be inside domain) ---
x = np.linspace(0, 10, 120)
y = np.linspace(-3, 3, 120)
X, Y = np.meshgrid(x, y)

# Swap the rows and columns
X = X.T
Y = Y.T

# --- FTLE parameters ---
initial_time = time_steps[0]
final_time = 3
dt = 0.2

# --- Run FTLE computation ---
ftle, traj, iso, bftle, btraj, biso = run_FTLE_2d(
    velocity_points,
    velocity_vectors,
    X,
    Y,
    dt,
    initial_time,
    final_time,
    time_steps,
    plot_ftle=True,
    save_plot_path=None  # or provide a path like 'bickley_ftle_output.png'
)




