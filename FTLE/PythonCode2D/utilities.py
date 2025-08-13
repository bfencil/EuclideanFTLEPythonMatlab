import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata



def subdivide_time_steps(time_steps, dt):
    fine_time = []
    for i in range(len(time_steps) - 1):
        start = time_steps[i]
        end = time_steps[i + 1]
        num_subdivisions = int(1 / dt)
        step = (end - start) * dt
        # Include intermediate points
        for j in range(num_subdivisions):
            fine_time.append(start + j * step)
    fine_time.append(time_steps[-1])  # add the last original value
    return fine_time

    
def plot_FTLE_2d(
    x_grid_parts,
    y_grid_parts,
    ftle,
    isotropy,
    back_ftle,
    back_isotropy,
    initial_time, 
    final_time,
    resolution=200,
    method='linear',
    save_plot_path=None
):
    """
    Interpolates and plots 2D scalar fields (FTLE/isotropy, forward/backward) in 2x2 subplots.

    Parameters:
        x_grid_parts, y_grid_parts (ndarray): original 2D meshgrid arrays of particle positions.
        ftle, isotropy, back_ftle, back_isotropy (ndarray): flattened scalar field values.
        resolution (int): grid resolution for interpolation.
        method (str): interpolation method: 'linear', 'cubic', or 'nearest'.
        save_plot_path (str or None): if not None, path to save the plot as an image.
    """

    # Flatten the grid positions to match the field values
    particles = np.vstack([x_grid_parts.flatten(), y_grid_parts.flatten()]).T

    x, y = particles[:, 0], particles[:, 1]

    xi = np.linspace(x.min(), x.max(), int(resolution))
    yi = np.linspace(y.min(), y.max(), int(resolution))
    X, Y = np.meshgrid(xi, yi)

    fields = [
        (f"Forward FTLE, Time: {initial_time} to {final_time}", ftle),
        (f"Forward Isotropy, Time: {initial_time} to {final_time}", isotropy),
        (f"Backward FTLE, Time: {final_time} to {initial_time}", back_ftle),
        (f"Backward Isotropy, Time: {final_time} to {initial_time}", back_isotropy)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (title, field) in zip(axes.flat, fields):
        Z = griddata(particles, field, (X, Y), method=method)
        pcm = ax.pcolormesh(X, Y, Z, shading='auto', cmap='plasma')
        fig.colorbar(pcm, ax=ax, label=title)
        ax.set_title(title)
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_aspect('equal')

    plt.tight_layout()

    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300)
        print(f"Plot saved to {save_plot_path}")

    plt.show()
    return None



