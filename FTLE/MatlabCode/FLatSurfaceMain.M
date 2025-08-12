% FlatSurfaceMain.m
% MATLAB version of the Flat FTLE computation interface.

function [ftle, trajectories, isotropy, back_ftle, back_trajectories, back_isotropy] = run_FTLE_2d(velocity_points, velocity_vectors, x_grid, y_grid, dt, initial_time, final_time, time_steps, time_independent, plot_ftle, save_plot_path)

    if nargin < 10, time_independent = false; end
    if nargin < 11, plot_ftle = false; end
    if nargin < 12, save_plot_path = ''; end

    if ~ismember(initial_time, time_steps) || ~ismember(final_time, time_steps)
        error('Initial and final times must be in `time_steps`.');
    end

    if initial_time >= final_time
        error('initial_time must be less than final_time');
    end

    if dt > 1 || dt <= 0
        error('dt must be in (0,1]');
    end

    initial_index = find(time_steps == initial_time);
    final_index = find(time_steps == final_time);

    [ftle, trajectories, isotropy] = FTLE_2d_compute_interface(velocity_points, velocity_vectors, x_grid, y_grid, dt, initial_index, final_index, time_steps, 'forward', time_independent);

    [back_ftle, back_trajectories, back_isotropy] = FTLE_2d_compute_interface(velocity_points, velocity_vectors, x_grid, y_grid, dt, final_index, initial_index, time_steps, 'backward', time_independent);

    if plot_ftle
        plot_FTLE_2d(x_grid, y_grid, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, save_plot_path);
    end
end

function [ftle, trajectories, isotropy] = run_FTLE_3d(velocity_points, velocity_vectors, x_grid, y_grid, z_grid, dt, initial_time, final_time, time_steps, time_independent, plot_ftle, save_plot_path)

    if nargin < 10, time_independent = false; end
    if nargin < 11, plot_ftle = false; end
    if nargin < 12, save_plot_path = ''; end

    if ~ismember(initial_time, time_steps) || ~ismember(final_time, time_steps)
        error('Initial and final times must be in `time_steps`.');
    end

    if initial_time >= final_time
        error('initial_time must be less than final_time');
    end

    if dt > 1 || dt <= 0
        error('dt must be in (0,1]');
    end

    initial_index = find(time_steps == initial_time);
    final_index = find(time_steps == final_time);

    [ftle, trajectories, isotropy] = FTLE_3d_compute_interface(velocity_points, velocity_vectors, x_grid, y_grid, z_grid, dt, initial_index, final_index, time_steps, 'forward', time_independent);

    [back_ftle, back_trajectories, back_isotropy] = FTLE_3d_compute_interface(velocity_points, velocity_vectors, x_grid, y_grid, z_grid, dt, final_index, initial_index, time_steps, 'backward', time_independent);

    if plot_ftle
        plot_FTLE_3d(x_grid, y_grid, z_grid, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, save_plot_path);
    end
end












function [ftle, trajectories, isotropy] = FTLE_2d_compute_interface(
    velocity_points, velocity_vectors,
    x_grid_parts, y_grid_parts,
    dt, initial_time_index, final_time_index,
    time_steps, direction, time_independent)

    % Handle time reversal if needed
    if strcmpi(direction, 'backward')
        temp_initial = length(time_steps) - final_time_index;
        final_time_index = length(time_steps) - initial_time_index;
        initial_time_index = temp_initial;

        if ~time_independent
            velocity_vectors = velocity_vectors(:, :, end:-1:1);
        end
        dt = -dt;
    end

    % Set up grid and particles
    [x_dim1, x_dim2] = size(x_grid_parts);
    particles_positions = [x_grid_parts(:), y_grid_parts(:)];
    num_particles = size(particles_positions, 1);

    fine_time = subdivide_time_steps(time_steps(initial_time_index:final_time_index), abs(dt));
    fine_time_length = length(fine_time);

    trajectories = zeros(num_particles, 2, fine_time_length);
    trajectories(:, :, 1) = particles_positions;

    fine_time = fine_time(1:end-1);

    trajectories = RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent);

    % Compute FTLE
    x_traj = reshape(trajectories(:, 1, end), x_dim1, x_dim2);
    y_traj = reshape(trajectories(:, 2, end), x_dim1, x_dim2);

    [ftle, isotropy] = FTLE_2d_compute(x_grid_parts, y_grid_parts, x_traj, y_traj, time_steps(initial_time_index), time_steps(final_time_index));
end


function [ftle, trajectories, isotropy] = FTLE_3d_compute_interface(
    velocity_points, velocity_vectors,
    x_grid_parts, y_grid_parts, z_grid_parts,
    dt, initial_time_index, final_time_index,
    time_steps, direction, time_independent)

    % Handle time reversal if needed
    if strcmpi(direction, 'backward')
        temp_initial = length(time_steps) - final_time_index;
        final_time_index = length(time_steps) - initial_time_index;
        initial_time_index = temp_initial;

        if ~time_independent
            velocity_vectors = velocity_vectors(:, :, end:-1:1);
        end
        dt = -dt;
    end

    % Grid setup
    [x_dim1, x_dim2, x_dim3] = size(x_grid_parts);
    particle_positions = [x_grid_parts(:), y_grid_parts(:), z_grid_parts(:)];
    num_particles = size(particle_positions, 1);

    fine_time = subdivide_time_steps(time_steps(initial_time_index:final_time_index), abs(dt));
    fine_time_length = length(fine_time);

    trajectories = zeros(num_particles, 3, fine_time_length);
    trajectories(:, :, 1) = particle_positions;

    fine_time = fine_time(1:end-1);

    trajectories = RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent);

    % Reshape to grid
    x_traj = reshape(trajectories(:, 1, end), x_dim1, x_dim2, x_dim3);
    y_traj = reshape(trajectories(:, 2, end), y_dim1, y_dim2, y_dim3);
    z_traj = reshape(trajectories(:, 3, end), z_dim1, z_dim2, z_dim3);

    [ftle, isotropy] = FTLE_3d_compute(x_grid_parts, y_grid_parts, z_grid_parts, x_traj, y_traj, z_traj, time_steps(initial_time_index), time_steps(final_time_index));
end


