function [ftle, trajectories, isotropy, back_ftle, back_trajectories, back_isotropy] = ...
    Run_FTLE_3D(velocity_points, velocity_vectors, ...
                x_grid_parts, y_grid_parts, z_grid_parts, ...
                dt, initial_time, final_time, time_steps, ...
                plot_ftle, save_plot_path)
%RUN_FTLE_3D  MATLAB version of FlatFTLEMain3D.run_FTLE_3D

    if nargin < 11 || isempty(plot_ftle),   plot_ftle = false; end
    if nargin < 12,                         save_plot_path = []; end

    time_steps = time_steps(:).';
    if ~ismember(initial_time, time_steps) || ~ismember(final_time, time_steps)
        error('Initial and final times must be in time_steps.');
    end
    if ~(dt > 0 && dt <= 1)
        error('dt must be in (0,1].');
    end
    if initial_time >= final_time
        error('initial_time must be less than final_time.');
    end

    initial_time_index = find(time_steps == initial_time, 1, 'first');
    final_time_index   = find(time_steps == final_time,   1, 'first');

    % ---- Forward ----
    [ftle, trajectories, isotropy] = FTLE_3D( ...
        velocity_points, velocity_vectors, ...
        x_grid_parts, y_grid_parts, z_grid_parts, ...
        dt, initial_time_index, final_time_index, time_steps, 'Forward');

    % ---- Backward ----
    [back_ftle, back_trajectories, back_isotropy] = FTLE_3D( ...
        velocity_points, velocity_vectors, ...
        x_grid_parts, y_grid_parts, z_grid_parts, ...
        dt, final_time_index, initial_time_index, time_steps, 'Backward');

    if plot_ftle
        particles_positions = [x_grid_parts(:), y_grid_parts(:), z_grid_parts(:)];
        Plot_FTLE_3D(particles_positions, ftle, isotropy, ...
                     back_ftle, back_isotropy, initial_time, final_time, save_plot_path);
    end
end


