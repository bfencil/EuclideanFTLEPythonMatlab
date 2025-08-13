function [ftle, trajectories, isotropy] = FTLE_2D( ...
    velocity_points, velocity_vectors, x_grid_parts, y_grid_parts, ...
    dt, initial_time_index, final_time_index, time_steps, direction)
%FTLE_2D  Driver that advects particles and computes FTLE (MATLAB port)
%
% Inputs follow the Python version. direction = 'forward' or 'backward'.
%
% Outputs
%   ftle          : [Nx*Ny x 1] (flattened) or however FTLE_2D_compute returns
%   trajectories  : [Nparticles x 2 x K] positions across fine-time
%   isotropy      : whatever FTLE_2D_compute returns for isotropy

    direction = lower(string(direction));

    % handle backward: reverse time axis and indices, flip dt
    if direction == "backward"
        T = numel(time_steps);
        % map indices as in the Python code
        tmp_initial = T - final_time_index;   % (len - final - 1) + 1 for MATLAB indexing -> T - final
        new_final   = T - initial_time_index; % same mapping
        initial_time_index = new_final;
        final_time_index   = tmp_initial;

        % reverse time axis of the data along 3rd dim if it exists
        if ndims(velocity_vectors) == 3
            velocity_vectors = velocity_vectors(:, :, end:-1:1);
        end
        dt = -dt;
    end

    % --- Grid / particles ---
    [nx, ny] = size(x_grid_parts);
    particles_positions = [x_grid_parts(:), y_grid_parts(:)];
    Np = size(particles_positions, 1);

    % build fine-time vector over the selected integer range
    sel = time_steps(initial_time_index:final_time_index);
    fine_time = subdivide_time_steps(sel, abs(dt));      % user-supplied helper
    K = numel(fine_time);

    % allocate trajectories and set initial
    trajectories = zeros(Np, 2, K, 'like', particles_positions);
    trajectories(:, :, 1) = particles_positions;

    % exclude final point for the RK4 stepping loop (like Python)
    if K > 1
        fine_time_eval = fine_time(1:end-1);
    else
        fine_time_eval = fine_time; % degenerate but safe
    end

    % advance
    trajectories = RK4_advection_2D(velocity_points, velocity_vectors, ...
                                    trajectories, dt, fine_time_eval);

    % reshape for FTLE compute at final time slice
    x_traj = reshape(trajectories(:, 1, :), nx, ny, K);
    y_traj = reshape(trajectories(:, 2, :), nx, ny, K);

    [ftle, isotropy] = FTLE_2D_compute( ...
        x_grid_parts, y_grid_parts, ...
        x_traj(:, :, end), y_traj(:, :, end), ...
        time_steps(initial_time_index), time_steps(final_time_index));
end
