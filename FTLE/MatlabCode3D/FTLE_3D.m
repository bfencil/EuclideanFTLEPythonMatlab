function [ftle, trajectories, isotropy] = FTLE_3D( ...
    velocity_points, velocity_vectors, ...
    x_grid_parts, y_grid_parts, z_grid_parts, ...
    dt, initial_time_index, final_time_index, time_steps, direction)
%FTLE_3D  Advect a 3-D grid with RK4 and compute FTLE (MATLAB port)

    direction = lower(string(direction));

    if direction == "backward"
        T = numel(time_steps);
        tmp_initial        = T - final_time_index;
        new_final          = T - initial_time_index;
        initial_time_index = new_final;
        final_time_index   = tmp_initial;

        if ndims(velocity_vectors) == 3
            velocity_vectors = velocity_vectors(:, :, end:-1:1);
        end
        dt = -dt;
    end

    % --- Grid setup ---
    [nx, ny, nz] = size(x_grid_parts);

    particle_positions = [x_grid_parts(:), y_grid_parts(:), z_grid_parts(:)];
    Np = size(particle_positions, 1);

    % Fine time sequence
    sel = time_steps(initial_time_index:final_time_index);
    fine_time = subdivide_time_steps(sel, abs(dt));
    K = numel(fine_time);

    % trajectories: Np x 3 x K
    trajectories = zeros(Np, 3, K, 'like', particle_positions);
    trajectories(:, :, 1) = particle_positions;

    % exclude last point when stepping (mirror Python)
    if K > 1
        fine_eval = fine_time(1:end-1);
    else
        fine_eval = fine_time;
    end

    % Advection
    trajectories = RK4_advection_3D(velocity_points, velocity_vectors, ...
                                    trajectories, dt, fine_eval);

    % Reshape & compute FTLE at final slice
    x_traj = reshape(trajectories(:,1,:), nx, ny, nz, K);
    y_traj = reshape(trajectories(:,2,:), nx, ny, nz, K);
    z_traj = reshape(trajectories(:,3,:), nx, ny, nz, K);

    [ftle, isotropy] = FTLE_3D_compute( ...
        x_grid_parts, y_grid_parts, z_grid_parts, ...
        x_traj(:,:,:,end), y_traj(:,:,:,end), z_traj(:,:,:,end), ...
        time_steps(initial_time_index), time_steps(final_time_index));
end
