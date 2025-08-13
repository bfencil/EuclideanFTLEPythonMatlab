function trajectories = RK4_advection_3d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent)

    interpolate = @(floor_data, ceiling_data, t_fraction) t_fraction .* ceiling_data + (1 - t_fraction) .* floor_data;

    if time_independent
        interp_u = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), velocity_vectors(:,1), 'linear', 'none');
        interp_v = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), velocity_vectors(:,2), 'linear', 'none');
        interp_w = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), velocity_vectors(:,3), 'linear', 'none');

        for t_index = 1:length(fine_time)
            x_curr = trajectories(:, 1, t_index);
            y_curr = trajectories(:, 2, t_index);
            z_curr = trajectories(:, 3, t_index);

            k1_x = interp_u(x_curr, y_curr, z_curr);
            k1_y = interp_v(x_curr, y_curr, z_curr);
            k1_z = interp_w(x_curr, y_curr, z_curr);

            k2_x = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);
            k2_y = interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);
            k2_z = interp_w(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);

            k3_x = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);
            k3_y = interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);
            k3_z = interp_w(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);

            k4_x = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);
            k4_y = interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);
            k4_z = interp_w(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x);
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y);
            z_next = z_curr + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z);

            trajectories(:, 1, t_index + 1) = x_next;
            trajectories(:, 2, t_index + 1) = y_next;
            trajectories(:, 3, t_index + 1) = z_next;
        end
    else
        for t_index = 1:length(fine_time)
            t = fine_time(t_index);
            t_floor = floor(t);
            t_ceil = ceil(t);
            t_frac = t - t_floor;

            u_interp = interpolate(velocity_vectors(:,1,t_floor+1), velocity_vectors(:,1,t_ceil+1), t_frac);
            v_interp = interpolate(velocity_vectors(:,2,t_floor+1), velocity_vectors(:,2,t_ceil+1), t_frac);
            w_interp = interpolate(velocity_vectors(:,3,t_floor+1), velocity_vectors(:,3,t_ceil+1), t_frac);

            interp_u = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), u_interp, 'linear', 'none');
            interp_v = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), v_interp, 'linear', 'none');
            interp_w = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), w_interp, 'linear', 'none');

            x_curr = trajectories(:, 1, t_index);
            y_curr = trajectories(:, 2, t_index);
            z_curr = trajectories(:, 3, t_index);

            k1_x = interp_u(x_curr, y_curr, z_curr);
            k1_y = interp_v(x_curr, y_curr, z_curr);
            k1_z = interp_w(x_curr, y_curr, z_curr);

            k2_x = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);
            k2_y = interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);
            k2_z = interp_w(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y, z_curr + 0.5 * dt * k1_z);

            k3_x = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);
            k3_y = interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);
            k3_z = interp_w(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y, z_curr + 0.5 * dt * k2_z);

            k4_x = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);
            k4_y = interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);
            k4_z = interp_w(x_curr + dt * k3_x, y_curr + dt * k3_y, z_curr + dt * k3_z);

            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x);
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y);
            z_next = z_curr + (dt / 6.0) * (k1_z + 2*k2_z + 2*k3_z + k4_z);

            trajectories(:, 1, t_index + 1) = x_next;
            trajectories(:, 2, t_index + 1) = y_next;
            trajectories(:, 3, t_index + 1) = z_next;
        end
    end
end
