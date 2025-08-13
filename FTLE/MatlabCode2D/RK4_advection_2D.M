function trajectories = RK4_advection_2D(velocity_points, velocity_vectors, trajectories, dt, fine_time)
%RK4_ADVECTION_2D  MATLAB version of the Python RK4 advection (2D).
%
% Inputs:
%   velocity_points  : [M x 2] sample locations (fixed in time)
%   velocity_vectors : [M x 2 x T] velocity at those locations for integer time indices
%   trajectories     : [N x 2 x K] particle positions; trajectories(:, :, 1) already set
%   dt               : scalar time step (can be negative for backward)
%   fine_time        : [1 x (K-1)] or [K-1 x 1] vector of sub-times (e.g., 0, 0.1, 0.2, ...)
%
% Notes:
% - Matches Python’s use of LinearNDInterpolator with linear interpolation.
% - Outside the convex hull, Python used fill_value=0. Here we emulate that by
%   using 'linear','none' and replacing NaNs with 0.

    % linear interpolation in time (lerp)
    lerp = @(a, b, t) t .* b + (1 - t) .* a;

    % sizes
    T = size(velocity_vectors, 3);
    Nsteps = numel(fine_time);

    for t_index = 1:Nsteps
        t = fine_time(t_index);

        % Python uses 0-based integer times; MATLAB is 1-based
        t_floor   = floor(t) + 1;
        t_ceiling = ceil(t)  + 1;

        % clamp to valid range [1, T]
        t_floor   = max(1, min(T, t_floor));
        t_ceiling = max(1, min(T, t_ceiling));

        if t_ceiling == t_floor
            t_fraction = 0.0;
        else
            % exact fraction between the two integer indices
            % (mirrors Python’s t - floor(t))
            t_fraction = (t - (t_floor - 1)) / ((t_ceiling - 1) - (t_floor - 1));
        end

        % Time-interpolate the velocity field at the M sample sites
        u_interp = lerp(velocity_vectors(:, 1, t_floor),   velocity_vectors(:, 1, t_ceiling), t_fraction);
        v_interp = lerp(velocity_vectors(:, 2, t_floor),   velocity_vectors(:, 2, t_ceiling), t_fraction);
        u_interp = u_interp(:);
        v_interp = v_interp(:);

        % Spatial interpolants (linear). 'none' -> NaN outside hull.
        Fu = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), u_interp, 'linear', 'none');
        Fv = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), v_interp, 'linear', 'none');

        % Current particle positions
        x_curr = trajectories(:, 1, t_index);
        y_curr = trajectories(:, 2, t_index);

        % k1
        k1_x = Fu(x_curr, y_curr);
        k1_y = Fv(x_curr, y_curr);

        % emulate fill_value=0 outside hull
        k1_x(isnan(k1_x)) = 0;  k1_y(isnan(k1_y)) = 0;

        % k2
        k2_x = Fu(x_curr + 0.5*dt.*k1_x, y_curr + 0.5*dt.*k1_y);
        k2_y = Fv(x_curr + 0.5*dt.*k1_x, y_curr + 0.5*dt.*k1_y);
        k2_x(isnan(k2_x)) = 0;  k2_y(isnan(k2_y)) = 0;

        % k3
        k3_x = Fu(x_curr + 0.5*dt.*k2_x, y_curr + 0.5*dt.*k2_y);
        k3_y = Fv(x_curr + 0.5*dt.*k2_x, y_curr + 0.5*dt.*k2_y);
        k3_x(isnan(k3_x)) = 0;  k3_y(isnan(k3_y)) = 0;

        % k4
        k4_x = Fu(x_curr + dt.*k3_x, y_curr + dt.*k3_y);
        k4_y = Fv(x_curr + dt.*k3_x, y_curr + dt.*k3_y);
        k4_x(isnan(k4_x)) = 0;  k4_y(isnan(k4_y)) = 0;

        % RK4 update
        x_next = x_curr + (dt/6.0) .* (k1_x + 2*k2_x + 2*k3_x + k4_x);
        y_next = y_curr + (dt/6.0) .* (k1_y + 2*k2_y + 2*k3_y + k4_y);

        trajectories(:, 1, t_index + 1) = x_next;
        trajectories(:, 2, t_index + 1) = y_next;
    end
end
