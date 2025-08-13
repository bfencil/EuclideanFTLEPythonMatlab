function trajectories = RK4_advection_3D(velocity_points, velocity_vectors, trajectories, dt, fine_time)
%RK4_ADVECTION_3D  RK4 advection for scattered 3D velocity data (MATLAB port).
%
% Inputs:
%   velocity_points  : [M x 3] sample locations (fixed in time)
%   velocity_vectors : [M x 3 x T] velocity at those locations for integer time indices
%   trajectories     : [N x 3 x K] particle positions; trajectories(:, :, 1) already set
%   dt               : scalar timestep (can be negative for backward)
%   fine_time        : [1 x (K-1)] times within integer indices (e.g., 0,0.2,0.4,...)
%
% Notes:
% - Emulates Python's LinearNDInterpolator(..., fill_value=0) using
%   scatteredInterpolant(...,'linear','none') and replacing NaNs with 0.

    lerp = @(a,b,t) t.*b + (1-t).*a;  % linear time interpolation
    T = size(velocity_vectors, 3);
    Nsteps = numel(fine_time);

    for t_index = 1:Nsteps
        t = fine_time(t_index);

        % Python uses 0-based frame indices; MATLAB is 1-based
        tf = floor(t) + 1;   % floor frame
        tc = ceil(t)  + 1;   % ceil  frame
        tf = max(1, min(T, tf));
        tc = max(1, min(T, tc));

        if tc == tf
            tfrac = 0.0;
        else
            tfrac = (t - (tf - 1)) / ((tc - 1) - (tf - 1));  % exact fraction between frames
        end

        % Time-interpolated velocity samples at the M sites
        u = lerp(velocity_vectors(:,1,tf), velocity_vectors(:,1,tc), tfrac);
        v = lerp(velocity_vectors(:,2,tf), velocity_vectors(:,2,tc), tfrac);
        w = lerp(velocity_vectors(:,3,tf), velocity_vectors(:,3,tc), tfrac);
        u = u(:); v = v(:); w = w(:);

        % Spatial interpolants (linear), NaN outside hull (we'll zero-fill)
        Fu = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), u, 'linear','none');
        Fv = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), v, 'linear','none');
        Fw = scatteredInterpolant(velocity_points(:,1), velocity_points(:,2), velocity_points(:,3), w, 'linear','none');

        x = trajectories(:,1,t_index);
        y = trajectories(:,2,t_index);
        z = trajectories(:,3,t_index);

        % k1
        k1x = Fu(x,y,z);  k1y = Fv(x,y,z);  k1z = Fw(x,y,z);
        k1x(isnan(k1x))=0; k1y(isnan(k1y))=0; k1z(isnan(k1z))=0;

        % k2
        k2x = Fu(x + 0.5*dt.*k1x, y + 0.5*dt.*k1y, z + 0.5*dt.*k1z);
        k2y = Fv(x + 0.5*dt.*k1x, y + 0.5*dt.*k1y, z + 0.5*dt.*k1z);
        k2z = Fw(x + 0.5*dt.*k1x, y + 0.5*dt.*k1y, z + 0.5*dt.*k1z);
        k2x(isnan(k2x))=0; k2y(isnan(k2y))=0; k2z(isnan(k2z))=0;

        % k3
        k3x = Fu(x + 0.5*dt.*k2x, y + 0.5*dt.*k2y, z + 0.5*dt.*k2z);
        k3y = Fv(x + 0.5*dt.*k2x, y + 0.5*dt.*k2y, z + 0.5*dt.*k2z);
        k3z = Fw(x + 0.5*dt.*k2x, y + 0.5*dt.*k2y, z + 0.5*dt.*k2z);
        k3x(isnan(k3x))=0; k3y(isnan(k3y))=0; k3z(isnan(k3z))=0;

        % k4
        k4x = Fu(x + dt.*k3x, y + dt.*k3y, z + dt.*k3z);
        k4y = Fv(x + dt.*k3x, y + dt.*k3y, z + dt.*k3z);
        k4z = Fw(x + dt.*k3x, y + dt.*k3y, z + dt.*k3z);
        k4x(isnan(k4x))=0; k4y(isnan(k4y))=0; k4z(isnan(k4z))=0;

        % RK4 update
        x_next = x + (dt/6).*(k1x + 2*k2x + 2*k3x + k4x);
        y_next = y + (dt/6).*(k1y + 2*k2y + 2*k3y + k4y);
        z_next = z + (dt/6).*(k1z + 2*k2z + 2*k3z + k4z);

        trajectories(:,1,t_index+1) = x_next;
        trajectories(:,2,t_index+1) = y_next;
        trajectories(:,3,t_index+1) = z_next;
    end
end
