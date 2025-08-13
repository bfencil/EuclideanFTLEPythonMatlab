function [FTLE_flat, isotropy_flat] = FTLE_2D_compute( ...
    x_initial, y_initial, x_final, y_final, initial_time, final_time)
%FTLE_2D_COMPUTE  Compute FTLE on a uniform 2D grid using centered differences.
%
% Inputs:
%   x_initial, y_initial : [Nx x Ny] initial grid coordinates
%   x_final,   y_final   : [Nx x Ny] advected grid coordinates (same layout)
%   initial_time, final_time : scalars defining total advection time
%
% Outputs:
%   FTLE_flat     : [Nx*Ny x 1] vector of FTLE values (NaN at boundaries/invalid)
%   isotropy_flat : [Nx*Ny x 1] vector of isotropy measure (same masking)
%
% Mirrors the Python logic (including skipping borders and NaN handling).

    [nx, ny] = size(x_initial);
    FTLE     = nan(nx, ny);
    isotropy = nan(nx, ny);

    time = abs(final_time - initial_time);
    if time <= 0
        error('initial_time and final_time must differ to compute FTLE.');
    end

    % Reuse a small 2x2 for the deformation gradient
    F = zeros(2,2);

    % Skip the outermost ring (need centered differences)
    for i = 2:(nx-1)
        for j = 2:(ny-1)

            if isnan(x_initial(i,j)) || isnan(y_initial(i,j))
                continue;
            end

            % Local grid spacing (centered)
            dx = x_initial(i+1,j) - x_initial(i-1,j);
            dy = y_initial(i,j+1) - y_initial(i,j-1);

            if dx == 0 || dy == 0
                % fprintf('Zero differential at (%d,%d)\n', i, j);
                continue;
            end

            % Deformation gradient F = dX_final / dX_initial (centered)
            F(1,1) = (x_final(i+1,j) - x_final(i-1,j)) / (2*dx);
            F(1,2) = (x_final(i,j+1) - x_final(i,j-1)) / (2*dy);
            F(2,1) = (y_final(i+1,j) - y_final(i-1,j)) / (2*dx);
            F(2,2) = (y_final(i,j+1) - y_final(i,j-1)) / (2*dy);

            % Cauchy–Green tensor
            C = F.' * F;

            if any(~isfinite(C), 'all')
                % fprintf('Non-finite C at (%d,%d)\n', i, j);
                continue;
            end

            % Symmetric -> eigenvalues are real and nonnegative (ideally)
            ev = eig((C + C.')/2);           % enforce symmetry numerically
            max_ev = max(ev);

            if ~(isfinite(max_ev) && max_ev > 0)
                continue;
            end

            FTLE(i,j) = (1/(2*time)) * log(sqrt(max_ev));

            detC = det(C);
            if isfinite(detC) && detC > 0
                isotropy(i,j) = (1/(2*time)) * log(detC);
            else
                % leave NaN if determinant nonpositive/nonfinite
            end
        end
    end

    % Flatten like Python .flatten()
    FTLE_flat     = FTLE(:);
    isotropy_flat = isotropy(:);
end
