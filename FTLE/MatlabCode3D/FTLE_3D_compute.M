function [FTLE_flat, isotropy_flat] = FTLE_3D_compute( ...
    x_initial, y_initial, z_initial, ...
    x_final,   y_final,   z_final, ...
    initial_time, final_time)
%FTLE_3D_COMPUTE  Compute 3D FTLE on a uniform grid using centered differences.
%
% Inputs:
%   x_initial,y_initial,z_initial : [Nx x Ny x Nz] initial grid coordinates
%   x_final,  y_final,  z_final   : [Nx x Ny x Nz] advected grid coordinates
%   initial_time, final_time      : scalars
%
% Outputs:
%   FTLE_flat     : [Nx*Ny*Nz x 1] vector (NaN at boundaries/invalid)
%   isotropy_flat : [Nx*Ny*Nz x 1] vector

    [nx, ny, nz] = size(x_initial);
    FTLE     = nan(nx, ny, nz);
    isotropy = nan(nx, ny, nz);

    T = abs(final_time - initial_time);
    if T <= 0
        error('initial_time and final_time must differ to compute FTLE.');
    end

    % reusable 3x3 deformation gradient
    F = zeros(3,3);

    % skip outermost ring (centered differences)
    for k = 2:(nz-1)        % z-index
        for i = 2:(nx-1)    % x-index
            for j = 2:(ny-1)% y-index

                if isnan(x_initial(i,j,k)) || isnan(y_initial(i,j,k))
                    continue;
                end

                % local spacings (centered)
                dx = x_initial(i+1,j,k) - x_initial(i-1,j,k);
                dy = y_initial(i,j+1,k) - y_initial(i,j-1,k);
                dz = z_initial(i,j,k+1) - z_initial(i,j,k-1);

                if dx == 0 || dy == 0 || dz == 0
                    continue;
                end

                % deformation gradient F = dX_final / dX_initial
                F(1,1) = (x_final(i+1,j,k) - x_final(i-1,j,k)) / (2*dx);
                F(1,2) = (x_final(i,j+1,k) - x_final(i,j-1,k)) / (2*dy);
                F(1,3) = (x_final(i,j,k+1) - x_final(i,j,k-1)) / (2*dz);

                F(2,1) = (y_final(i+1,j,k) - y_final(i-1,j,k)) / (2*dx);
                F(2,2) = (y_final(i,j+1,k) - y_final(i,j-1,k)) / (2*dy);
                F(2,3) = (y_final(i,j,k+1) - y_final(i,j,k-1)) / (2*dz);

                F(3,1) = (z_final(i+1,j,k) - z_final(i-1,j,k)) / (2*dx);
                F(3,2) = (z_final(i,j+1,k) - z_final(i,j-1,k)) / (2*dy);
                F(3,3) = (z_final(i,j,k+1) - z_final(i,j,k-1)) / (2*dz);

                % Cauchy–Green tensor
                C = F.' * F;

                if any(~isfinite(C), 'all')
                    continue;
                end

                % enforce symmetry numerically, then eigenvalues
                Csym = 0.5*(C + C.');
                ev = eig(Csym);
                max_ev = max(ev);

                if ~(isfinite(max_ev) && max_ev > 0)
                    continue;
                end

                FTLE(i,j,k) = (1/(2*T)) * log(sqrt(max_ev));

                detC = det(Csym);
                if isfinite(detC) && detC > 0
                    isotropy(i,j,k) = (1/(2*T)) * log(detC);
                end
            end
        end
    end

    % flatten like numpy .flatten()
    FTLE_flat     = FTLE(:);
    isotropy_flat = isotropy(:);
end
