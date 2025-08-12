function [FTLE_flat, isotropy_flat] = FTLE_2d_compute(x_initial, y_initial, x_final, y_final, initial_time, final_time)
    [nx, ny] = size(x_initial);
    FTLE = nan(nx, ny);
    isotropy = nan(nx, ny);
    F = zeros(2, 2);

    for i = 2:nx-1
        for j = 2:ny-1

            if isnan(x_initial(i,j)) || isnan(y_initial(i,j))
                continue;
            end

            dx = x_initial(i+1,j) - x_initial(i-1,j);
            dy = y_initial(i,j+1) - y_initial(i,j-1);

            if dx == 0 || dy == 0
                continue;
            end

            F(1,1) = (x_final(i+1,j) - x_final(i-1,j)) / (2 * dx);
            F(1,2) = (x_final(i,j+1) - x_final(i,j-1)) / (2 * dy);
            F(2,1) = (y_final(i+1,j) - y_final(i-1,j)) / (2 * dx);
            F(2,2) = (y_final(i,j+1) - y_final(i,j-1)) / (2 * dy);

            C = F' * F;

            if any(isnan(C(:))) || any(isinf(C(:)))
                continue;
            end

            eigvals = eig(C);
            max_eigenvalue = max(eigvals);

            if max_eigenvalue <= 0
                continue;
            end

            time = abs(initial_time - final_time);
            FTLE(i,j) = (1 / (2 * time)) * log(sqrt(max_eigenvalue));
            isotropy(i,j) = (1 / (2 * time)) * log(det(C));
        end
    end

    FTLE_flat = FTLE(:);
    isotropy_flat = isotropy(:);
end


function [FTLE_flat, isotropy_flat] = FTLE_3d_compute(x_initial, y_initial, z_initial, x_final, y_final, z_final, initial_time, final_time)
    [nx, ny, nz] = size(x_initial);
    FTLE = nan(nx, ny, nz);
    isotropy = nan(nx, ny, nz);
    F_right = zeros(3, 3);

    for z_index = 2:nz-1
        for x_index = 2:nx-1
            for y_index = 2:ny-1

                if isnan(x_initial(x_index,y_index,z_index)) || isnan(y_initial(x_index,y_index,z_index))
                    continue;
                end

                dx = x_initial(x_index+1, y_index, z_index) - x_initial(x_index-1, y_index, z_index);
                dy = y_initial(x_index, y_index+1, z_index) - y_initial(x_index, y_index-1, z_index);
                dz = z_initial(x_index, y_index, z_index+1) - z_initial(x_index, y_index, z_index-1);

                if dx == 0 || dy == 0 || dz == 0
                    continue;
                end

                F_right(1,1) = (x_final(x_index+1, y_index, z_index) - x_final(x_index-1, y_index, z_index)) / (2 * dx);
                F_right(1,2) = (x_final(x_index, y_index+1, z_index) - x_final(x_index, y_index-1, z_index)) / (2 * dy);
                F_right(1,3) = (x_final(x_index, y_index, z_index+1) - x_final(x_index, y_index, z_index-1)) / (2 * dz);

                F_right(2,1) = (y_final(x_index+1, y_index, z_index) - y_final(x_index-1, y_index, z_index)) / (2 * dx);
                F_right(2,2) = (y_final(x_index, y_index+1, z_index) - y_final(x_index, y_index-1, z_index)) / (2 * dy);
                F_right(2,3) = (y_final(x_index, y_index, z_index+1) - y_final(x_index, y_index, z_index-1)) / (2 * dz);

                F_right(3,1) = (z_final(x_index+1, y_index, z_index) - z_final(x_index-1, y_index, z_index)) / (2 * dx);
                F_right(3,2) = (z_final(x_index, y_index+1, z_index) - z_final(x_index, y_index-1, z_index)) / (2 * dy);
                F_right(3,3) = (z_final(x_index, y_index, z_index+1) - z_final(x_index, y_index, z_index-1)) / (2 * dz);

                C = F_right' * F_right;

                if any(isnan(C(:))) || any(isinf(C(:)))
                    continue;
                end

                eigvals = eig(C);
                max_eigen = max(eigvals);

                if max_eigen <= 0
                    continue;
                end

                time = abs(initial_time - final_time);
                FTLE(x_index, y_index, z_index) = (1 / (2 * time)) * log(sqrt(max_eigen));
                isotropy(x_index, y_index, z_index) = (1 / (2 * time)) * log(det(C));
            end
        end
    end

    FTLE_flat = FTLE(:);
    isotropy_flat = isotropy(:);
end
