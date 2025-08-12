function fine_time = subdivide_time_steps(time_steps, dt)
    % Subdivide time intervals based on dt
    fine_time = [];
    for i = 1:length(time_steps) - 1
        start_time = time_steps(i);
        end_time = time_steps(i + 1);
        num_subdivisions = round(1 / dt);
        step = (end_time - start_time) * dt;
        for j = 0:(num_subdivisions - 1)
            fine_time(end + 1) = start_time + j * step; %#ok<AGROW>
        end
    end
    fine_time(end + 1) = time_steps(end);
end

function plot_FTLE_2d(x_grid_parts, y_grid_parts, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, resolution, method, save_path)
    if nargin < 11
        save_path = [];
    end
    if nargin < 10
        method = 'linear';
    end
    if nargin < 9
        resolution = 200;
    end

    particles = [x_grid_parts(:), y_grid_parts(:)];
    x = particles(:,1);
    y = particles(:,2);

    xi = linspace(min(x), max(x), resolution);
    yi = linspace(min(y), max(y), resolution);
    [X, Y] = meshgrid(xi, yi);

    fields = {
        sprintf("Forward FTLE, Time: %g to %g", initial_time, final_time), ftle;
        sprintf("Forward Isotropy, Time: %g to %g", initial_time, final_time), isotropy;
        sprintf("Backward FTLE, Time: %g to %g", final_time, initial_time), back_ftle;
        sprintf("Backward Isotropy, Time: %g to %g", final_time, initial_time), back_isotropy
    };

    figure;
    for i = 1:4
        subplot(2,2,i);
        V = griddata(x, y, fields{i,2}, X, Y, method);
        imagesc(xi, yi, V);
        axis equal tight;
        set(gca,'YDir','normal');
        colormap('parula');
        colorbar;
        title(fields{i,1});
        xlabel('X Position'); ylabel('Y Position');
    end

    if ~isempty(save_path)
        saveas(gcf, save_path);
    end
end

function plot_FTLE_3d(coords, ftle, isotropy, back_ftle, back_isotropy, initial_time, final_time, grid_resolution, save_path)
    if nargin < 9
        save_path = [];
    end
    if nargin < 8
        grid_resolution = 50;
    end

    x = coords(:,1);
    y = coords(:,2);
    z = coords(:,3);

    xlin = linspace(min(x), max(x), grid_resolution);
    ylin = linspace(min(y), max(y), grid_resolution);
    zlin = linspace(min(z), max(z), grid_resolution);
    [Xg, Yg, Zg] = meshgrid(xlin, ylin, zlin);

    F = {@(v) griddata(x, y, z, v, Xg, Yg, Zg, 'linear')};
    datasets = {
        ftle, isotropy, back_ftle, back_isotropy
    };
    titles = {
        sprintf("Forward FTLE, Time: %g to %g", initial_time, final_time),
        sprintf("Forward Isotropy, Time: %g to %g", initial_time, final_time),
        sprintf("Backward FTLE, Time: %g to %g", final_time, initial_time),
        sprintf("Backward Isotropy, Time: %g to %g", final_time, initial_time)
    };

    figure;
    for i = 1:4
        subplot(2,2,i);
        V = F{1}(datasets{i});
        V(isnan(V)) = 0;
        scatter3(Xg(:), Yg(:), Zg(:), 15, V(:), 'filled');
        axis equal tight;
        view(3);
        colormap('parula');
        colorbar;
        title(titles{i});
    end

    if ~isempty(save_path)
        saveas(gcf, save_path);
    end
end
