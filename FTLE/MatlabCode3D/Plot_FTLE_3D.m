function Plot_FTLE_3D(coords, ftle, isotropy, back_ftle, back_isotropy, ...
                      initial_time, final_time, grid_resolution, save_plot_path)
%PLOT_FTLE_3D  Interpolate and visualize forward/backward FTLE and isotropy in 3D.
%
% coords         : [N x 3] particle coordinates
% ftle, isotropy : forward values (N x 1)
% back_ftle, back_isotropy : backward values (N x 1)
% initial_time, final_time : scalars for labeling
% grid_resolution : scalar (default 50)
% save_plot_path  : optional string to save figure

    if nargin < 9, save_plot_path = []; end
    if nargin < 8 || isempty(grid_resolution), grid_resolution = 50; end

    x = coords(:,1);
    y = coords(:,2);
    z = coords(:,3);

    % Build regular grid
    xi = linspace(min(x), max(x), grid_resolution);
    yi = linspace(min(y), max(y), grid_resolution);
    zi = linspace(min(z), max(z), grid_resolution);
    [Xg, Yg, Zg] = meshgrid(xi, yi, zi);

    % Interpolate each field (linear)
    ftle_grid  = griddata(x, y, z, ftle,        Xg, Yg, Zg, 'linear');
    iso_grid   = griddata(x, y, z, isotropy,    Xg, Yg, Zg, 'linear');
    bftle_grid = griddata(x, y, z, back_ftle,   Xg, Yg, Zg, 'linear');
    biso_grid  = griddata(x, y, z, back_isotropy,Xg, Yg, Zg, 'linear');

    % Fill NaNs with nearest
    ftle_grid  = fillnans3d(ftle_grid,  x, y, z, ftle,        Xg, Yg, Zg);
    iso_grid   = fillnans3d(iso_grid,   x, y, z, isotropy,    Xg, Yg, Zg);
    bftle_grid = fillnans3d(bftle_grid, x, y, z, back_ftle,   Xg, Yg, Zg);
    biso_grid  = fillnans3d(biso_grid,  x, y, z, back_isotropy,Xg, Yg, Zg);

    % Flatten for scatter plot
    Xf = Xg(:); Yf = Yg(:); Zf = Zg(:);
    vals = {ftle_grid(:), iso_grid(:), bftle_grid(:), biso_grid(:)};
    titles = { ...
        sprintf('Forward FTLE, Time: %g to %g', initial_time, final_time), ...
        sprintf('Forward Isotropy, Time: %g to %g', initial_time, final_time), ...
        sprintf('Backward FTLE, Time: %g to %g', final_time, initial_time), ...
        sprintf('Backward Isotropy, Time: %g to %g', final_time, initial_time) ...
    };

    % Create figure
    figure('Position',[100 100 1000 900]);
    for k = 1:4
        subplot(2,2,k);
        scatter3(Xf, Yf, Zf, 5, vals{k}, 'filled');
        axis equal off
        title(titles{k});
        colormap(gca, 'parula');
        colorbar;
    end
    sgtitle('FTLE 3D Results');

    if ~isempty(save_plot_path)
        try
            exportgraphics(gcf, save_plot_path, 'Resolution', 300);
        catch
            saveas(gcf, save_plot_path);
        end
    end
end

function grid_out = fillnans3d(grid_in, x, y, z, vals, Xg, Yg, Zg)
    if any(isnan(grid_in(:)))
        nearest_grid = griddata(x, y, z, vals, Xg, Yg, Zg, 'nearest');
        nan_mask = isnan(grid_in);
        grid_out = grid_in;
        grid_out(nan_mask) = nearest_grid(nan_mask);
    else
        grid_out = grid_in;
    end
end
