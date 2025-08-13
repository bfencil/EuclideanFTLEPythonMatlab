function plot_FTLE_2d(x_grid_parts, y_grid_parts, ...
    ftle, isotropy, back_ftle, back_isotropy, ...
    initial_time, final_time, resolution, method, save_plot_path)
%PLOT_FTLE_2D  Interpolate & plot forward/backward FTLE & isotropy (2x2).
%
% Inputs mirror the Python version. Optional:
%   resolution (default 200)
%   method     (default 'linear')  % 'linear'|'cubic'|'nearest'
%   save_plot_path (default [])
%
% Notes:
% - Uses griddata(x,y,v,X,Y,method) for interpolation
% - Uses tiledlayout for clean 2x2 plotting

    if nargin < 9 || isempty(resolution), resolution = 200; end
    if nargin < 10 || isempty(method),     method     = 'linear'; end
    if nargin < 11,                        save_plot_path = []; end

    % Flatten original grid positions to match field vectors
    x = x_grid_parts(:);
    y = y_grid_parts(:);

    % Build interpolation grid
    xi = linspace(min(x), max(x), resolution);
    yi = linspace(min(y), max(y), resolution);
    [X, Y] = meshgrid(xi, yi);

    fields = { ...
        sprintf('Forward FTLE, Time: %g to %g',  initial_time, final_time),   ftle(:); ...
        sprintf('Forward Isotropy, Time: %g to %g', initial_time, final_time), isotropy(:); ...
        sprintf('Backward FTLE, Time: %g to %g',  final_time, initial_time),   back_ftle(:); ...
        sprintf('Backward Isotropy, Time: %g to %g', final_time, initial_time), back_isotropy(:) ...
    };

    tl = tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    for k = 1:4
        nexttile;
        titleStr = fields{k,1};
        v        = fields{k,2};

        % Interpolate to dense grid
        Z = griddata(x, y, v, X, Y, method);

        % Plot
        p = pcolor(X, Y, Z);
        set(p,'EdgeColor','none');  % smooth shading
        axis equal tight
        xlabel('X Position'); ylabel('Y Position');
        title(titleStr);
        cb = colorbar;
        cb.Label.String = titleStr;  % mimic Python labeling
        colormap(parula);            % or 'turbo' (MATLAB), swap if you prefer
    end

    % Save if requested
    if ~isempty(save_plot_path)
        try
            exportgraphics(gcf, save_plot_path, 'Resolution', 300);
        catch
            % fallback for older MATLAB
            saveas(gcf, save_plot_path);
        end
        fprintf('Plot saved to %s\n', save_plot_path);
    end
end
