function [ftle, trajectories, isotropy, back_ftle, back_trajectories, back_isotropy] = ...
    run_FTLE_2d(velocity_points, velocity_vectors, x_grid_parts, y_grid_parts, ...
                dt, initial_time, final_time, time_steps, plot_ftle, save_plot_path)
%RUN_FTLE_2d  Top-level wrapper (MATLAB version of FlatFTLEMain2d.run_FTLE_2d)
%
% Inputs
%   velocity_points  : [M x 2]
%   velocity_vectors : [M x 2 x T] (or [M x 2] for time-independent; treat T=1)
%   x_grid_parts     : [Nx x Ny] meshgrid X
%   y_grid_parts     : [Nx x Ny] meshgrid Y
%   dt               : scalar, 0 < dt <= 1
%   initial_time     : value from time_steps
%   final_time       : value from time_steps
%   time_steps       : [1 x T] or [T x 1] vector of integer steps
%   plot_ftle        : logical (optional, default false)
%   save_plot_path   : string/char (optional)
%
% Outputs mirror the Python version.

    if nargin < 9 || isempty(plot_ftle), plot_ftle = false; end
    if nargin < 10, save_plot_path = []; end

    % ensure row vector for comparisons/indexing
    time_steps = time_steps(:).';
    if ~ismember(initial_time, time_steps) || ~ismember(final_time, time_steps)
        error('Initial and final times must be in time_steps.');
    end
    if ~(dt > 0 && dt <= 1)
        error('dt must be in (0,1].');
    end
    if initial_time >= final_time
        error('initial_time must be less than final_time.');
    end

    initial_time_index = find(time_steps == initial_time, 1, 'first');
    final_time_index   = find(time_steps == final_time,   1, 'first');

    % ---- Forward ----
    [ftle, trajectories, isotropy] = FTLE_2d( ...
        velocity_points, velocity_vectors, x_grid_parts, y_grid_parts, ...
        dt, initial_time_index, final_time_index, time_steps, 'Forward');

    % ---- Backward ----
    [back_ftle, back_trajectories, back_isotropy] = FTLE_2d( ...
        velocity_points, velocity_vectors, x_grid_parts, y_grid_parts, ...
        dt, final_time_index, initial_time_index, time_steps, 'Backward');

    if plot_ftle
        plot_FTLE_2d(x_grid_parts, y_grid_parts, ...
            ftle, isotropy, back_ftle, back_isotropy, ...
            initial_time, final_time, save_plot_path);
    end
end
