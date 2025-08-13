% BickleyFlow2D_Example.m
% Example usage of the 2D FTLE pipeline on Bickley flow data.

% --- Add MatlabCode2D folder to path ---
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'MatlabCode2D'));

% --- Load Bickley flow data (.mat) ---
dataFile = fullfile(fileparts(mfilename('fullpath')), 'Data', 'bickley_flow_data.mat');
S = load(dataFile);  % expects variables: points, vectors, time_steps
velocity_points  = S.points;    % [M x 2]
velocity_vectors = S.vectors;   % [M x 2 x T]
time_steps       = S.time_steps; % [1 x T] or [T x 1]

% --- Define particle seeding grid (inside domain) ---
x = linspace(0, 10, 80);
y = linspace(-3, 3, 80);
[X, Y] = meshgrid(x, y);

% Swap rows/columns 
X = X';
Y = Y';

% --- FTLE parameters ---
initial_time = time_steps(1);
final_time   = 3;
dt           = 0.2;

% --- Run FTLE computation ---
[ftle, traj, iso, bftle, btraj, biso] = Run_FTLE_2d( ...
    velocity_points, ...
    velocity_vectors, ...
    X, ...
    Y, ...
    dt, ...
    initial_time, ...
    final_time, ...
    time_steps, ...
    true, ...     % plot_ftle
    [] ...        % save_plot_path (empty = no save)
);

% Optionally, save outputs
% save('bickley_ftle_results.mat', 'ftle', 'traj', 'iso', 'bftle', 'btraj', 'biso');
