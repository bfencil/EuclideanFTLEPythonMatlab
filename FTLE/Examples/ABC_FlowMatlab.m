% ABC_FlowMatlab.m
% Example: run the 3D FTLE pipeline on the ABC flow data.

%% Make sure the 3D code is on the path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'MatlabCode3D'));

%% Load data (prefer .mat, fallback to .h5)
dataDir  = fullfile(fileparts(mfilename('fullpath')), 'Data');
matFile  = fullfile(dataDir, 'abc_flow_data.mat');
h5File   = fullfile(dataDir, 'abc_flow_data.h5');

if exist(matFile, 'file')
    S = load(matFile);                      % expects: points, vectors, time_steps
    velocity_points  = S.points;            % [M x 3]
    velocity_vectors = S.vectors;           % [M x 3 x T]
    time_steps       = S.time_steps(:).';   % row vector
elseif exist(h5File, 'file')
    velocity_points  = h5read(h5File, '/points').';    % h5read -> columns; transpose to [M x 3]
    velocity_vectors = h5read(h5File, '/vectors');     % expect [3 x M x T] or [M x 3 x T] depending on write
    % If vectors came in as [3 x M x T], permute:
    if size(velocity_vectors,1) == 3 && size(velocity_vectors,2) ~= 3
        velocity_vectors = permute(velocity_vectors, [2 1 3]);  % -> [M x 3 x T]
    end
    time_steps       = h5read(h5File, '/time_steps').';
else
    error('Could not find abc_flow_data.(mat|h5) in %s', dataDir);
end

%% Define the 3D seeding grid (inside domain)
x = linspace(0, 2*pi, 30);
y = linspace(0, 2*pi, 30);
z = linspace(0, 2*pi, 30);

% Use ndgrid to match Python’s indexing='ij' without transposes
[X, Y, Z] = ndgrid(x, y, z);

%% FTLE parameters
initial_time = time_steps(1);
final_time   = 4;
dt           = 1;       % allowed since dt ∈ (0, 1]

%% Run FTLE (plots enabled)
[ftle, traj, iso, bftle, btraj, biso] = Run_FTLE_3D( ...
    velocity_points, ...
    velocity_vectors, ...
    X, Y, Z, ...
    dt, ...
    initial_time, ...
    final_time, ...
    time_steps, ...
    true, ...   % plot_ftle
    [] ...      % save_plot_path ('' or [] to skip saving)
);

% Optionally save results
% save(fullfile(dataDir, 'abc_ftle_results.mat'), 'ftle','traj','iso','bftle','btraj','biso');
