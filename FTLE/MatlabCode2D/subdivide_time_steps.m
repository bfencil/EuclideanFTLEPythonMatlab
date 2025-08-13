function fine_time = subdivide_time_steps(time_steps, dt)
%SUBDIVIDE_TIME_STEPS  Mirror of the Python helper.
%   fine_time = subdivide_time_steps(time_steps, dt)
%   time_steps : vector of integer-like times (e.g. 0,1,2,...)
%   dt         : step fraction in (0,1]; Python used int(1/dt)

    if ~(dt > 0 && dt <= 1)
        error('dt must be in (0,1].');
    end

    time_steps = time_steps(:).';           % row
    fine_time  = [];

    % match Python's int(1/dt): truncate toward zero
    num_subdivisions = floor(1/dt);

    for i = 1:(numel(time_steps)-1)
        start_i = time_steps(i);
        end_i   = time_steps(i+1);
        step    = (end_i - start_i) * dt;

        % include intermediate points
        for j = 0:(num_subdivisions-1)
            fine_time(end+1) = start_i + j*step; %#ok<AGROW>
        end
    end

    % add the last original value
    fine_time(end+1) = time_steps(end);
end
