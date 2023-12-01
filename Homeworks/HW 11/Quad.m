% MATLAB Script for Adaptive Quadrature
clear;

% Interval and values of x
a = 0;
b = 10;
x_values = [2, 4, 6, 8, 10];




for x = x_values
    % Use quad for adaptive quadrature

    func = @(t) t.^(x-1) .* exp(-t);

    tic;
    [quad_result,quad_output] = quad(func, a, b);
    time_elapsed = toc;

    fprintf('For x = %d:\n', x);
    fprintf('Adaptive Quadrature Result: %f\n', quad_result);
    fprintf('Number of Function Evaluations: %d\n', quad_output);

end
