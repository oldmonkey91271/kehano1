function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

%   Inputs: X - [m x n+1] matrix of input training samples / features
%           y - [m x 1] column vector of training output samples
%           theta - [n+1 x 1] column vector of optimization weights
%           alpha - the scalar learning rate of the optimization
%           num_iters - the number of iterations to perform the optimization
%
%   Outputs: theta-
%            J_history - [num_iter x 1] Cost function J(theta) values
%
%   Internal: m - the number of training samples
%             n+1 - the number of optimization weights (theta_0..theta_n)
% -----------------------------------------------------------------------------

% Initialize some useful values
m = length(y);        % number of input training samples
n1 = length(theta);   % number of optimization weights

% Initialize the cost function history
J_history = zeros(num_iters, 1);

% Transpose X to (n+1 x m)
X = X';

% Iterate the batch Gradient Descent linear regression optimization
for iter = 1:num_iters

    % Initialize an internal variable for the optimization
    z = zeros(n1, 1);
    
    % Iterate over each optimization weight theta_j (n+1 weights)
    for j = [1:n1]

      % Iterate over every training sample
      for i = [1:m]
        z(j) = z (j) + ((theta' * X(:, i) - y(i)) * X(j, i));
      end
      % Normalize by the learning rate over the number of training
      z(j) = z(j) * (alpha / m);
      
    end

    % Update simultaneously our optimization weights theta_j
    theta = theta - z;  

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X', y, theta);

end

end
