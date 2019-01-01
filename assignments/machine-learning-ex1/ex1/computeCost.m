function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%
%   INPUTS:
%   --> X is a (m x n+1) matrix of input feature samples
%   --> y is a (m x 1) column vector of output samples
%   --> theta is a (n+1 x 1) row vector of coefficients
%
%   OUTPUTS:
%   --> J is scalar cost function value
% -----------------------------------------------------------------------------

% Initialize some useful values
m = length(y); % number of training examples

% Transpose X to (n+1 x m)
X = X';

% Compute the cost of a particular choice of theta
J = 0;

% Iterate over every training sample
for i = [1:m]
  J = J + (theta' * X(:, i) - y(i))^2;
end

% Normalize by 1/2M
J = J * 1 / (2 * m);

end
