function [J, grad] = costFunction(theta, X, y)
%   COSTFUNCTION Compute cost and gradient for logistic regression
%   [J, grad] = COSTFUNCTION(theta, X, y) computes the cost of using theta as
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
% 
%   INPUTS:
%     theta = [n+1 x 1] optimization coefficient vector
%     X     = [m x n+1] training samples
%     y     = [m x 1] output vector
%
%   OUTPUTS: 
%     J     = Scalar value of the cost function
%     grad  = [n+1 x 1] optimization gradient (change) vector
%
% -----------------------------------------------------------------------------

% Determine the size of the training
[m, n1] = size(X);

% Transpose X to (n+1 x m) (per Jason's class notes)
X = X';

% You need to return the following variables correctly 

% Initialize the cost function to zero
J = 0;
% Initialize the gradient vector to zeros
grad = zeros(size(theta));

% First lets compute the hypothesis (Sigmoid) function based on the training 
% and optimizaton coefficients
h_x = sigmoid(theta' * X);    % (1 x m) vector of hypothesis values

% Compute the cost function of logistic regression:
J = (-1 / m ) * ((log(h_x) * y) + ((log(1 .- h_x) * (1 .- y))));

% Compute the gradients
for j = 1:n1
  grad(j) = (1 / m) * sum((h_x' .- y) .* X(j,:)');
end

end
