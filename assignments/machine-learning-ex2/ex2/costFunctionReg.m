function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
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

% Initialize some useful values
%m = length(y); % number of training examples

% Determine the size of the training
[m, n1] = size(X);

% Transpose X to (n+1 x m)
X = X';

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% First lets compute the hypothesis (Sigmoid) function based on the training 
% and optimizaton coefficients
h_x = sigmoid(theta' * X);    % (1 x m) vector of hypothesis values

% Compute the cost function of logistic regression:
J = (-1 / m ) * ((log(h_x) * y) + ((log(1 .- h_x) * (1 .- y))));
% Add the regularization term to the cost
J = J + ((lambda / 2 / m) * sum(theta(2:end).^2));

% Compute the gradients (theta_0, theta_1..n)
grad(1) = (1 / m) * sum((h_x' .- y) .* X(1,:)');
% Compute the gradients w/regularization
for j = 2:n1
  grad(j) = (1 / m) * sum((h_x' .- y) .* X(j,:)');
  % Add the regularization term on the gradient
  grad(j) = grad(j) + ((lambda / m) * theta(j));
end





% =============================================================

end
