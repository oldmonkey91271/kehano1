function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
%
% INPUTS:
%     theta = [n+1 x 1] optimization coefficient vector
%     X     = [m x n+1] training samples
% OUTPUTS
%     p     = [m x 1] vector or probabilities
% -----------------------------------------------------------------------------
[m, n1] = size(X);    % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% Compute the predictions from the hypothesis for each input (m x 1)
p = round(sigmoid(X * theta));
 
end
