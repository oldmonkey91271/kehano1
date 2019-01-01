function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%
%   Inputs: X - [m x n] matrix of training samples / features (not including X0)
%
%   Outputs: X_norm - [m x n] matrix of normalized training samples 
%            mu - [1 x n] row vector of training means
%            sigma - [1 x n] row vector of training standard deviations
%
%   Normalization is : X_norm = (X - mean) / (standard_deviation)
% -----------------------------------------------------------------------------

% You need to set these values correctly
X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

% Obtain the size of the training
[m, n] = size(X);

% Compute the means of each training feature
mu = mean(X);

% Compute the standard deviations of each training feature
sigma = std(X);

% Perform the normalization
X_norm = X_norm - repmat(mu, m, 1);
X_norm = X_norm ./ repmat(sigma, m, 1);

end
