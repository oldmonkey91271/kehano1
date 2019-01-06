function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.
%
%   g = SIGMOID(z) = 1 / (1 + e^-z)

% we return g with the same dimesions as input z 
g = zeros(size(z));

% g = 1 / (1 + e^-z)
g = 1 ./ (1 .+ exp(-1 * z));

end
