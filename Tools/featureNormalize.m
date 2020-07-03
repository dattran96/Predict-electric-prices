function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. Note that mean(X) returns the mean along the columns. The same goes
%   for std(X).
% 
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);

sigma = std(X_norm);
X_norm = bsxfun(@rdivide, X_norm, sigma);

%or:
% 
% mu = mean(X);
% X_norm = X - mu;  % implicict expansion like X-mu only works in 2016b!
% sigma = std(X);
% X_norm = X_norm./sigma;

end