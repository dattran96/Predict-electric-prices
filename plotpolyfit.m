function plotpolyfit(min_x, max_x, mu, sigma, w, p)
%PLOTFIT Plots a learned polynomial regression fit over an existing figure.
%Also works with linear regression.
%   plotpolyfit(min_x, max_x, mu, sigma, w, p) plots the learned polynomial
%   fit with power p and feature normalization (mu, sigma).

% Hold on to the current figure
hold on;

% We plot a range slightly bigger than the min and max values to get
% an idea of how the fit will vary outside the range of the data points
x = (min_x - 10: 0.05 : max_x + 10)';

% Map the X values 
X_poly = polyfeatures(x, p);
X_poly = bsxfun(@minus, X_poly, mu);
X_poly = bsxfun(@rdivide, X_poly, sigma);

% Add ones
Phi_poly = [ones(size(x, 1), 1) X_poly];

% Plot
plot(x, Phi_poly * w, '--', 'LineWidth', 2)

% Hold off to the current figure
hold off

end
