function [w] = trainlinreg(Phi, y, lambda)
%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
%regularization parameter lambda

w0 = zeros(size(Phi,2),1);
costFunction = @(weights) linregcost(Phi, y, weights, lambda);

% Now, costFunction is a function that takes in only one argument
options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton','GradObj','on');
% Minimize using fminunc
w = fminunc(costFunction, w0, options);

end
