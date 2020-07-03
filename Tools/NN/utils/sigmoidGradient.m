function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z

% ====================== YOUR CODE HERE ======================

g = sigmoid(z).*(1.-sigmoid(z));

% =============================================================

end
