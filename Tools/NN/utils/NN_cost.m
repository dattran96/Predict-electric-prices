function [J, grad] = NN_cost(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_outputs, ...
                                   X, y)                              
%NNCOSTFUNCTION Implements  neural network cost function and gradient for a two layer
%neural network which performs regression.

% Reshape nn_params back into the parameters W1 and W2
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_outputs, (hidden_layer_size + 1));

n = size(X, 1); %number of observations
%% Feedforward
% 1.1) Feed forward
a1 = [ones(n,1), X];
z2 = W1*a1';
a2 = [ones(1,n);sigmoid(z2)];
z3 = W2*a2;
a3 = z3;

%% Loss
err = (y-a3').^2;
J = 1/(n*num_outputs) * sum(err(:));

%% Backpropagation
delta3 = 2*(a3-y');
delta2 = (W2(:,2:end))'*delta3.*sigmoidGradient(z2);

D1 = 1/n * delta2*a1;
D2 = 1/n * delta3*a2';

% Unroll gradients
grad = [D1(:) ; D2(:)];

end
