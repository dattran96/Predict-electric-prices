function [W1,W2] = NN_train(X_train,y_train,hidden_layer_size,max_iter)
% [W1,W2] = NN_train(X_train,y_train,hidden_layer_size,max_iter)
% Trains a Neural Network with one output by minimizing the squared error.
% Uses 2 hidden layers with sigmoid activation.
% Returns:
%  weight matrices W1,W2.
%Inputs:
% NN_train trains a two layer NN for regression
% X_train / input  / NxM matrix
% y_train / output / Nx1 vector
% hidden_layer_size / number of neurons in the hidden layers / integer
% max_iter / maximum iterations in traning / integer

%% get params
input_layer_size  = size(X_train, 2);
num_outputs = size(y_train,2); 
options = optimset('MaxIter', max_iter);

%% randomly initialize weights
initial_W1 = NN_randInitializeWeights(input_layer_size, hidden_layer_size);
initial_W2 = NN_randInitializeWeights(hidden_layer_size, num_outputs);
initial_nn_params = [initial_W1(:) ; initial_W2(:)];

%% Train
costFunction = @(p) NN_cost(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_outputs, X_train, y_train);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% Obtain W1 and W2 back from nn_params
W1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

W2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_outputs, (hidden_layer_size + 1));
             
end