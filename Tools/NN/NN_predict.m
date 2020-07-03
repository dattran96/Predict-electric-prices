function y_hat = NN_predict(W1, W2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(W1, W2, X) outputs the predicted label of X given the
%   trained weights of a neural network (W1,W2)

m = size(X, 1);
num_labels = size(W2, 1);

h1 = sigmoid([ones(m, 1) X] * W1');
y_hat = [ones(m, 1) h1] * W2';

end