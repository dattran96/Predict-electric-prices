function X_resclaled = rescale(X,mu,sigma)
% X_denorm = denormalize(X,mu,sigma)

X_resclaled = X.*sigma + mu;

end
