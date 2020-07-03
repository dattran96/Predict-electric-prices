function [cost,grad]=linregcost(Phi,y,w,lambda)
n = length(y);

%cost = 1/(2*n)*sum((Phi*w-y).^2)+ lambda/(2*n)*sum((w(2:end)).^2));

% vectorized version:
cost = 1/(2*n)*(Phi*w-y)'*(Phi*w-y)+ lambda/(2*n)*((w(2:end))'*w(2:end));
grad = 1/n *Phi'*(Phi*w-y)+ lambda/n * [0;w(2:end)];

end


