function w = linreg(Phi,y,lambda)
% computes weights from normal equation
I=eye(size(Phi,2));
I(1,1)=0;

w = (Phi'*Phi+ lambda*I) \ (Phi'*y);
end














% 