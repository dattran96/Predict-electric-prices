function X_scaled=scale(X,mu,sigma)
% X_scale=scale(X,mu,sigma)
%  scales X by subtracting mu and dividing by sigma

temp = bsxfun(@minus,X,mu);
X_scaled = bsxfun(@rdivide, temp, sigma);
%X_scaled=(X-mu)./sigma; % implicict expansion like X-mu only works from 2016b on!

end