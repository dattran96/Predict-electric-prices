function X_scaled=scale(X,mu,sigma)
% X_norm=normalize(X,mu,sigma)

temp = bsxfun(@minus,X,mu);
X_scaled = bsxfun(@rdivide, temp, sigma);

%X_scaled=(X-mu)./sigma; % implicict expansion like X-mu only works from 2016b on!

end