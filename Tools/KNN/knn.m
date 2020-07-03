function [y_hat,sigma_pred] = knn(X_train,y_train,X_test,k)
% [y_hat,sigma_pred] = knn(X_train,y_train,X_test,k)
% performs kNN regression using euclidean distance

y_hat = zeros(size(X_test,1),1);
sigma_pred = zeros(size(X_test,1),1);

for i=1:size(X_test,1)
    dist = sqrt( sum( bsxfun(@minus,X_train,X_test(i,:)).^2 , 2 ) ); %compute euclidean distance to all points in x
    [~,I] = sort(dist,'ascend');        % get index for nearest neighbours
    y_hat(i) = mean(y_train(I(1:k)));  % predict y using k nearest points, all k points are weighted euqualiy
    if k > 1
        sigma_pred(i) = sqrt(1/(k-1) * sum((y_hat(i)-y_train(I(1:k))).^2));
    end
end

end


