function [class_predict] = evaluate(y_pred_test,y_true,t)

temp = ones(size(y_true,1),1);
t=temp.*t;

p = abs(y_pred_test-y_true);

class_predict = bsxfun(@le,p,t); 

end







