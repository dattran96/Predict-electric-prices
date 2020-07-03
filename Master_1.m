%------------MACHINE LEARNING & ENERGY WS 19/20------------
%-----------------------Miniproject-----------------------
% To keep thing simple, firstly predict only one hour, that is hour 14.
% Use linear regression to train model with only 30 sample of training data.
% To ensure i.i.d, use row 1 to 40 for training and use row 500 to 699 for
% validation
% 
%   Expecting: High Varience
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% 
% 
clear;close all;clc;

%% 1) import and plot data
%Training Data
Data_Load = readtable('Load_train.csv');
Data_Wind = readtable('Wind_train.csv');
Data_PV = readtable('PV_train.csv');
Data_Price = readtable('Price_train.csv');

%Testing Data
Data_Load_true=readtable('Load_test.csv');
Data_Wind_true=readtable('Wind_test.csv');
Data_PV_true=readtable('PV_test.csv');
Data_Price_true=readtable('Price_test.csv');

%Learning Curve Plot
%CV_Error_OverSampleSize = [];
%Train_Error_OverSampleSize = [];

%training data, predict h14:
NumberofSample = 30;
X_train=table2array(Data_PV(1:NumberofSample,15))+table2array(Data_Wind(1:NumberofSample,15))-table2array(Data_Load(1:NumberofSample,15));
y_train=table2array(Data_Price(1:NumberofSample,15));

%testing data, predict h14
X_true=table2array(Data_PV_true(:,15))+table2array(Data_Wind_true(:,15))-table2array(Data_Load_true(:,15));
y_true=table2array(Data_Price_true(:,15));


% validation data, predict h14:
X_val=table2array(Data_PV(500:699,15))+table2array(Data_Wind(500:699,15))-table2array(Data_Load(500:699,15));
y_val=table2array(Data_Price(500:699,15));


figure()
scatter(X_train,y_train);
hold on;
scatter(X_val,y_val);
legend('training data','validation data')
hold off;


%% 2) find best lambda by using cross-validation and loss function
% possible lambdas
lambdas=[0 0.001 0.005 0.01 0.1 0.5 1 2 5 10]; 
n = 10; % number of folds

% predict and loss functions
predict = @(weights,Phi_Mat) Phi_Mat*weights;
loss = @(y_true, y_pred) sqrt(1/length(y_true)*sum((y_true-y_pred).^2)); % RMSE loss function
%Index to split training and cross-validation data
ii = ceil(randperm(size(X_train,1))/size(X_train,1)*n); %create vector for random splitting of data


% preallocate
error_train = zeros(length(lambdas),1);
error_val   = zeros(length(lambdas),1); 
ws = cell(length(lambdas),1); % for storing weights
Index_smallestCV=cell(length(lambdas),1);
ws_n = cell(n,1); % for storing weights every time we change fold training

CV_error_lambdas=zeros(1,length(lambdas));
Training_error_lambdas=zeros(1,length(lambdas));

for i=1:length(lambdas)
    %For each lambda, we have n different RMSE CV-Error
    RMSE_cv  = zeros(1,n);
    RMSE_train  = zeros(1,n);
    for j=1:n
        %Seperate Data
        X_train_cv = X_train(ii~=j,:); % training data for X
        y_train_cv = y_train(ii~=j);   % training data for y
        %X_test_cv = X_train(ii==j,:);  % validation data for X
        %y_test_cv = y_train(ii==j);    % validation data for y 
        X_test_cv = X_val;
        y_test_cv = y_val;
        
        % create polynomial design matrix
        p=10; %desgin polynomial degree
        X_poly_train = polyfeatures(X_train_cv,p);                  % generate polynomial features
        [X_poly_train, mu, sigma] = featureNormalize(X_poly_train); % normalize
        Phi_poly_train = [ones(length(X_train_cv),1) X_poly_train]; % add ones

        X_poly_val   = polyfeatures(X_test_cv,p);                   % generate polynomial features
        X_poly_val   = scale(X_poly_val,mu,sigma);                  % scale
        Phi_poly_val = [ones(length(X_test_cv),1) X_poly_val];      % add ones
        %Training
        w_tmp = trainlinreg(Phi_poly_train,y_train_cv,lambdas(i));
        ws_n{j} = w_tmp;
        %Predict and CV-Error Estimation
        y_pred_val = predict(w_tmp,Phi_poly_val);
        RMSE_cv(j) = loss(y_test_cv, y_pred_val);
        %Train-Error Estimation
        y_pred_train = predict(w_tmp,Phi_poly_train);
        RMSE_train(j)=loss(y_train_cv, y_pred_train);
    end
    [~,Index_weight] = min(RMSE_cv);
    ws{i} = ws_n{Index_weight};
    Index_smallestCV{i} = Index_weight;
    CV_error_lambdas(i)=mean(RMSE_cv);
    Training_error_lambdas(i)=mean(RMSE_train);
end


%Result : Best Lambda, Weights
[~,ind]=min(CV_error_lambdas);
best_lambda = lambdas(ind);
w_opt=ws{ind};
fprintf('The best lambda is: %f \n',best_lambda);

%To plot learning curve over size of data, we need CV-Error and Train-Error corresponding to this best Lambda:
%CV_Error_OverSampleSize = [CV_Error_OverSampleSize;CV_error_lambdas(ind)];
%Train_Error_OverSampleSize =[Train_Error_OverSampleSize;Training_error_lambdas(ind)];


%% 3) test polynomial model
%Estimate Error on Test Set
X_poly_test   = polyfeatures(X_true,p);                   
X_poly_test   = scale(X_poly_test,mu,sigma);                  % scale
Phi_poly_test = [ones(length(X_true),1) X_poly_test];
y_train_pred= predict(w_opt,Phi_poly_test);
RMSE_test =loss(y_true, y_train_pred);
fprintf('RMSE of Test Set is: %f \n',RMSE_test);



%% 4) plot relative figures
 % plot CV error
figure()
bar(CV_error_lambdas);
set(gca,'xtick',1:length(lambdas));
xlabel('Model number');
ylabel('Crossvalidation Error');

%Plot CV-Error and Training-Error
figure()
plot(Training_error_lambdas,'DisplayName','training error');hold on;plot(CV_error_lambdas,'DisplayName','validation error');hold off;
legend('show')
title('High varience still there !!! Regardless how good lambda is')
axis([1 10 2 8])

%Plot model and training data, test(true) data together
figure() 
scatter(X_train,y_train);
hold on;
scatter(X_true,y_true);
plotpolyfit(min(X_train), max(X_train), mu, sigma, w_opt, p);
legend('training data','test data','model')
title('regularized fit')

