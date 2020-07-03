%------------MACHINE LEARNING & ENERGY WS 19/20------------
%-----------------------Miniproject-----------------------
% Use 24 different model to predict 24 hours by using linear regression
%           We saw in Master_2 that, 1000 sample is unnesesary.
%           400 samples(a year of data) is enough.
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
CV_Error_OverSampleSize = [];
Train_Error_OverSampleSize = [];
Avarage_Error_24 = [];
for hour=2:25 
%training data, predict h14:
NumberofSample = 400;
X_train=table2array(Data_PV(1:NumberofSample,hour))+table2array(Data_Wind(1:NumberofSample,hour))-table2array(Data_Load(1:NumberofSample,hour));
y_train=table2array(Data_Price(1:NumberofSample,hour));
%testing data, predict h14
X_true=table2array(Data_PV_true(:,hour))+table2array(Data_Wind_true(:,hour))-table2array(Data_Load_true(:,hour));
y_true=table2array(Data_Price_true(:,hour));



%% 2) find best lambda by using cross-validation and loss function 
    % then training to obtain weight and calculate CV-Error, Train-Error
lambdas=[0 0.001 0.005 0.01 0.1 0.5 1 2 5 10]; 
n = 10; % number of folds

% predict and loss functions
predict = @(weights,Phi_Mat) Phi_Mat*weights;
loss = @(y_true, y_pred) sqrt(1/length(y_true)*sum((y_true-y_pred).^2)); % RMSE loss function
%Index to split training and cross-validation data
ii = ceil(randperm(size(X_train,1))/size(X_train,1)*n); %create vector for random splitting of data
%CV-Error for n different lamdas
%CV_error=zeros(1,length(lambdas));

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
        X_train_cv = X_train(ii~=j,:); % tr aining data for X
        y_train_cv = y_train(ii~=j);   % training data for y
        X_test_cv = X_train(ii==j,:);  % validation data for X
        y_test_cv = y_train(ii==j);    % validation data for y
        
        % create polynomial design matrix
        p=10;
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
%fprintf('The best lambda is: %f \n',best_lambda);
%To plot learning curve over size of data, we need CV-Error and Train-Error corresponding to this best Lambda:
CV_Error_OverSampleSize = [CV_Error_OverSampleSize;CV_error_lambdas(ind)];
Train_Error_OverSampleSize =[Train_Error_OverSampleSize;Training_error_lambdas(ind)];


%Estimate Error on Test Set
X_poly_test   = polyfeatures(X_true,p);                   % generate polynomial features
X_poly_test   = scale(X_poly_test,mu,sigma);                  % scale
Phi_poly_test = [ones(length(X_true),1) X_poly_test];
y_pred_test = predict(w_opt,Phi_poly_test);
RMSE_test =loss(y_true, y_pred_test);
%fprintf('RMSE of hour number %f is: %f \n',hour,RMSE_test);
Avarage_Error_24 = [Avarage_Error_24 RMSE_test];
fprintf('Training and Calculate RMSE ... \n');
end

%% 3)Calculate root mean square of all 24 hours
real_RMSE= sqrt(sum(Avarage_Error_24.^2)/24);

%% 4)Print RMSE price of each hour and RMSE price of all hour
%Print out RMSE each hour
for hour_print =1 :24
   fprintf('RMSE of hour number %d is: %f \n',hour_print,Avarage_Error_24(1,hour_print));
end
%Print out RMSE of overall average 24h
fprintf('Overall RMSE is: %f \n',real_RMSE);

plot(Avarage_Error_24);
xlabel('Hour');
ylabel('RMSE price');
title('Error of each hour')
axis([1 24 1 14])

