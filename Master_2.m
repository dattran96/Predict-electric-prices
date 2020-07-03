%------------MACHINE LEARNING & ENERGY WS 19/20------------
%-----------------------Miniproject-----------------------
%   To keep thing simple, firstly predict only one hour, that is hour 14.
%   To plot the LEARNING CURVE, we train model with different training size
%   
%   Expecting:
%   Increasing data helps reducing high varience, 
%   But too much data (1000 samples) results high bias.
%   
%----------------------------------------------------------
%
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

%Learning Curve Plot with different sample size 
CV_Error_OverSampleSize = [];
Train_Error_OverSampleSize = [];
NumberofSample = [5 70 150 250 300 350 400 1000];

for k=1:length(NumberofSample)

%training data, predict h14:
X_train=table2array(Data_PV(1:NumberofSample(k),15))+table2array(Data_Wind(1:NumberofSample(k),15))-table2array(Data_Load(1:NumberofSample(k),15));
y_train=table2array(Data_Price(1:NumberofSample(k),15));
%testing data, predict h14
X_true=table2array(Data_PV_true(:,15))+table2array(Data_Wind_true(:,15))-table2array(Data_Load_true(:,15));
y_true=table2array(Data_Price_true(:,15));

%load('Data_large')
%X_train=Data_large(:,1);
%y_train=Data_large(:,2);

% validation data, predict h12:
%X_val=table2array(Data_PV(500:699,15))+table2array(Data_Wind(500:699,15))-table2array(Data_Load(500:699,15));
%y_val=table2array(Data_Price(500:699,15));


% figure()
% scatter(X_train,y_train);
% title('training data')
% xlabel('Wind+PV-Load');
% ylabel('Price');
% title('training data visualization,we can see that the data points forms approximately a curve, so possible to use Linear Regression')
% hold on;
%scatter(X_val,y_val);
%title('Validation data');
%hold off;


%% 2) find best lambda by using cross-validation and loss function 
    % then training to obtain weight and calculate CV-Error, Train-Error
    
% possible lambdas
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
        X_train_cv = X_train(ii~=j,:); % training data for X
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

%Estimate Error on Test Set
X_poly_test   = polyfeatures(X_true,p);                   % generate polynomial features
X_poly_test   = scale(X_poly_test,mu,sigma);                  % scale
Phi_poly_test = [ones(length(X_true),1) X_poly_test];
y_pred_test = predict(w_opt,Phi_poly_test);
RMSE_test =loss(y_true, y_pred_test);

%Print ou RMSE and Lambda of different sample size
if(k==1)
    fprintf('%dst traning with %d Samples: The best lambda is: %f,RMSE is %f \n',k,NumberofSample(k),best_lambda,RMSE_test);
elseif(k==2)
    fprintf('%dnd traning with %d Samples: The best lambda is: %f,RMSE is %f \n',k,NumberofSample(k),best_lambda,RMSE_test);
elseif(k==3)
    fprintf('%drd traning with %d Samples: The best lambda is: %f,RMSE is %f \n',k,NumberofSample(k),best_lambda,RMSE_test);
else         
    fprintf('%dth Traning with %d Samples: The best lambda is: %f,RMSE is %f \n',k,NumberofSample(k),best_lambda,RMSE_test);
end
%To plot learning curve over size of data, we need CV-Error and Train-Error corresponding to this best Lambda:
CV_Error_OverSampleSize = [CV_Error_OverSampleSize;CV_error_lambdas(ind)];
Train_Error_OverSampleSize =[Train_Error_OverSampleSize;Training_error_lambdas(ind)];
end

%% 3) Plot Learning Curve and Model
figure()
plot(NumberofSample,CV_Error_OverSampleSize);
hold on;
plot(NumberofSample,Train_Error_OverSampleSize)
hold off;
legend('Cross Validation Error','Train Error')
title('Error depending on sample size, we solved high varience, since both errors converge as increasing data')


%Plot model and training data, test(true) data together
figure() 
scatter(X_train,y_train);
hold on;
scatter(X_true,y_true);
%w_opt=[50;0;0;0;0;0;0;0;0;0;0];
plotpolyfit(min(X_train), max(X_train), mu, sigma, w_opt, p);
%axis([-5 85 20 100])
legend('training data','test data','model')
title('%Plot model and training data, test(true) data together')

%%4) Conclusion
 fprintf('-------------------------------------  \n');
 fprintf('Incresing Data Training Size helps reducing High Varience, but 1000 samples is redundant, since Data repeats the same pattern every year \n');
 fprintf('Use 400 Sample of Data is enough(one year of data) \n');
 imshow('PV_sum_Everyday.png');
 title('Data repeats every one year, so use one year Data is enough');