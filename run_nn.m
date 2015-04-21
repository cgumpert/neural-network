# 1. network architecture
# specify number of nodes per layer in <architecture>
# - first layer is the input layer
# - last layer is the output layer
# - bias nodes should not be included

data = load('sherpa_nn_test.dat');
X = data(:,1:18);
y = data(:,21);
y = log(y);
y = (y - min(y))./(max(y) - min(y));

% split into training and test set
train_size = 0.5;
train_idx = floor(train_size * size(X,1));
sel = randperm(size(X, 1));
X_train = X(sel(1:train_idx),:);
y_train = y(sel(1:train_idx),:);
X_test = X(sel(train_idx+1:end),:);
y_test = y(sel(train_idx+1:end),:);

arch = [18,25,1];

[nn cost] = trainNetwork(X_train,y_train,arch,1);
fprintf("training cost: %.3f\n",cost);
y_pred = predict(arch,nn,X_test);
fprintf("test cost: %.3f",nnCostFunction(nn,arch,X_test,y_test));

break
load('../../coursera/ML/machine-learning-ex4/ex4/ex4data1.mat');
load('../../coursera/ML/machine-learning-ex4/ex4/ex4weights.mat');
m = size(X, 1);
y_vec = zeros(m,10);
for i = 1:m
  y_vec(i,y(i)) = 1;
end


displayData(X(sel, :));
pause;

[nn cost] = trainNetwork(X,y_vec,arch,1);

pred = predict(arch,nn,X);

[max_val,y_pred] = max(pred,[],2) ;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y)) * 100);
