# 1. network architecture
# specify number of nodes per layer in <architecture>
# - first layer is the input layer
# - last layer is the output layer
# - bias nodes should not be included

data = load('sherpa_nn_test.dat');
X = data(:,1:18);
y_raw = data(:,21);
log_y = log10(y_raw);
y = (log_y .- min(log_y))./(max(log_y) - min(log_y)) .* 0.4 .+ 0.3;

% split into training and test set
train_size = 0.7;
train_idx = floor(train_size * size(X,1));
sel = randperm(size(X, 1));
X_train = X(sel(1:train_idx),:);
y_train = y(sel(1:train_idx),:);
X_test = X(sel(train_idx+1:end),:);
y_test = y(sel(train_idx+1:end),:);

arch = [18,50,10,1];

[nn cost] = trainNetwork(X_train,y_train,arch,0);
fprintf("training cost: %.3f\n",cost);
y_pred = predict(arch,nn,X_test);
fprintf("test cost: %.3f\n",nnCostFunction(nn,arch,X_test,y_test));
break

load('../machine-learning-ex4/ex4/ex4data1.mat');
#load('../../coursera/ML/machine-learning-ex4/ex4/ex4weights.mat');
m = size(X, 1);
y_vec = zeros(m,10);
for i = 1:m
  y_vec(i,y(i)) = 1;
end

sel = randperm(size(X, 1));

displayData(X(sel(1:100), :));
pause;

[nn cost] = trainNetwork(X,y_vec,arch,1);

pred = predict(arch,nn,X);

[max_val,y_pred] = max(pred,[],2) ;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y)) * 100);
