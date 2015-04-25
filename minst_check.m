# 1. network architecture
# specify number of nodes per layer in <architecture>
# - first layer is the input layer
# - last layer is the output layer
# - bias nodes should not be included

load('../../coursera/ML/machine-learning-ex4/ex4/ex4data1.mat');
#load('../../coursera/ML/machine-learning-ex4/ex4/ex4weights.mat');
m = size(X, 1);
y_vec = zeros(m,10);
for i = 1:m
  y_vec(i,y(i)) = 1;
end

arch = [400,25,10];
sel = randperm(size(X, 1));

#displayData(X(sel(1:100), :));

[nn, c] = trainNetwork(X,y_vec,arch,0.1);

pred = predict(X,arch,nn);

[max_val,y_pred] = max(pred,[],2) ;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y)) * 100);
