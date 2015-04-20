# 1. network architecture
# specify number of nodes per layer in <architecture>
# - first layer is the input layer
# - last layer is the output layer
# - bias nodes should not be included

arch = [400,25,10];

load('../../coursera/ML/machine-learning-ex4/ex4/ex4data1.mat');
m = size(X, 1);
y_vec = zeros(10,m);
for i = 1:m
    y_vec(i,y(i)) = 1;
end
y = y_vec;

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
pause;

[nn cost] = trainNetwork(X,y,arch,0);

pred = predict(arch,nn,X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
