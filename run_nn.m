# 1. network architecture
# specify number of nodes per layer in <architecture>
# - first layer is the input layer
# - last layer is the output layer
# - bias nodes should not be included

arch = [400,25,10];

load('../machine-learning-ex4/ex4/ex4data1.mat');
m = size(X, 1);
y_vec = zeros(m,10);
for i = 1:m
  y_vec(i,y(i)) = 1;
end


% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

displayData(X(sel, :));
pause;

[nn cost] = trainNetwork(X,y_vec,arch,1);

pred = predict(arch,nn,X);

[max_val,y_pred] = max(pred,[],2) ;
fprintf('\nTraining Set Accuracy: %f\n', mean(double(y_pred == y)) * 100);
