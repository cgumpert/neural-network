X = rand(1000,1) * 2 - 1;
y = X.^2;

arch = [1,50,1];

[nn, cost, wm, gm] = trainNetwork(X,y,arch,0);

#y_test = rand(500,1) * 2 - 1;
#X_test = abs(y_test);
y_pred = predict(X,arch,nn);
plot(X,y,'r.');
hold on;
plot(X,y_pred,'bo');
