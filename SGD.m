function [weights, cost, wm, gm] = SGD(fcn,X,y,w)
  # mini-batch size (1% of training sample, at least 10)
  mini_batch_size = 10;#min(size(X,1),max(10,floor(0.01 * size(X,1))))
  n = floor(size(X,1) / mini_batch_size)

  # epochs for training
  epochs = 30;
  cost = zeros(n,epochs);
  velocity = zeros(size(w));

  # learning rate
  eta = 0.000001;
  # momentum
  mu = 0.0;

  # loop over epochs
  for i = 1:epochs
    # shuffle
    sel = randperm(size(X, 1))(1:n*mini_batch_size);
    X_shuffled = X(sel,:);
    y_shuffled = y(sel,:);
    X_shuffled = reshape(X_shuffled,[],size(X,2),n);
    y_shuffled = reshape(y_shuffled,[],size(y,2),n);

    for j = 1:n
      [cost(j,i), grad] = fcn(X_shuffled(:,:,j),y_shuffled(:,:,j),w);
      velocity = mu * velocity - eta * grad;
      w += velocity;
    endfor
    fprintf("cost after epoch %d: %.3g\r",i,cost(n,i));
  endfor
  fprintf("\n");
  weights = w;
endfunction
