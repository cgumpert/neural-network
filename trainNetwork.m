function [nn, cost, wm, gm] = trainNetwork(X,y,arch,lambda,w=0)
  num_w = sum(arch(1:end-1) .* arch(2:end)) + sum(arch(2:end));
  if w == 0
    nn = zeros(num_w,1);
    cost = 0;
    start = 1;
    for l = 1:numel(arch)-1
      in = arch(l)+1;
      out = arch(l+1);
      epsilon = sqrt(12.0/in);
      nn(start:start + in*out-1) = (rand(out,in) * 2 * epsilon - epsilon)(:);
      start += in*out;
    endfor
  else
    nn = w;
  endif

  [nn, cost, wm, gm] = SGD(@(X_,y_,w_) CrossEntropy(X_,y_,arch,w_,lambda),X,y,nn);
#  checkNNGradients(@(weights) CrossEntropy(X(1:5,:),y(1:5,:),arch,weights,lambda),nn)
#  options = optimset('MaxIter', 100);
#  costFunction = @(p) CrossEntropy(X,y,arch,p,lambda);
#  [nn, cost] = fmincg(costFunction, nn, options);
endfunction
