function [nn cost] = trainNetwork(X,y,arch,lambda)
  num_w = sum(arch(1:end-1) .* arch(2:end)) + sum(arch(2:end));
  nn = zeros(1,num_w);
  cost = 0;
  start = 1;
  for l = 1:numel(arch)-1
    in = arch(l)+1;
    out = arch(l+1);
    epsilon = sqrt(12.0/in);
    nn(start:start + in*out-1) = (rand(out,in) * 2 * epsilon - epsilon)(:);
    start += in*out;
  endfor

  options = optimset('MaxIter', 20);
  costFunction = @(p) nnCostFunctionReg(p,arch,X,y,lambda);
  [nn cost] = fmincg(costFunction, nn, options);
endfunction
