function J = nnCostFunction(weights,arch,X,y,lambda)
  J = 0;
  
  # convert array of weights into weight matrices for each layer
  start = 1;
  m = size(X,1);
  a = X';
  for l = 1:numel(arch)-1
    in = arch(l)+1;
    out = arch(l+1);
    w = reshape(weights(start:start+in*out-1),out,in);
    start += in*out;
    a = [ones(1,m); a];
    a = sigmoid(w*a);
  endfor
  costMatrix = (-y' .* log(a) - (1-y') .* log(1-a));
  J = sum(costMatrix(:))/m;
endfunction
