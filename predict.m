function pred = predict(arch,weights,X)
  start = 1;
  m = size(X,1);
  pred = X';
  for l = 1:numel(arch)-1
    in = arch(l)+1;
    out = arch(l+1);
    w = reshape(weights(start:start+in*out-1),out,in);
    start += in*out;
    pred = [ones(1,m); a];
    pred = sigmoid(w*a);
  endfor
endfunction
