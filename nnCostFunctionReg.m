function [J grad] = nnCostFunctionReg(weights,arch,X,y,lambda)
  J = 0;
  grad = zeros(size(weights));
  
  # convert array of weights into weight matrices for each layer
  start = 1;
  m = size(X,1);
  a = cell(numel(arch),1);
  z = cell(numel(arch),1);
  d = cell(numel(arch),1);
  w = cell(numel(arch)-1,1);
  nabla = cell(numel(arch)-1,1);

  a{1} = X';
  z{1} = a{1};
  s_wb2 = 0;
  for l = 1:numel(arch)-1
    in = arch(l)+1;
    out = arch(l+1);
    w{l} = reshape(weights(start:start+in*out-1),out,in);
    s_wb2 += sum(w{l}(:,1).^2);
    start += in*out;
    a{l} = [ones(1,m); a{l}];
    z{l+1} = w{l}*a{l};
    a{l+1} = sigmoid(z{l+1});
    nabla{l} = zeros(out,in);
  endfor
  
  costMatrix1 = (-y' .* log(a{end}) - (1-y') .* log(1-a{end}));
  costMatrix = (a{end} - y').^2;
  J = sum(costMatrix(:))/(2*m);
  J += (sum(weights(:).^2) - s_wb2)*lambda/(2*m);

  d{end} = (a{end} - y') .* sigmoidGradient(z{end});
  d{end} = [ones(1,m); d{end}];
  ende = size(weights,1);
  for l = numel(arch)-1:-1:1
    d{l} = w{l}' * d{l+1}(2:end,:) .* sigmoidGradient([ones(1,m);z{l}]);
    if isempty(nabla{l})
      nabla{l} = zeros(arch(l+1),arch(l)+1);
    endif
    nabla{l} = (d{l+1}(2:end,:) * a{l}')./m; 
    nabla{l}(:,2:end) += lambda .* w{l}(:,2:end)./m;
    grad(ende-prod(size(nabla{l}))+1:ende) = nabla{l}(:);
    ende -= prod(size(nabla{l}));
  endfor
endfunction
