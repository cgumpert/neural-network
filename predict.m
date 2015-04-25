function [y, z, a, w] = predict(X,arch,weights)
%function [y, z, a, w] = predict(X,arch,weights)
%
%predicts the output values for a given set if inputs
%
% parameters:
% X       ... k input values for m examples (size = m x k)
% arch    ... architecture of the network as list of nodes per layer, including input and output layer, excluding bias nodes
% weights ... weight vector representing the network
%
% result [y, z, a, w]:
% y ... predicted output values (size = m x l with being the number of output nodes)
% z ... cell array (dim = numel(arch) x 1) containing the network inputs for each layer
%       z{l+1} = w{l} * a{l}	 
% a ... cell array (dim = numel(arch) x 1) containing the activations for each layer
%	a{l} = [ones(1,m); sigmoid(z{l})]
% w ... cell array (dim = (numel(arch) - 1) x 1) containing the weight matrices for each layer
%       dim w{l} = arch(l+1) x (arch(l) + 1)
%
% current implementation only supports calling the function with either one or four return values	 
%
  m = size(X,1);
  l = arch(end);

  # initialise return values
  y = zeros(m,l);
  z = 0;
  a = 0;
  w = 0;

  if (nargout == 1)
    # we only want to know the prediction
    # -> do not store intermediate values

    # helper variable for reshaping weight vector into matrices
    start = 1;
    # set activations to input
    a = X';
    # feed-forward through network
    for l = 1:numel(arch)-1
      in = arch(l)+1;
      out = arch(l+1);
      w = reshape(weights(start:start+in*out-1),out,in);
      start += in*out;
      a = [ones(1,m); a];
      a = sigmoid(w*a);
    endfor
    y = a';
  elseif (nargout == 4)
    # store intermediate values as well
    z = cell(numel(arch),1);
    a = cell(numel(arch),1);
    w = cell(numel(arch)-1,1);

    # helper variable for reshaping weight vector into matrices
    start = 1;
    # set activations to input
    a{1} = X';
    z{1} = a{1};
    for l = 1:numel(arch)-1
      in = arch(l)+1;
      out = arch(l+1);
      w{l} = reshape(weights(start:start+in*out-1),out,in);
      start += in*out;
      a{l} = [ones(1,m); a{l}];
      z{l+1} = w{l}*a{l};
      a{l+1} = sigmoid(z{l+1});
    endfor
    y = a{end}';
  else
    # wrong number of output values
    print_usage();
  endif
endfunction
