function grad = backpropagate(arch,delta,w,z,a,lambda = 0,dActivation = @sigmoidGradient)
% function grad = backpropagate(arch, delta, w, z, a, lambda)
%
% calculate the gradient of the network cost function with respect to all weights using backpropagation
%	 
% parameters:	 
% arch   ... architecture of the network as list of nodes per layer, including input and output
%           layers, excluding bias nodes
% delta  ... error of prediction in output layer
% w      ... cell array (dim = (numel(arch) - 1) x 1) containing the weight matrices for each layer
%            dim w{l} = arch(l+1) x (arch(l) + 1)
% z      ... cell array (dim = numel(arch) x 1) containing the network inputs for each layer
%            z{l+1} = w{l} * a{l}	 
% a      ... cell array (dim = numel(arch) x 1) containing the activations for each layer
%	     a{l} = [ones(1,m); sigmoid(z{l})]
% lambda ... L2 regularisation parameter (default = 0)
%
% result [grad]:
% grad ... gradient of the network cost function with respect to the weights 
%	 
  # get number of weights
  num_w = sum(arch(1:end-1) .* arch(2:end)) + sum(arch(2:end));
  # initialise return value
  grad = zeros(num_w,1);
  # helper variable for unrolling gradient matrices
  ende = num_w;
  # number of training examples
  m = size(a{1},2);

  # add pseudo delta for bias nodes
  delta = [ones(1,m); delta];
  
  # perform backward propagation
  for l = numel(arch)-1:-1:1
    # calculate gradient with respect to weights in this  layer
    nabla = (delta(2:end,:) * a{l}')./m;
    # calculate delta for layer l
    delta = w{l}' * delta(2:end,:) .* dActivation([ones(1,m);z{l}]);
    # add regularisation term
    if (lambda > 0)
      nabla(:,2:end) += lambda .* w{l}(:,2:end)./m;
    endif
    # unroll gradient matrix
    grad(ende-prod(size(nabla))+1:ende) = nabla(:);
    ende -= prod(size(nabla));
  endfor
endfunction
