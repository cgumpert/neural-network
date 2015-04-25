function [J, grad] = CrossEntropy(X,y,arch,weights,lambda = 0)
%function [J, grad] = CrossEntropy(X,y,arch,weights,lambda)
%
%calculates the cross entropy for the given network architecture and data
%
% parameters:
% X       ... k input values for m training examples (size = m x k)
% y       ... l target output values for m training examples (size = m x l)
% arch    ... architecture of the network as list of nodes per layer, including input and output layer, excluding bias nodes
% weights ... weight vector representing the network
% lambda  ... L2 regularisation parameter (default = 0)
%
% result [J, grad]:
% J    ... cross entropy defined as 1/m sum_(i=1)^m sum_(j=1)^l [ -y_(ij) * log(a_(ij)) - (1 - y_(ij)) * log(1 - a_(ij)) ]
%	   with: m      ... number of training examples
%     	         l      ... number of output nodes
%	         y_(ij) ... j-th target output value of the i-th training example
%	         a_(ij) ... j-th predicted output value of the i-th training example
% grad ... gradient of the cost function with respect to the weights
%
  # initialise return values
  J = 0;
  grad = zeros(size(weights));

  # number of examples
  m = size(X,1);
  
  if (nargout < 2)
    # we only want to know the predicted output
    # -> skip expensive calculation of the gradient
    y_pred = predict(X,arch,weights);
  elseif (nargout == 2)
    # calculate prediction and store intermediated activations and
    # weight matrices for gradient calculation

    # initialise delta matrices for each layer
    d = cell(numel(arch),1);
    nabla = cell(numel(arch)-1,1);

    # get prediction and intermediate values from feed-forward propagation
    [y_pred, z, a, w] = predict(X,arch,weights);

    # calculate delta for output layer
    d{end} = (a{end} - y'); # .* sigmoidGradient(z{end});
    d{end} = [ones(1,m); d{end}];

    # helper variable for unrolling gradient matrices
    ende = size(weights,1);
    # perform backward propagation
    for l = numel(arch)-1:-1:1
      # calculate delta for layer l
      d{l} = w{l}' * d{l+1}(2:end,:) .* sigmoidGradient([ones(1,m);z{l}]);
      # calculate gradient with respect to weights in this  layer
      nabla{l} = (d{l+1}(2:end,:) * a{l}')./m;
      # add regularisation term
      if (lambda > 0)
	nabla{l}(:,2:end) += lambda .* w{l}(:,2:end)./m;
      endif
      # unroll gradient matrix
      grad(ende-prod(size(nabla{l}))+1:ende) = nabla{l}(:);
      ende -= prod(size(nabla{l}));
    endfor
  else
    # wrong number of output values
    print_usage();
    return;
  endif

  # calculate cost
  costMatrix = (-y .* log(y_pred) - (1 - y) .* log(1 - y_pred));
  J = sum(costMatrix(:))/m;
  # add regularisation penalty
  if (lambda > 0)
    J += sum(weights(:).^2)*lambda/(2*m);
    # do not regularise weights for bias nodes
    sw2 = 0;
    start = 1;
    for l = 1:numel(arch)-1
      sw2 += sum(weights(start:start+arch(l+1)-1).^2);
      start += (arch(l) + 1) * arch(l+1);
    endfor
    J -= sw2*lambda/(2*m);
  endif
endfunction
