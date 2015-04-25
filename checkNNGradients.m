function diff = checkNNGradients(fcn,args)
%function diff = checkNNGradients(fcn,args)
%
%performs a numerical check of the gradient 
%
%parameters:
% fcn  ... a function handle which takes args as argument and returns [val, grad]
%           where val is the function value and gradient the gradient with respect
%           to the args parameters at the given parameter point
% args ... parameter point at which the function and its gradient are evaluated
%
% result [diff]:
% diff ... normalised difference of the returned gradient and the numerical computatiion
%          norm(grad - num_grad) / norm(grad + num_grad)
%	 

  # intialise return values
  diff = 0;

  # get function value and gradient
  [val, grad] = fcn(args);

  # compute gradient numerically
  numgrad = zeros(size(args));
  perturb = zeros(size(args));
  
  e = 1e-4;
  for p = 1:numel(args)
    % Set perturbation vector
    perturb(p) = e;
    loss1 = fcn(args - perturb);
    loss2 = fcn(args + perturb);
    % Compute Numerical Gradient
    numgrad(p) = (loss2 - loss1) / (2*e);
    perturb(p) = 0;
  end

  #disp([numgrad grad]);

  # calculate normalised difference
  diff = norm(numgrad-grad)/norm(numgrad+grad);
end
