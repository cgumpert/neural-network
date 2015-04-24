function y = backtransform(in,minimum,maximum)
  y = (in .- 0.3) ./ 0.4 .* (maximum - minimum) .+ minimum;
  y = exp(y);
endfunction