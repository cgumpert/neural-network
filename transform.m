function [y, minimum, maximum] = transform(in)
  in = log(in);
  minimum = min(in);
  maximum = max(in);
  y = (in .- minimum)./(maximum - minimum) .* 0.4 .+ 0.3;
endfunction
