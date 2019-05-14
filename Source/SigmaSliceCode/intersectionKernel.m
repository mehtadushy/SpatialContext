function K = intersectionKernel(U,V)

 %vectorized and more efficient apparantly
 K = pdist2(U, V, @(x, Y) sum(bsxfun(@min, x, Y), 2));
end
