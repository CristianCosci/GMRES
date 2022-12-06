dim = 10000;
A = sprand(dim, dim, 1.);
b = rand(dim, 1);
x0 = zeros(dim, 1);

[x, flag, relres, iter, resvec] = gmres(A, b, [], 1e-10, 10);