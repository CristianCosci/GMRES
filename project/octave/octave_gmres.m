% dim = 10000;
% A = sprand(dim, dim, 1.);
% b = rand(dim, 1);
% x0 = zeros(dim, 1);

% [x, flag, relres, iter, resvec] = gmres(A, b, [], 1e-10, 10);

clear

function A = loadFunctionFromData(dataPath)
    disp(["🦍 Loading file: "  dataPath])
    load(dataPath)
    
%   se nel file è presente una `struct` allora vuol dire 
%   che sto utilizzando il nuovo formato e va convertito
%   per essere utilizzato da Octave
    if isstruct(Problem.A)
        n = length(Problem.A.jc)-1;
        ii = repelems(1:n, [1:n; diff(Problem.A.jc)]);
        A = sparse(Problem.A.ir+1, ii, Problem.A.data);
    else
        A = Problem.A;
    end
endfunction

function [A b x0] = generateData(dim, den, seed=69, randx0=false)
    rand("seed", seed);
    A = sprand(dim, dim, den);
    b = rand(dim, 1);
    
    if (randx0)
        x0 = rand(dim, 1);
    else
        x0 = zeros(dim, 1);
    end
endfunction

function [x res] = mygmres(A, b, x0, k, tol)
    n = size(b)(1);
    r0 = b - A * x0;
    beta = norm(r0, 2);
    
    H = zeros(k+2, k+1);
    Q = zeros(n, k+1);
    Q(:, 1) = r0/norm(r0, 2);
    
    en = zeros(k+2, 1);
    en(1) = 1;

    for j = 1:k  %% TODO: ricontrollare
%         imagesc(H);
        v = A*Q(:, j);   
        for i = 1:j
            H(i,j) = Q(:,i)' * v;
            v = v - H(i, j) * Q(:, i);
        end
        
        H(j+1, j) = norm(v, 2);
        Q(:, j+1) = v / H(j+1, j);

%         if (abs(H(j+1, j)) > tol)
%             Q(:, j+1) = Q(:,j+1)/H(j+1,j);
%         end
        
        e1 = en(1:j+2);

        y =(H(1:j+2,1:j+1))\(beta *e1);
        res(j) = norm(H(1:j+2,1:j+1)*y - beta*e1, 2);

        if (res(j) < tol)
            x = Q(:, 1:j+1)*y +x0;
            disp('Raggiunta Tolleranza, stop 🚀');
            imshow(H);
            return;
        end
    end
    x = Q(:, 1:j+1)*y +x0;
    disp('Raggiunto massimo numero di Iterazioni 🐌');
    imagesc(H);

endfunction

[A, b, x0] = generateData(100, .5);
A
b
x0

[x, res] = mygmres(A, b, x0, 50, 1e-10);
input("> ");

x
x_true = A\b
max(x_true-x)

subplot(2, 1, 1)
plot(res);
subplot(2, 1, 2)
semilogy(res);

input("> ");
