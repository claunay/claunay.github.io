% Sampling Determinantal point processes (DPPs) from the
%   sequential thinning algorithm
%
% NB: You can find a short script to test this algorithm at the end 
%       of the file.
% 
% Input:
% - K: any DPP kernel, an hermitian matrix with 
%       eigenvalues between 0 and 1
% [- q: the associated Bernoulli probabilities, 
%       from the function sequential_thinning_init, optional]
%
% Output:
% - A: DPP realization of kernel K
% [- X: Bernoulli realization, before the sequential thinning procedure]
%
% C. Launay, B. Galerne, A. Desolneux (c) 2018

function [A,X] = sequential_thinning_simulation(K,q)
if(nargin<2)
    q = sequential_thinning_init(K);
end

opts.LT = true;

% Draw dominating Bernouilli process
X = find(rand(size(q)) < q);

% Initialization
A = []; % Indices of selected points
B = []; % Indices of unseclected points
NB = []; % New indices of B between points of X 
L = []; % Cholesky decomposition of (I-K)_B
ImK = eye(size(K))-K;
previous_point = 0;

for k=X % SHOULD BE A LOOP ON X
    % update list of new point not in Y since previous point of X:
    NB = [NB, (previous_point+1):(k-1)];
    % update Cholesky decomposition if B has new points:
    if(~isempty(NB))
        L = cholesky_update_add_bloc(L, ImK(B,NB), ImK(NB,NB), opts);
    end
    % update B:
    B = [B,NB];
    if(isempty(B))
        H = K([A,k],[A,k]);
    else
        J = linsolve(L, K(B,[A,k]));
        H = K([A,k],[A,k]) + J'*J;
    end
    v = H(1:(end-1),end);
    pk = H(end,end) - v'*(H(1:(end-1),1:(end-1))\v);
    if rand() < pk/q(k) % the point is selected
        A = [A,k];
        NB = [];
    else % the point is not selected and put in the list NB
        NB = k; 
    end
    % update previous point
    previous_point = k;
end
end


function LM = cholesky_update_add_bloc(LA, B, C, opts)
    if(isempty(LA))
        LM = chol(C,'lower');
    else
        V = linsolve(LA, B, opts);
        LX = chol(C - V'*V,'lower');
        LM = [LA, zeros(size(B)); V' , LX];
    end
end


function q = sequential_thinning_init(K)

N = size(K,1);
[L,err] = chol(eye(N)-K,'lower');

if err == 0
    B = triu(K(1:(N-1),2:N));
    opts.LT = true;
    q = (diag(K))' + [0, sum(triu(linsolve(L(1:(N-1),1:(N-1)),B,opts)).^2)];
else %slower procedure if I-K is singular
    q = ones(1,N);
    k=1;
    ImK = eye(N,N) - K;
    q(k) = K(k,k);
    opts.SYM = true;
    while( (k < N) && (q(k) < 1))
        k = k+1;
        q(k) = real(K(k,k) + K(k,1:(k-1))*linsolve(ImK(1:(k-1),1:(k-1)), K(1:(k-1),k), opts));
    end        
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing code for sampling experiments with a random DPP kernel
%
% N = 5000; % Size of the groundset
% expectCard = 20; % Expected size of the sample
% projsym = @(K) 0.5*(K'+K)
% 
% % Definition of the random DPP kernel
% [Q,R] = qr(rand(N,N));
% lambda = rand(1,N);
% K = projsym(Q'*diag(lambda)*Q);
% 
% % Control of the expected cardinal by binary search   
% mu = lambda./(1-lambda);
% l = 0; r = 10^10;
% E = sum(r*mu./(1+r*mu)); alpha = 0;
% while E ~= expectCard && l<r
%    % if l>r
%    %     break
%    % end
%     alpha = (l+r)/2;
%     E =  sum(alpha*mu./(alpha*mu +1));
%     if E < expectCard
%         l = alpha+eps;
%     elseif E > expectCard
%         r = alpha-eps;
%     end    
%     disp(num2str(r));
% end
% lambda_r = r*mu./(1+r*mu);
% K_r = projsym(Q'*diag(lambda_r)*Q);
%     
% % Sampling 
% Y = sequential_thinning_simulation(K_r);
