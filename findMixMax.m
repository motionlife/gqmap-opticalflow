function flow = findMixMax(alpha, muu, sigmau, muv, sigmav)
% Find the maximum value of mixture of each grid node variable.
[M,N,~] = size(muu);
flou = zeros(M,N);
flov = zeros(M,N);

parfor i=1:M
    for j=1:N
        u1 = muu(i,j,:);
        o1 = sigmau(i,j,:);
        u2 = muv(i,j,:);
        o2 = sigmav(i,j,:);
        %calculate flow-u
        func1 = @(x) -sum(arrayfun(@(a,u,o) a*normpdf(x,u,o), alpha,u1,o1));
        [spike, uid] = min(arrayfun(func1,u1));
        [x,fval] = fminbnd(func1, min(u1), max(u1));
        if fval < spike
            flou(i,j) = x;
        else
            flou(i,j) = u1(uid);
        end
        % calculate flow-v 
        func2 = @(x) -sum(arrayfun(@(a,u,o) a*normpdf(x,u,o), alpha,u2,o2));
        [spike, uid] = min(arrayfun(func2,u2));
        [x,fval] = fminbnd(func2, min(u2), max(u2));
        if fval < spike
            flov(i,j) = x;
        else
            flov(i,j) = u2(uid);
        end
    end
end
flow =  cat(3,flou,flov);