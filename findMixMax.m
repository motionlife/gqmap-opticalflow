function x = findMixMax(mu, sigma, alpha)
%FINDMIXMAX Find maximum of single-variable gaussian mixutre on fixed interval
%construct the mixtures for each node
[~,N] = size(mu);
x = zeros(1,N);
% options = optimset('MaxFunEvals',5000,'MaxIter',1000);
for i =1:N
    idx = find(alpha(:,i));
    mus = mu(idx,i);
    func = @(x) -sum(arrayfun(@(m,s,a) a*normpdf(x,m,s),mus,sigma(idx,i),alpha(idx,i)));
    [spike, uid] = min(arrayfun(func,mus));
%   [xp,fval] = fminbnd(func, min(mus), max(mus), options);
    [xp,fval] = fminbnd(func, min(mus), max(mus));
    if fval < spike
        x(i) = xp;
    else
        x(i) = mus(uid);
    end
end


