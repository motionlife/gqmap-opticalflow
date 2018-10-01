function flow = findMixMax(a, mu1, sigma1, mu2, sigma2)
    % Find the maximum value of mixture of each grid node variable.
    [M,N,L] = size(mu1);
    flou = zeros(M,N);
    flov = zeros(M,N);
    parfor i=1:M
        for j=1:N
            flou(i,j) = findmin(a, mu1(i,j,:),sigma1(i,j,:), L);
            flov(i,j) = findmin(a, mu2(i,j,:),sigma2(i,j,:), L);
        end
    end
    flow = cat(3,flou,flov);
end

function flo = findmin(a,u,o,L)
    spk_y=Inf;sid=1;sqrt2pi=sqrt(2*pi);
    for l1=1:L
        vl=0;
        for l2=1:L
            vl = vl - a(l2)*exp(-(u(l1)-u(l2))^2/(2*o(l2)^2))/(sqrt2pi*o(l2));
        end
        if vl<spk_y,spk_y=vl;sid=l1;end
    end
    [x,fval] = fminbnd(@(x) neg_mixture(x,a,u,o,L), min(u), max(u));
    if fval < spk_y
        flo = x;
    else
        flo = u(sid);
    end
end

function v = neg_mixture(x,a,u,o,L)
    v=0;sqrt2pi=sqrt(2*pi);
    for l=1:L
        v = v + a(l)*exp(-(x-u(l))^2/(2*o(l)^2))/(sqrt2pi*o(l));
    end
    v=-v;
end
% function flow = findMixMax(alpha, muu, sigmau, muv, sigmav)
% [M,N,~] = size(muu);
% flou = zeros(M,N);
% flov = zeros(M,N);

% parfor i=1:M
%     for j=1:N
%         u1 = muu(i,j,:);
%         o1 = sigmau(i,j,:);
%         u2 = muv(i,j,:);
%         o2 = sigmav(i,j,:);
%         %calculate flow-u
%         func1 = @(x) -sum(arrayfun(@(a,u,o) a*normpdf(x,u,o), alpha,u1,o1));
%         [spike, uid] = min(arrayfun(func1,u1));
%         [x,fval] = fminbnd(func1, min(u1), max(u1));
%         if fval < spike
%             flou(i,j) = x;
%         else
%             flou(i,j) = u1(uid);
%         end
%         % calculate flow-v 
%         func2 = @(x) -sum(arrayfun(@(a,u,o) a*normpdf(x,u,o), alpha,u2,o2));
%         [spike, uid] = min(arrayfun(func2,u2));
%         [x,fval] = fminbnd(func2, min(u2), max(u2));
%         if fval < spike
%             flov(i,j) = x;
%         else
%             flov(i,j) = u2(uid);
%         end
%     end
% end
% flow =  cat(3,flou,flov);