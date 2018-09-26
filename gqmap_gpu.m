function [mu, sigma, rou] = gqmap_gpu(options,flow)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its; K = options.K; var = options.var; gama = options.gama; dta = options.dta;
[X, W] = GaussHermite_2(K); X = gpuArray(X);  W = gpuArray(W);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ;
sqrt2 = sqrt(2); sqrtpi = sqrt(pi); sqrt2pi = sqrt(2/pi); sqrt2dpi = sqrt(2)/pi;

[M,N,~] = size(flow);
it = 1;
tor = 1e-3;
flow = gpuArray(flow);
mu = rand(M,N,2)+flow;%try single type later
sigma = rand(M,N,2,'gpuArray') + 2;% make sure it's a large initialization
rou = zeros(M,N,2,2,'gpuArray');

while 1
    mu_u = mu(:,:,1); mu_v =mu(:,:,2); sigma_u = sigma(:,:,1); sigma_v = sigma(:,:,2);
    [dmu_u, dsigma_u] = arrayfun(@node_grad, mu_u, sigma_u, flow(:,:,1));
    [dmu_v, dsigma_v] = arrayfun(@node_grad, mu_v, sigma_v, flow(:,:,2));
    [dmu1_u,dmu2_u,dsigma1_u,dsigma2_u,drou_u] = arrayfun(@edge_grad, repmat(mu_u,[1 1 2]), cat(3,circshift(mu_u,-1), circshift(mu_u,-1,2)),...
        repmat(sigma_u,[1 1 2]), cat(3,circshift(sigma_u,-1),circshift(sigma_u,-1,2)), rou(:,:,:,1));
    [dmu1_v,dmu2_v,dsigma1_v,dsigma2_v,drou_v] = arrayfun(@edge_grad, repmat(mu_v,[1 1 2]), cat(3,circshift(mu_v,-1), circshift(mu_v,-1,2)),...
        repmat(sigma_v,[1 1 2]), cat(3,circshift(sigma_v,-1),circshift(sigma_v,-1,2)), rou(:,:,:,2));
   
    dmu_u = dmu_u + sum(dmu1_u,3);dmu_v = dmu_v + sum(dmu1_v,3);
    dmu_u(2:end,:) = dmu_u(2:end,:) + dmu2_u(1:end-1,:,1); dmu_u(:,2:end) = dmu_u(:,2:end) + dmu2_u(:,1:end-1,2);
    dmu_v(2:end,:) = dmu_v(2:end,:) + dmu2_v(1:end-1,:,1); dmu_v(:,2:end) = dmu_v(:,2:end) + dmu2_v(:,1:end-1,2);
    dsigma_u = dsigma_u + sum(dsigma1_u,3); dsigma_v = dsigma_v + sum(dsigma1_v,3);
    dsigma_u(2:end,:) = dsigma_u(2:end,:) + dsigma2_u(1:end-1,:,1); dsigma_v(:,2:end) = dsigma_v(:,2:end) + dsigma2_v(:,1:end-1,2);

    step = 0.1/(1+it/1000);
    mu = mu + cat(3, dmu_u,dmu_v) * step;
    sigma = abs(sigma + cat(3,dsigma_u,dsigma_v) * step);
    rou = max(min(rou + cat(4, drou_u,drou_v) * step, 0.97), -0.97);
    %if it>700,sigma = sigma*0.999;end% quench; Tempurature gradually down.
    fprintf('#[%3d], \x0394(mu) = %d, \x0394(sigma) = %d, \x0394(rou) = %d\n',...
        it, max(abs(dmu_u(:))), max(abs(dsigma_u(:))),max(abs(drou_u(:))));
    it = it + 1;
    if it > its || it>100 && max(abs(dmu_u(:))) < tor, break; end
end

    function [du,do] = node_grad(u,o,arg)
        du = 0; do = 0;
        for k=1:K
            x = sqrt2*o*X(k) + u;
            dval = W(k)*(arg-x)/var;%potential derivative evaluation
            du = du + dval;
            do = do + dval*X(k);
        end
        du = du/sqrtpi; do = do*sqrt2pi;
    end
    function [du1,du2,do1,do2,dp] = edge_grad(u1,u2,o1,o2,p)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
        q = sqrt(1+p); r=sqrt(1-p);
        s = (q+r)/2; t = (q-r)/2;
        ds = (1/q - 1/r)/4; dt = (1/q + 1/r)/4;
        for k=1:K^2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1 = sqrt2*o1*zi+u1;
            x2 = sqrt2*o2*zj+u2;
            diff = x2-x1;
            if abs(diff) > dta, diff = 0;end
            df1 = WIWJ(k)*diff/gama;%potential derivative evaluation
            df2 = -df1;
            du1 = du1 + df1;
            du2 = du2 + df2;
            do1 = do1 + df1*zi;
            do2 = do2 + df2*zj;
            dp =  dp + (o1*df1*(ds*XI(k)+dt*XJ(k)) + o2*df2*(dt*XI(k)+ds*XJ(k)));
        end
        du1 = du1/pi; du2 = du2/pi;
        do1 = do1*sqrt2dpi; do2 = do2*sqrt2dpi;
        dp = dp*sqrt2dpi;
    end
end