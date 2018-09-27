function [mu, sigma, rou, best_aepe, bestat] = gqmap_gpuV3(options,I1,I2,GRDT)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its; K = options.K; sg = options.sg; lambdad = options.lambdad; lambdas=options.lambdas;
I1=gpuArray(I1); I2=gpuArray(I2);
[X, W] = GaussHermite_2(K); X = gpuArray(X);  W = gpuArray(W);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ;
XIXJ = XI.*XJ;
XI2 = 2*XI.^2;
XJ2 = 2*XJ.^2;
% XI2aXJ2 = XI.^2 + XJ.^2;
% XI2mXJ2 = XI.^2 - XJ.^2;
K2 = K^2; sqrt2=sqrt(2);
[M,N] = size(I1); rg=1; wdo=(2*rg+1)^2; [MM, NN] = size(I2); M_= rg+1:M-rg;	N_=rg+1:N-rg; 
rfc=4;	rfc2=2^rfc;	
I2_cont = interp2(I2,rfc,'cubic');
[I2u,I2v] = imgradientxy(I2,'prewitt');
%   ko  = [ 0.004711  0.069321  0.245410  0.361117  0.245410  0.069321  0.004711];
%   do  = [ 0.018708  0.125376  0.193091  0.000000 -0.193091 -0.125376 -0.018708];
% I2u = conv2(do,ko,I2,'same');I2v = conv2(ko,do,I2,'same');
I2u_cont = interp2(I2u,rfc,'cubic'); 
I2v_cont = interp2(I2v,rfc,'cubic');

[ns,ms] = meshgrid(gpuArray(1:N),gpuArray(1:M));
it = 1; tor = 1e-2;best_it=1;best_aepe=Inf;
muu = (rand(M,N,'gpuArray')-0.5)*0.3;%gpuArray(xini(:,:,1));%
muv = (rand(M,N,'gpuArray')-0.5)*0.3;%gpuArray(xini(:,:,2));%
sigmau = rand(M,N,'gpuArray') + 3;% make sure it's a large initialization
sigmav = rand(M,N,'gpuArray') + 3;
pn = zeros(M,N,'gpuArray');
rou = zeros(M,N,2,2,'gpuArray');
% timestamp = datestr(now, 'yyyymmddHHMMSS');
% mkdir(timestamp);
while 1
    [dmuu,dmuv,dsigmau,dsigmav,dpn] = arrayfun(@node_grad_spectral,muu,muv,sigmau,sigmav,pn,ms,ns);
    [dmu1,dmu2,dsigma1,dsigma2,drou] = arrayfun(@edge_grad_spectral, cat(4,repmat(muu,[1 1 2]),repmat(muv,[1 1 2])),...
        cat(4,cat(3,circshift(muu,-1), circshift(muu,-1,2)),cat(3,circshift(muv,-1), circshift(muv,-1,2))),...
        cat(4,repmat(sigmau,[1 1 2]),repmat(sigmav,[1 1 2])),...
        cat(4,cat(3,circshift(sigmau,-1), circshift(sigmau,-1,2)),cat(3,circshift(sigmav,-1), circshift(sigmav,-1,2))),rou);

    dmuu = dmuu + sum(dmu1(:,:,:,1),3) + circshift(dmu2(:,:,1,1),1) + circshift(dmu2(:,:,2,1),1,2);
    dmuv = dmuv + sum(dmu1(:,:,:,2),3) + circshift(dmu2(:,:,1,2),1) + circshift(dmu2(:,:,2,2),1,2);
    dsigmau = dsigmau + sum(dsigma1(:,:,:,1),3) + circshift(dsigma2(:,:,1,1),1) + circshift(dsigma2(:,:,2,1),1,2);
    dsigmav = dsigmav + sum(dsigma1(:,:,:,2),3) + circshift(dsigma2(:,:,1,2),1) + circshift(dsigma2(:,:,2,2),1,2);
    
    step = 0.03/(1+it/1000);
    muu(M_,N_) = min(max(muu(M_,N_) + dmuu(M_,N_) * step,options.minu),options.maxu);
    muv(M_,N_) = min(max(muv(M_,N_) + dmuv(M_,N_) * step,options.minv),options.maxv);
    sigmau(M_,N_) = min(max(sigmau(M_,N_) + dsigmau(M_,N_) * step,1e-3),20);
    sigmav(M_,N_) = min(max(sigmav(M_,N_) + dsigmav(M_,N_) * step,1e-3),20);
    rou(M_,N_,:,:) = max(min(rou(M_,N_,:,:) + drou(M_,N_,:,:) * step, 0.98), -0.98);
    pn(M_,N_) = max(min(pn(M_,N_) + dpn(M_,N_) * step, 0.98), -0.98);
% 	if it>1000,sigmau(M_,N_) = sigmau(M_,N_)*0.991;sigmav(M_,N_) = sigmav(M_,N_)*0.991;end
     aepe = mean(mean(sqrt((GRDT(M_,N_,1)-muu(M_,N_)).^2+(GRDT(M_,N_,2)-muv(M_,N_)).^2)));
    if aepe < best_aepe, bestat = it; best_aepe = aepe;end
    flc = gather(flowToColor(cat(3, muu(M_,N_),muv(M_,N_))));
    imshow(flc);
    if mod(it,10)==0, imwrite(flc,[options.timestamp,'/',num2str(it),'.png']);end
    fprintf('#[%3d], \x0394(mu) = %d, \x0394(sigma) = %d, \x0394(rou) = %d, AEPE=%d, best at#%d\n',...
        it, mean(abs(dmuu(:))), mean(abs(dsigmau(:))),mean(abs(drou(:))), aepe, bestat);
    it = it + 1;
    if it > its || it>1250 && mean(abs(dmuu(:))) < tor, break; end  
end

    function [du1,du2,do1,do2,dp] = node_grad(u1,u2,o1,o2,p,m,n)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
        if (m<=rg||m>M-rg||n<=rg||n>N-rg),return;end
        sqrt1p2=sqrt(1-p^2); pdivrt = p/sqrt1p2;
        for k=1:K2
            x1=sqrt2*o1*XI(k) + u1;
            x2=sqrt2*o2*(p*XI(k) + sqrt1p2*XJ(k)) + u2;
            super = 0;
            for i=(m-rg):(m+rg)
                for j=(n-rg):(n+rg)
                    super = super + sqrt(sg+(I1(i,j) - I2_cont(min(max(round((i+x2-1)*rfc2+1),1),MM),min(max(round((j+x1-1)*rfc2+1),1),NN)))^2);
                end
            end
            eval = WIWJ(k)*super/wdo;
            dp = dp + eval*(p - p*XJ2(k) + 2*sqrt1p2*XIXJ(k));
            du1 = du1 + eval*(XI(k) - pdivrt*XJ(k));
            du2 = du2 + eval*XJ(k);
            do1 = do1 + eval*(XI2(k) - 2*pdivrt*XIXJ(k) - 1);
            do2 = do2 + eval*(XJ2(k) + 2*pdivrt*XIXJ(k) - 1);
        end
        du1 = -lambdad*du1*sqrt2/o1/pi;
        du2 = -lambdad*du2*sqrt2/sqrt1p2/o2/pi;
        do1 = -lambdad*do1/o1/pi;
        do2 = -lambdad*do2/o2/pi;
        dp = -lambdad*dp/(1-p^2)/pi;
    end
    function [du1,du2,do1,do2,dp] = node_grad_spectral(u1,u2,o1,o2,p,m,n)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
        if (m<=rg||m>M-rg||n<=rg||n>N-rg),return;end
        q = sqrt(1+p); r=sqrt(1-p);
        s = (q+r)/2; t = (q-r)/2;
        ds = (1/q - 1/r)/4; dt = (1/q + 1/r)/4;
        for k=1:K2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1=sqrt2*o1*zi+u1; x2=sqrt2*o2*zj+u2;
            m_ = min(max(round((m+x2-1)*rfc2+1),1),MM);
            n_ = min(max(round((n+x1-1)*rfc2+1),1),NN);
            diff = I1(m,n)-I2_cont(m_,n_); deno = sqrt(sg + diff^2);
            super1 = 0; super2=0;
            for i=(m-rg):(m+rg)
                for j=(n-rg):(n+rg)
                    super1 = super1 + diff*I2u_cont(m_,n_)/deno;
                    super2=super2+diff*I2v_cont(m_,n_)/deno;
                end
            end
%             df1 = WIWJ(k)*diff*I2u_cont(m_,n_)/deno;
%             df2 = WIWJ(k)*diff*I2v_cont(m_,n_)/deno;
            df1 = WIWJ(k)*super1/wdo;
            df2 = WIWJ(k)*super2/wdo;
            dp =  dp+ o1*df1*(ds*XI(k)+dt*XJ(k)) + o2*df2*(dt*XI(k)+ds*XJ(k));
            du1 = du1 + df1;
            du2 = du2 + df2;
            do1 = do1 + df1*zi;
            do2 = do2 + df2*zj;
        end
        du1 = -lambdad*du1/pi;
        du2 = -lambdad*du2/pi;
        do1 = -lambdad*do1*sqrt2/pi;
        do2 = -lambdad*do2*sqrt2/pi;
        dp = -lambdad*dp*sqrt2/pi;
    end
    function [du1,du2,do1,do2,dp] = edge_grad(u1,u2,o1,o2,p)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
        sqrt1p2=sqrt(1-p^2); pdivrt = p/sqrt1p2;
        for k=1:K2
            eval = WIWJ(k)*sqrt(sg + (sqrt2*o1*XI(k)+u1 - sqrt2*o2*(p*XI(k)+sqrt1p2*XJ(k))-u2).^2);
            dp = dp + eval*(p - p*XJ2(k) + 2*sqrt1p2*XIXJ(k));
            du1 = du1 + eval*(XI(k) - pdivrt*XJ(k));
            du2 = du2 + eval*XJ(k);
            do1 = do1 + eval*(XI2(k) - 2*pdivrt*XIXJ(k) - 1);
            do2 = do2 + eval*(XJ2(k) + 2*pdivrt*XIXJ(k) - 1);
        end
        du1 = -lambdas*du1*sqrt2/o1/pi;
        du2 = -lambdas*du2*sqrt2/sqrt1p2/o2/pi;
        do1 = -lambdas*do1/o1/pi;
        do2 = -lambdas*do2/o2/pi;
        dp = -lambdas*dp/(1-p^2)/pi;
    end
    function [du1, du2, do1, do2, dp] = edge_grad_spectral(u1,u2,o1,o2,p)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
        q = sqrt(1+p); r=sqrt(1-p);
        s = (q+r)/2; t = (q-r)/2;
        ds = (1/q - 1/r)/4; dt = (1/q + 1/r)/4;
        for k=1:K2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1=sqrt2*o1*zi+u1; x2=sqrt2*o2*zj+u2;
            df1 = WIWJ(k)*(x1-x2)/sqrt(sg+(x1-x2)^2);
            df2 = -df1;
            dp =  dp+ o1*df1*(ds*XI(k)+dt*XJ(k)) + o2*df2*(dt*XI(k)+ds*XJ(k));
            du1 = du1 + df1;
            du2 = du2 + df2;
            do1 = do1 + df1*zi;
            do2 = do2 + df2*zj;
        end
        du1 = -lambdas*du1/pi;
        du2 = -lambdas*du2/pi;
        do1 = -lambdas*do1*sqrt2/pi;
        do2 = -lambdas*do2*sqrt2/pi;
        dp = -lambdas*dp*sqrt2/pi;
    end
mu=cat(3,muu,muv);
sigma=cat(3,sigmau,sigmav);
end