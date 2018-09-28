function [mu, sigma, rou, AEPE, Energy] = gqmap_gpu_mixture(options,I1,I2,GRDT)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its;  K = options.K;  L=3;
epsn = options.epsn;    lambdad = options.lambdad;  lambdas = options.lambdas;
minu=options.minu;  maxu=options.maxu;  minv=options.minv;  maxv=options.maxv;
I1=gpuArray(I1); I2=gpuArray(I2);K2 = K^2; sqrt2=sqrt(2); corr_tor=0.999;
[X, W] = GaussHermite_2(K); X = gpuArray(X);  W = gpuArray(W);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ; XIXJ = XI.*XJ; XI2aXJ2 = XI.^2 + XJ.^2;XI2mXJ2 = XI.^2 - XJ.^2;%XI2 = 2*XI.^2; XJ2 = 2*XJ.^2; 
[M,N] = size(I1); rg=1; M_=(1+rg):(M-rg); N_=(1+rg):(N-rg);
rfc=6;	rfc2=2^rfc;	I2_cont = interp2(I2,rfc,'cubic');	[MM, NN] = size(I2_cont);
[ns,ms] = meshgrid(gpuArray(1:N),gpuArray(1:M));

%initialize gpu data
alpha = rand(L,1,'gpuArray'); alpha = reshape(alpha./sum(alpha),1,1,L);
muu = minu+rand(M,N,L,'gpuArray')*(maxu-minu);
muv = minv+rand(M,N,L,'gpuArray')*(maxv-minv);
sigmau = rand(M,N,L,'gpuArray') + 3;%(maxu-minu);% make sure it's a large initialization
sigmav = rand(M,N,L,'gpuArray') + 3;%(maxv-minv);
pn = zeros(M,N,L,'gpuArray');
rou = zeros(M,N,L,2,2,'gpuArray');
%for verbose info and profiling
bestat=1;best_aepe=Inf; AEPE=ones(its,1,'gpuArray')*17;
Energy = zeros(its,1,'gpuArray');%logP = zeros(its,1,'gpuArray');
it = 1;tor = 1e-4;tic;
while 1
    step = 0.01/(1+it/5000);
    %dim_1:(M)--dim_2:(N)--dim_3:L(mixture components)
    [dan,dmuu,dmuv,dsigmau,dsigmav,dpn,nEnergy] = arrayfun(@node_grad_spectral,repmat(alpha,M,N,1),muu,muv,sigmau,sigmav,pn,ms,ns);
    %dim_1:(M)--dim_2:(N)--dim_3:L(mixture components)--dim_4:(vertical/horizontal edges)--dim_5:(u/v edges)
    [dae,dmu1,dmu2,dsigma1,dsigma2,drou,eEnergy] = arrayfun(@edge_grad_spectral,repmat(alpha,M,N,1,2,2),cat(5,repmat(muu,[1 1 1 2]),repmat(muv,[1 1 1 2])),...
        cat(5,cat(4,circshift(muu,-1), circshift(muu,-1,2)),cat(4,circshift(muv,-1), circshift(muv,-1,2))),...
        cat(5,repmat(sigmau,[1 1 1 2]),repmat(sigmav,[1 1 1 2])),...
        cat(5,cat(4,circshift(sigmau,-1), circshift(sigmau,-1,2)),cat(4,circshift(sigmav,-1), circshift(sigmav,-1,2))),rou);
    
    dalpha = sum(sum(dan(M_,N_,:),1),2) + sum(sum(sum(sum(dae(M_,N_,:,:,:),1),2),4),5); 
    dmuu = dmuu + sum(dmu1(:,:,:,:,1),4) + circshift(dmu2(:,:,:,1,1),1) + circshift(dmu2(:,:,:,2,1),1,2);
    dmuv = dmuv + sum(dmu1(:,:,:,:,2),4) + circshift(dmu2(:,:,:,1,2),1) + circshift(dmu2(:,:,:,2,2),1,2);
    dsigmau = dsigmau + sum(dsigma1(:,:,:,:,1),4) + circshift(dsigma2(:,:,:,1,1),1) + circshift(dsigma2(:,:,:,2,1),1,2);
    dsigmav = dsigmav + sum(dsigma1(:,:,:,:,2),4) + circshift(dsigma2(:,:,:,1,2),1) + circshift(dsigma2(:,:,:,2,2),1,2);
    muu(M_,N_,:) = min(max(muu(M_,N_,:) + dmuu(M_,N_,:) * step, minu), maxu);
    muv(M_,N_,:) = min(max(muv(M_,N_,:) + dmuv(M_,N_,:) * step, minv), maxv);
    sigmau(M_,N_,:) = min(max(sigmau(M_,N_,:) + dsigmau(M_,N_,:) * step,0.01),23);
    sigmav(M_,N_,:) = min(max(sigmav(M_,N_,:) + dsigmav(M_,N_,:) * step,0.01),23);
    rou(M_,N_,:,:,:) = min(max(rou(M_,N_,:,:,:) + drou(M_,N_,:,:,:) * step, -corr_tor), corr_tor);
    pn(M_,N_,:) = min(max(pn(M_,N_,:) + dpn(M_,N_,:) * step, -corr_tor), corr_tor);
    
    Energy(it) = sum(sum(sum(nEnergy(M_,N_,:)))) + sum(sum(sum(sum(sum(eEnergy(M_,N_,:,:,:))))));
    %how to set a reasonable dalpha step size
    if it>500, alpha = projsplx(alpha + dalpha * step * 1E-7);end    
    
    ptdmu=abs(dmuu(M_,N_,:)); ptdsigma=abs(dsigmau(M_,N_,:)); 
    ptdmu = mean(ptdmu(:)); ptdsigma=mean(ptdsigma(:));
    fprintf('[%3d], \x0394(mu) = %e, \x0394(sigma) = %e, Energy = %e \n', it, ptdmu, ptdsigma, Energy(it));

    %Profile every n iterations 
    if mod(it,500)==0 || it==1
        [alf,mu_u,sig_u,mu_v,sig_v] = gather(alpha,muu,sigmau, muv,sigmav);
        flow = findMixMax(alf, mu_u, sig_u, mu_v, sig_v);
        flc = flowToColor(flow);
        imshow(flc);
        %imwrite(flc,[options.dir,'/',num2str(it),'.png']);
        aepe = mean(mean(sqrt(sum((GRDT(M_,N_,:) - flow(M_,N_,:)).^2,3))));AEPE(it)=aepe;
        if aepe < best_aepe, bestat = it; best_aepe = aepe;end
        % logP(it) = =>How to profile logP??? 
        fprintf('[%3d], AEPE=%e, best at# %e \n', it,  aepe, bestat);
    end
    it = it + 1;
    if it > its || ptdmu < tor, break; end
end
toc;

    function [da,du1,du2,do1,do2,dp,Ei] = node_grad_spectral(a,u1,u2,o1,o2,p,m,n)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0; Ei = 0;
        %if (m<=rg||m>M-rg||n<=rg||n>N-rg),return;end
        s = (sqrt(1+p)+sqrt(1-p))/2;
        t = (sqrt(1+p)-sqrt(1-p))/2;
        pr = 1 - p^2; sqrtpr = sqrt(pr);
        o1pr = sqrt2/(o1*pr); o2pr = sqrt2/(o2*pr);
        for k=1:K2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1=sqrt2*o1*zi+u1; x2=sqrt2*o2*zj+u2;
            fval= WIWJ(k)*node_pot(x1,x2,m,n);
            if a~=0
                dp  = dp + fval*(p - p*XI2aXJ2(k) + 2*XIXJ(k));
                du1 = du1 + fval*(zi - p*zj);
                du2 = du2 + fval*(zj - p*zi);
                do1 = do1 + fval*(XI2aXJ2(k) - 1 + XI2mXJ2(k)/sqrtpr);
                do2 = do2 + fval*(XI2aXJ2(k) - 1 - XI2mXJ2(k)/sqrtpr);
            end
            Ei = Ei+fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        do1 = a*do1/pi/o1;
        do2 = a*do2/pi/o2;
        dp = a*dp/pi/pr;
        da = Ei/pi;
        Ei = a*da;
    end

    function [da,du1,du2,do1,do2,dp,Ei] = edge_grad_spectral(a,u1,u2,o1,o2,p)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0; Ei=0;
        s = (sqrt(1+p)+sqrt(1-p))/2;
        t = (sqrt(1+p)-sqrt(1-p))/2;
        pr = 1 - p^2; sqrtpr = sqrt(pr);
        o1pr = sqrt2/(o1*pr); o2pr = sqrt2/(o2*pr);
        for k=1:K2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1=sqrt2*o1*zi+u1; x2=sqrt2*o2*zj+u2;
            fval = WIWJ(k)*edge_pot(x1,x2);
            if a~=0
                dp  = dp + fval*(p - p*XI2aXJ2(k) + 2*XIXJ(k));
                du1 = du1 + fval*(zi - p*zj);
                du2 = du2 + fval*(zj - p*zi);
                do1 = do1 + fval*(XI2aXJ2(k) - 1 + XI2mXJ2(k)/sqrtpr);
                do2 = do2 + fval*(XI2aXJ2(k) - 1 - XI2mXJ2(k)/sqrtpr);
            end
            Ei = Ei+fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        do1 = a*do1/pi/o1;
        do2 = a*do2/pi/o2;
        dp = a*dp/pi/pr;
        da = Ei/pi;
        Ei = a*da;
    end

    function npt = node_pot(u,v,m,n)
        npt = -lambdad*sqrt(epsn + (I1(m,n) - I2_cont(min(max(round((m+v-1)*rfc2+1),1),MM), min(max(round((n+u-1)*rfc2+1),1),NN)))^2);
    end
    function ept = edge_pot(x1,x2)
        ept = -lambdas*sqrt(epsn+(x1-x2)^2);
    end
mu=gather(cat(3,muu,muv));
sigma=gather(cat(3,sigmau,sigmav));
rou=gather(rou);
AEPE=gather(AEPE);
Energy=gather(Energy);
end