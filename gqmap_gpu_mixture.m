function [mu, sigma, rou, Energy] = gqmap_gpu_mixture(options,I1,I2,~)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its;  K = options.K;  L=3;
epsn = options.epsn;    lambdad = options.lambdad;  lambdas=options.lambdas;
minu=options.minu;  maxu=options.maxu;  minv=options.minv;  maxv=options.maxv;
I1=gpuArray(I1); I2=gpuArray(I2);K2 = K^2; sqrt2=sqrt(2); corr_tor=0.999;
[X, W] = GaussHermite_2(K); X = gpuArray(X);  W = gpuArray(W);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ; XIXJ = XI.*XJ; XI2aXJ2 = XI.^2 + XJ.^2;XI2mXJ2 = XI.^2 - XJ.^2;%XI2 = 2*XI.^2; XJ2 = 2*XJ.^2; 
[M,N] = size(I1); rg=1; M_=(1+rg):(M-rg); N_=(1+rg):(N-rg);
rfc=2;	rfc2=2^rfc;	I2_cont = interp2(I2,rfc,'cubic');	[MM, NN] = size(I2_cont);
[ns,ms] = meshgrid(gpuArray(1:N),gpuArray(1:M));

%initialize gpu data
alpha = rand(L,1,'gpuArray'); alpha = reshape(alpha./sum(alpha),1,1,L);
muu = minu+rand(M,N,L,'gpuArray')*(maxu-minu);
muv = minv+rand(M,N,L,'gpuArray')*(maxv-minv);
sigmau = rand(M,N,L,'gpuArray') + (maxu-minu);% make sure it's a large initialization
sigmav = rand(M,N,L,'gpuArray') + (maxv-minv);
pn = zeros(M,N,L,'gpuArray');
rou = zeros(M,N,L,2,2,'gpuArray');
%for verbose info
% bestat=1;best_aepe=Inf; AEPE=ones(its,1,'gpuArray')*17;
Energy = zeros(its,1,'gpuArray');%logP = zeros(its,1,'gpuArray');
it = 1;tor = 1e-4;tic;
while 1
    step = 0.005/(1+it/5000);
    %dim_1:(M)--dim_2:(N)--dim_3:L(mixture components)
    [dmuu,dmuv,dsigmau,dsigmav,dpn,Nenergy] = arrayfun(@node_grad_spectral,repmat(alpha,M,N,1),muu,muv,sigmau,sigmav,pn,ms,ns);
    %dim_1:(M)--dim_2:(N)--dim_3:L(mixture components)--dim_4:(vertical/horizontal edges)--dim_5:(u/v edges)
    [dmu1,dmu2,dsigma1,dsigma2,drou,Eenergy] = arrayfun(@edge_grad_spectral,repmat(alpha,M,N,1,2,2),cat(5,repmat(muu,[1 1 1 2]),repmat(muv,[1 1 1 2])),...
        cat(5,cat(4,circshift(muu,-1), circshift(muu,-1,2)),cat(4,circshift(muv,-1), circshift(muv,-1,2))),...
        cat(5,repmat(sigmau,[1 1 1 2]),repmat(sigmav,[1 1 1 2])),...
        cat(5,cat(4,circshift(sigmau,-1), circshift(sigmau,-1,2)),cat(4,circshift(sigmav,-1), circshift(sigmav,-1,2))),rou);
    
    dalpha = sum(sum(Nenergy(M_,N_,:),1),2) + sum(sum(sum(sum(Eenergy(M_,N_,:,:,:),1),2),4),5); 
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
    
    Energy(it) = sum(dalpha);
    alpha = alpha + dalpha./Energy(it) * step;
    % logP(it) = sum(sum(sum(logPN(M_,N_,:)))) + sum(sum(sum(sum(sum(logPE(M_,N_,:,:,:))))));
    ptdmu=abs(dmuu(M_,N_,:)); ptdsigma=abs(dsigmau(M_,N_,:));
    fprintf('[%3d], \x0394(mu) = %f, \x0394(sigma) = %f, Energy=%f', it, mean(ptdmu(:)), mean(ptdsigma(:)), Energy(it));
    % aepe = mean(mean(sqrt((GRDT(M_,N_,1)-muu(M_,N_)).^2+(GRDT(M_,N_,2)-muv(M_,N_)).^2)));AEPE(it)=aepe;
    % if aepe < best_aepe, bestat = it; best_aepe = aepe;end
    % if mod(it,200)==0||it==1
    %     flc = gather(flowToColor(cat(3, muu(M_,N_),muv(M_,N_))));
    %     imshow(flc);
    %     %imwrite(flc,[options.dir,'/',num2str(it),'.png']);
    % end
    % fprintf('[%3d], \x0394(mu) = %d, \x0394(sigma) = %d, AEPE=%d, Energy=%d, best at#%d\n', it, ptdmu, ptdsigma, aepe,Energy(it), bestat);
    it = it + 1;
    if it > its || ptdmu < tor, break; end
end
toc;

% function [du1,du2,do1,do2,dp] = node_grad(u1,u2,o1,o2,p,m,n)
%     du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
%     %if (m<=rg||m>M-rg||n<=rg||n>N-rg),return;end
%     sqrt1p2=sqrt(1-p^2); pdivrt = p/sqrt1p2;
%     for k=1:K2
%         x1=sqrt2*o1*XI(k) + u1;
%         x2=sqrt2*o2*(p*XI(k) + sqrt1p2*XJ(k)) + u2;
%         eval = -lambdad*WIWJ(k)*sqrt(epsn+(I1(m,n) - I2_cont(min(max(round((m+x2-1)*rfc2+1),1),MM),min(max(round((n+x1-1)*rfc2+1),1),NN)))^2);
%         dp = dp + eval*(p - p*XJ2(k) + 2*sqrt1p2*XIXJ(k));
%         du1 = du1 + eval*(XI(k) - pdivrt*XJ(k));
%         du2 = du2 + eval*XJ(k);
%         do1 = do1 + eval*(XI2(k) - 2*pdivrt*XIXJ(k) - 1);
%         do2 = do2 + eval*(XJ2(k) + 2*pdivrt*XIXJ(k) - 1);
%     end
%     du1 = du1*sqrt2/o1/pi;
%     du2 = du2*sqrt2/sqrt1p2/o2/pi;
%     do1 = do1/o1/pi;
%     do2 = do2/o2/pi;
%     dp =  dp/(1-p^2)/pi;
% end
    function [du1,du2,do1,do2,dp,nener] = node_grad_spectral(a,u1,u2,o1,o2,p,m,n)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0; nener = 0;
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
            nener = nener+fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        do1 = a*do1/pi/o1;
        do2 = a*do2/pi/o2;
        dp = a*dp/pi/pr;
        nener = nener/pi;
        % logP = node_pot(u1,u2,m,n);
    end
% function [du1,du2,do1,do2,dp] = edge_grad(u1,u2,o1,o2,p)
%     du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;
%     sqrt1p2=sqrt(1-p^2); pdivrt = p/sqrt1p2;
%     for k=1:K2
%         eval = -lambdas*WIWJ(k)*sqrt(epsn + (sqrt2*o1*XI(k)+u1 - sqrt2*o2*(p*XI(k)+sqrt1p2*XJ(k))-u2)^2);
%         dp = dp + eval*(p - p*XJ2(k) + 2*sqrt1p2*XIXJ(k));
%         du1 = du1 + eval*(XI(k) - pdivrt*XJ(k));
%         du2 = du2 + eval*XJ(k);
%         do1 = do1 + eval*(XI2(k) - 2*pdivrt*XIXJ(k) - 1);
%         do2 = do2 + eval*(XJ2(k) + 2*pdivrt*XIXJ(k) - 1);
%     end
%     du1 = du1*sqrt2/o1/pi;
%     du2 = du2*sqrt2/sqrt1p2/o2/pi;
%     do1 = do1/o1/pi;
%     do2 = do2/o2/pi;
%     dp = dp/(1-p^2)/pi;
% end
    function [du1,du2,do1,do2,dp,eener] = edge_grad_spectral(a,u1,u2,o1,o2,p)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0; eener=0;
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
            eener = eener+fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        do1 = a*do1/pi/o1;
        do2 = a*do2/pi/o2;
        dp = a*dp/pi/pr;
        eener = eener/pi;
        % logP = edge_pot(u1,u2);
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
% AEPE=gather(AEPE);
Energy=gather(Energy);
end