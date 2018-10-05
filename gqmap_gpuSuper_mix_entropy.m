function [mu, sigma, alpha, AEPE,Energy,logP] = gqmap_gpuSuper_mix_entropy(options,I1,I2)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
tflow = options.trueFlow;  unidx=repmat(options.unknownIdx,1,1,2);
its = options.its; K = options.K; L=options.L; T=options.temperature; drate = options.drate;
epsn = options.epsn; lambdad = options.lambdad; lambdas=options.lambdas;
minu=options.minu;  maxu=options.maxu;  minv=options.minv;  maxv=options.maxv;
I1=gpuArray(I1); K2 = K^2; sqrt2=sqrt(2); const1=1+log(2*pi); corr_tor=1-1e-5;
[X, W] = GaussHermite_2(K); X = gpuArray(X);  W = gpuArray(W);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ; XIXJ = XI.*XJ; XI2aXJ2 = XI.^2 + XJ.^2;XI2mXJ2 = XI.^2 - XJ.^2;
[Mo,No] = size(I1); M = Mo/4; N = No/4; border=1; M_= (1+border):(M-border); N_ = (1+border):(N-border);
VV = gpuArray(getVV(I2)); M2 = Mo+2;
% rfc=6;rfc2=2^rfc;	I2_cont = interp2(I2,rfc,'cubic');[MM, NN] = size(I2_cont);
[ns,ms] = meshgrid(gpuArray(1:N),gpuArray(1:M));
best_aepe=Inf; AEPE=NaN(its,1,'gpuArray'); logP = NaN(its,1,'gpuArray'); Energy = zeros(its,1,'gpuArray');mark=1;
%initialize gpu data
w = rand(1,1,L,'gpuArray'); alpha = exp(w)./sum(exp(w));
muu = minu+rand(M,N,L,'gpuArray')*(maxu-minu);
muv = minv+rand(M,N,L,'gpuArray')*(maxv-minv);
sigmau = rand(M,N,L,'gpuArray') + (maxu-minu);%better if it's a large initialization
sigmav = rand(M,N,L,'gpuArray') + (maxv-minv);
pn = zeros(M,N,L,'gpuArray');
rou = zeros(M,N,L,2,2,'gpuArray');
it = 1; tor = 1e-4;
while 1
    step = 0.001/(1+it/4000);
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
    sigmau(M_,N_,:) = min(max(sigmau(M_,N_,:) + dsigmau(M_,N_,:) * step,0.01),25);
    sigmav(M_,N_,:) = min(max(sigmav(M_,N_,:) + dsigmav(M_,N_,:) * step,0.01),25);
    rou(M_,N_,:,:,:) = min(max(rou(M_,N_,:,:,:) + drou(M_,N_,:,:,:) * step, -corr_tor), corr_tor);
    pn(M_,N_,:) = min(max(pn(M_,N_,:) + dpn(M_,N_,:) * step, -corr_tor), corr_tor);
    
    Energy(it) = sum(sum(sum(nEnergy(M_,N_,:)))) + sum(sum(sum(sum(sum(eEnergy(M_,N_,:,:,:))))));
%     if it>200, alpha = projsplx(alpha + dalpha * step * 1E-6);end
    if it>500 && L~=1, alpha = updateAlpha();end
    
    if mod(it,300)==0 || it==1
        [alf,mu_u,sig_u,mu_v,sig_v] = gather(alpha,muu,sigmau,muv,sigmav);
        if L==1
            map = cat(3,mu_u,mu_v);
        else
            map = get_map_mex(alf, mu_u, sig_u, mu_v, sig_v);
        end
        flow = repelem(map,4,4);
        flc = flowToColor_mex(flow(5:end-4,5:end-4,:));
%         imshow(flc);
        imwrite(flc,[options.dir,'/',num2str(it),'.png']);
        flow(unidx)=0;
        aepe = mean(mean(sqrt(sum((tflow(5:end-4,5:end-4,:) - flow(5:end-4,5:end-4,:)).^2,3))));AEPE(it)=aepe;
        if aepe < best_aepe, best_aepe = aepe;end
        logP(it) = profile_logP(gpuArray(map));
        mark = it;
    end
    ptdmu=abs(dmuu(M_,N_,:)); ptdsigma=abs(dsigmau(M_,N_,:));
    ptdmu = mean(ptdmu(:)); ptdsigma=mean(ptdsigma(:));
    fprintf('[%3d], \x0394(mu) = %e, \x0394(sigma) = %e, Energy = %e, AEPE=%e,logP=%e \n', ...
        it, ptdmu, ptdsigma, Energy(it), best_aepe, logP(mark));
    if mod(it,500)==0,T = max(T*drate,0.001);end
    it = it + 1;
    if it > its || ptdmu < tor, break; end
end

    function alf=updateAlpha()
%         smw = sum(w.^2);
%         w = w + dalpha.*(smw-w.^2).*w/smw^2 * step*1E-5;
%         alf = w.^2/sum(w.^2);
        dw = alpha.*(dalpha - sum(dalpha.*alpha));
        w = min(max(w + dw*step*1E-7,-300),300);
        alf = exp(w)./sum(exp(w));
        % exp((w - max(w)) - log(sum(exp(w - max(w)))))
    end

    function [da,du1,du2,do1,do2,dp,Ei] = node_grad_spectral(a,u1,u2,o1,o2,p,m,n)
        du1 = 0; du2 = 0; do1 = 0; do2 = 0; dp = 0;Ei=0;
        %if (m<=rg||m>M-rg||n<=rg||n>N-rg),return;end
        s = (sqrt(1+p)+sqrt(1-p))/2;
        t = (sqrt(1+p)-sqrt(1-p))/2;
        pr = 1 - p^2; sqrtpr = sqrt(pr);
        o1pr = sqrt2/(o1*pr); o2pr = sqrt2/(o2*pr);
        bottom=4*m;top=bottom-3;
        right=4*n;left=right-3;
        for k=1:K2
            zi = s*XI(k)+t*XJ(k); zj = t*XI(k)+s*XJ(k);
            x1=sqrt2*o1*zi+u1; x2=sqrt2*o2*zj+u2;
            super=0;
            for i=top:bottom
                for j=left:right
                    super=super+node_pot(x1,x2,i,j);
                end
            end
            fval = WIWJ(k)*super;
            dp  = dp + fval*(p - p*XI2aXJ2(k) + 2*XIXJ(k));
            du1 = du1 + fval*(zi - p*zj);
            du2 = du2 + fval*(zj - p*zi);
            do1 = do1 + fval*(XI2aXJ2(k) - 1 + XI2mXJ2(k)/sqrtpr);
            do2 = do2 + fval*(XI2aXJ2(k) - 1 - XI2mXJ2(k)/sqrtpr);
            Ei = Ei + fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        %Considering entropy with Temperature T--------------------------------
        %Node entropy part of each mixture component: a*(1-d)*H(b)) [Note:this is not the best approximation but simplest]
        da = Ei/pi - 3*T*(const1+log(sqrtpr*o1*o2));
        do1 = a*(do1/pi - 3*T)/o1;
        do2 = a*(do2/pi - 3*T)/o2;
        dp = a*(dp/pi + 3*T*p)/pr;
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
            dp  = dp + fval*(p - p*XI2aXJ2(k) + 2*XIXJ(k));
            du1 = du1 + fval*(zi - p*zj);
            du2 = du2 + fval*(zj - p*zi);
            do1 = do1 + fval*(XI2aXJ2(k) - 1 + XI2mXJ2(k)/sqrtpr);
            do2 = do2 + fval*(XI2aXJ2(k) - 1 - XI2mXJ2(k)/sqrtpr);
            Ei = Ei + fval;
        end
        du1 = a*du1*o1pr/pi;
        du2 = a*du2*o2pr/pi;
        %Considering entropy with Temperature T--------------------------------
        %Node entropy part of each mixture component: a*H(b)) [Note:this is not the best approximation but simplest]
        da = Ei/pi + T*(const1+log(sqrtpr*o1*o2));
        do1 = a*(do1/pi + T)/o1;
        do2 = a*(do2/pi + T)/o2;
        dp = a*(dp/pi - T*p)/pr;
        Ei = a*da;
    end

    function lp = profile_logP(uv)
        us = uv(:,:,1);
        vs = uv(:,:,2);
        np = arrayfun(@node_lp,us,vs,ms,ns);
        ep = arrayfun(@edge_pot,cat(4,uv,uv),cat(4,circshift(uv,-1),circshift(uv,-1,2)));
        lp = sum(sum(np(M_,N_))) + sum(sum(sum(sum(ep(M_,N_,:,:)))));
    end

    function lp = node_lp(x1,x2,m,n)
        lp=0;
        bottom=4*m;top=bottom-3;
        right=4*n;left=right-3;
        for i=top:bottom
            for j=left:right
                lp=lp+node_pot(x1,x2,i,j);
            end
        end
    end

    function npt = node_pot(x1,x2,i,j)
        Xq = min(max(j + x1,1),No);
        Yq = min(max(i + x2,1),Mo);
        %Bicubic interpolation kernel version (from interp2)
        if Xq <= 1.0, ix = 1;elseif Xq <= No-1, ix = floor(Xq);else, ix = No-1;end
        if Yq <= 1.0, iy = 1;elseif Yq <= Mo-1, iy = floor(Yq);else, iy = Mo-1;end
        so = Xq - ix;
        to = Yq - iy;
        ss = ((2.0 - so) * so - 1.0) * so;
        iy1 = iy + M2 * (ix - 1);
        iy2 = iy1 + M2;
        iy3 = iy2 + M2;
        iy4 = iy3 + M2;
        Vq = ((VV(iy1) * ss * (((2.0 - to) * to - 1.0) * to) + VV(iy1+1) * ss * ((3.0 * to - 5.0) * to * to + 2.0)) + VV(iy1+2) * ss * (((4.0 - 3.0 * to) * to + 1.0) * to)) + VV(iy1+3) * ss * ((to - 1.0) * to * to);
        ss = (3.0 * so - 5.0) * so * so + 2.0;
        Vq = Vq + VV(iy2) * ss * (((2.0 - to) * to - 1.0) * to) + VV(iy2+1) * ss * ((3.0 * to - 5.0) * to * to + 2.0) + VV(iy2+2) * ss * (((4.0 - 3.0 * to) * to + 1.0) * to) + VV(iy2+3) * ss * ((to - 1.0) * to * to);
        ss = ((4.0 - 3.0 * so) * so + 1.0) * so;
        Vq = Vq + VV(iy3) * ss * (((2.0 - to) * to - 1.0) * to) + VV(iy3+1) * ss * ((3.0 * to - 5.0) * to * to + 2.0) + VV(iy3+2) * ss * (((4.0 - 3.0 * to) * to + 1.0) * to) + VV(iy3+3) * ss * ((to - 1.0) * to * to);
        ss = (so - 1.0) * so * so;
        Vq = Vq + VV(iy4) * ss * (((2.0 - to) * to - 1.0) * to) + VV(iy4+1) * ss * ((3.0 * to - 5.0) * to * to + 2.0) + VV(iy4+2) * ss * (((4.0 - 3.0 * to) * to + 1.0) * to) + VV(iy4+3) * ss * ((to - 1.0) * to * to);
        Vq = Vq/4;
        
        npt = -lambdad*sqrt(epsn + (I1(i,j) - Vq)^2);
    end

    function ept = edge_pot(x1,x2)
        ept = -lambdas*sqrt(epsn+(x1-x2)^2);
    end
mu=gather(cat(4,muu,muv));
sigma=gather(cat(4,sigmau,sigmav));
alpha=gather(alpha);
AEPE=gather(AEPE);
Energy=gather(Energy);
logP=gather(logP);
end
function VV = getVV(V)
    [M,N]=size(V);%M=388;N=584
    M2 = M+2;%M2=390;
    N2 = N+2;%N2=586;
    M2N2=M2*N2;
    VV=zeros(M2,N2);
    VV(2:end-1,2:end-1)=V;
    for i=1:N2
        ix = M2*(i-1)+1;
        iy = ix+M2-1;
        VV(ix) = (3.0*VV(ix+1) - 3.0*VV(ix+2)) + VV(ix+3);
        VV(iy) = (3.0*VV(iy-1) - 3.0*VV(iy-2)) + VV(iy-3);
    end
    for i=1:M2
        VV(i) = (3.0*VV(M2+i) - 3.0*VV(M2*2+i)) + VV(M2*3+i);
        VV(M2N2-M2+i) = (3.0*VV(M2N2-M2*2+i) - 3.0 * VV(M2N2-M2*3+i)) + VV(M2N2-M2*4+i);
    end
end
% function [minu,maxu] = getRange(u0,u1,pct)
%     if u0<0,minu=u0*(1+pct);else,minu=u0*(1-pct);end
%     if u1>0,maxu=u1*(1+pct);else,maxu=u1*(1-pct);end
% end
