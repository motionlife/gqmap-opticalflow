function [mu, sigma, rou] = gqmap_cpuV3(options,I1,I2,GRDT)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its; K = options.K; sg = options.sg; lambdad = options.lambdad; lambdas=options.lambdas;
[X, W] = GaussHermite_2(K);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);sqrt2=sqrt(2);
XIXJ = XI.*XJ;%XI2aXJ2 = XI.^2 + XJ.^2;XI2mXJ2 = XI.^2 - XJ.^2;
WIWJ = WI.*WJ;
XI2 = 2*XI.^2;
XJ2 = 2*XJ.^2;
[M,N] = size(I1); rg=1;dim = 2*rg+1; dim2=dim^2;
INDI= repmat(1:M,K^2,1,3); INDJ = repmat(1:N,K^2*dim,1);
mu = rand(M,N,2)-0.5;
pn = zeros(M,N); dpn = zeros(M,N);
sigma = rand(M,N,2) + 3; % make sure it's a large initialization
rou = zeros(M,N,2,2); % u,v share share rou crossing the edge??
it = 1;
tor = 1e-3;
dnode = zeros(M,N,2,2);% derivatives of nodes: u & o
dedge = zeros(M,N,2,5,2);% derivatives of x-1 and y-2 edges: u1, u2, o1, o2 and p
while 1
    parfor m=rg+1:M-rg
        for n=rg+1:N-rg % pillar(z) edge
            p = pn(m,n);sqrt1p2=sqrt(1-p^2);
            o1=sigma(m,n,1);o2 = sigma(m,n,2);  %#ok<*PFBNS>
            u1=mu(m,n,1); u2=mu(m,n,2);
            wdi = m-rg:m+rg;
            wdj = n-rg:n+rg;
            x1 = sqrt2*o1*XI+u1;
            x2 = sqrt2*o2*(p*XI+sqrt1p2*XJ)+u2;
            xq = min(max(reshape(INDJ(:,wdj),K,K,dim2) + x1,1),N);
            yq = min(max(reshape(INDI(:,wdi,:),K,K,dim2) + x2,1),M);
            eval = -lambdad*WIWJ.*mean(sqrt(sg+(reshape(I1(wdi,wdj),1,1,[])-interp2(I2,xq,yq,'cubic')).^2),3);
            du1 = sum(sum(eval.*(XI - p/sqrt1p2*XJ)))*sqrt(2)/o1/pi;
            du2 = sum(sum(eval.*XJ))*sqrt2/sqrt1p2/o2/pi;
            do1 = sum(sum(eval.*(XI2 - 2*p/sqrt1p2*XIXJ - 1)))/o1/pi;
            do2 = sum(sum(eval.*(XJ2 + 2*p*XIXJ/sqrt1p2 - 1)))/o2/pi;
            dpn(m,n) = sum(sum(eval.*(p - p*XJ2 + 2*sqrt1p2*XIXJ)))/(1-p^2)/pi;
            dnode(m,n,:,:) = [du1 du2;do1 do2];
            for j=1:2% x edge or y edge
                m2 = m + (j==1);
                n2 = n + (j==2);
                for t=1:2 % u edge or v edge
                    p = rou(m,n,j,t);sqrt1p2=sqrt(1-p^2);
                    o1 = sigma(m,n,t); u1=mu(m,n,t);
                    o2 = sigma(m2,n2,t); u2=mu(m2,n2,t);
                    eval = -lambdas*WIWJ.*sqrt(sg + (sqrt2*o1*XI+u1 - sqrt2*o2*(p*XI+sqrt1p2*XJ)-u2).^2);
                    du1 = sum(sum(eval.*(XI - p/sqrt1p2*XJ)))*sqrt2/o1/pi;
                    du2 = sum(sum(eval.*XJ))*sqrt2/sqrt1p2/o2/pi;
                    do1 = sum(sum(eval.*(XI2 - 2*p/sqrt1p2*XIXJ - 1)))/o1/pi;
                    do2 = sum(sum(eval.*(XJ2 + 2*p/sqrt1p2*XIXJ - 1)))/o2/pi;
                    dp = sum(sum(eval.*(p - p*XJ2 + 2*sqrt1p2*XIXJ)))/(1-p^2)/pi;
                    dedge(m,n,j,:,t) = [du1 du2 do1 do2 dp];
                end
            end
        end
    end
    dmu = squeeze(dnode(:,:,1,:)) + squeeze(sum(dedge(:,:,:,1,:),3)) + ...
        squeeze(circshift(dedge(:,:,1,2,:),1,1)) + squeeze(circshift(dedge(:,:,2,2,:),1,2));
    dsigma = squeeze(dnode(:,:,2,:)) + squeeze(sum(dedge(:,:,:,3,:),3)) + ...
        squeeze(circshift(dedge(:,:,1,4,:),1,1)) + squeeze(circshift(dedge(:,:,2,4,:),1,2));
    drou = squeeze(dedge(:,:,:,5,:));
    step = 0.1/(1+it/1000);

    mu(:,:,1) = min(max(mu(:,:,1) + dmu(:,:,1) * step,options.minu),options.maxu);
    mu(:,:,2) = min(max(mu(:,:,2) + dmu(:,:,2) * step,options.minv),options.maxv);
    sigma(:,:,1) = min(max(sigma(:,:,1) + dsigma(:,:,1) * step,1e-3),30);
    sigma(:,:,2) = min(max(sigma(:,:,2) + dsigma(:,:,2) * step,1e-3),30);
    rou = max(min(rou + drou * step, 0.98), -0.98);
    pn = max(min(pn + dpn * step, 0.98), -0.98);
    %if it>50,sigma = sigma*0.9;end% quench; Tempurature gradually down.
    flc=flowToColor(mu);
    imshow(flc);
    if mod(it,1)==0, imwrite(flc,[options.timestamp,'/',num2str(it),'.png']);end
    fprintf('#[%3d], \x0394(mu) = %d, \x0394(sigma) = %d, \x0394(rou) = %d, AEPE=%d\n',...
        it, mean(abs(dmu(:))),mean(abs(dsigma(:))), mean(abs(drou(:))), mean(mean(sqrt(sum((mu-GRDT).^2,3)))));
    it = it + 1;
    if it > its || it>100 && mean(abs(dmu(:))) < tor, break; end
end
end