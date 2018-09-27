function [mu, sigma, rou] = gqmap_cpu(options,flow)
%GQMAP perform MAP inference using Gaussian Quadruatre with gradient ascent method
its = options.its; K = options.K; var = options.var; gama = options.gama; dta = options.dta;
[X, W] = GaussHermite_2(K);
[XI,XJ] = meshgrid(X); [WI,WJ] = meshgrid(W);
WIWJ = WI.*WJ;

[M,N,~] = size(flow);
mu = flow;%rand(M,N,2);% U layer and V layer
sigma = rand(M,N,2) + 2; % make sure it's a large initialization
rou = zeros(M,N,2,2); % vertical and horizontal edege (dim3), u layer and v layer(dim4)
it = 1;
tor = 1e-3;
dnode = zeros(M,N,2,2);% derivatives of pillars: mu_u, mu_v & sig_u sig_v
dedge = zeros(M,N,2,5,2);% derivatives of vertical-1 and horizontal-2 edges: u1, u2, o1, o2 and p
while 1
    parfor m=1:M-1
        for n=1:N-1
            tmp1 = zeros(2,2); 
            for l=1:2
                x = sqrt(2)*sigma(m,n,l)*X + mu(m,n,l);
                dval = W.*(flow(m,n,l) - x)/var;
                du = sum(dval) / sqrt(pi);
                do = sum(dval.*X) * sqrt(2/pi);
                tmp1(:,l) = [du do];
            end
            dnode(m,n,:,:) = tmp1;
            for j=1:2
                m2 = m + (j==1);
                n2 = n + (j==2);
                tmp2 = zeros(5,2);
                for l=1:2
                    p = rou(m,n,j,l);
                    o1 = sigma(m,n,l); o2 = sigma(m2,n2,l); %#ok<PFBNS>
                    u1 = mu(m,n,l); u2 = mu(m2,n2,l); %#ok<PFBNS>
                    q = sqrt(1+p); r = sqrt(1-p);
                    s = (q+r)/2; t = (q-r)/2;
                    ds = (1/q - 1/r)/4; dt = (1/q + 1/r)/4;
                    ZI = s*XI+t*XJ; ZJ = t*XI+s*XJ;
                    x1 = sqrt(2)*o1*ZI + u1;
                    x2 = sqrt(2)*o2*ZJ + u2;
                    diff = x2-x1;
                    diff(abs(diff)>dta) = 0;
                    df1 = WIWJ.*diff/gama;
                    df2 = -df1;
                    du1 = 1/pi*sum(sum(df1));
                    du2 = 1/pi*sum(sum(df2));
                    do1 = 1/pi*sqrt(2)*sum(sum(df1.*ZI));
                    do2 = 1/pi*sqrt(2)*sum(sum(df2.*ZJ));
                    dp =  1/pi*sqrt(2)*sum(sum(o1*df1.*(ds*XI+dt*XJ) + o2*df2.*(dt*XI+ds*XJ)));
                    tmp2(:,l) = [du1 du2 do1 do2 dp];
                end
                dedge(m,n,j,:,:) = tmp2;
            end
        end
    end
    % -----.sum up (update)
    dmu = squeeze(dnode(:,:,1,:)) + squeeze(sum(dedge(:,:,:,1,:),3)) + squeeze(cat(1,dedge(2:M,1:N,1,2,:),zeros(1,N,1,1,2)) + cat(2,dedge(1:M,2:N,2,2,:),zeros(M,1,1,1,2)));
    dsigma = squeeze(dnode(:,:,2,:)) + squeeze(sum(dedge(:,:,:,3,:),3)) + squeeze(cat(1,dedge(2:M,1:N,1,4,:),zeros(1,N,1,1,2)) + cat(2,dedge(1:M,2:N,2,4,:),zeros(M,1,1,1,2)));
    drou = squeeze(dedge(:,:,:,5,:));
    step = 0.1/(1+it/1000);
    mu = mu + dmu * step;
    sigma = abs(sigma + dsigma * step);
    rou = max(min(rou + drou * step, 0.97), -0.97);
%     if it>100,sigma = sigma*0.999;end% quench; Tempurature gradually down.
    fprintf('#[%3d], \x0394(mu) = %d, \x0394(sigma) = %d, \x0394(rou) = %d\n',...
        it, max(abs(dmu(:))), max(abs(dsigma(:))), max(abs(drou(:))));
    it = it + 1;
    if it > its || it>100 && max(abs(dmu(:))) < tor, break; end
end

end