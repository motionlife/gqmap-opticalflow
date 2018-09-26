clear;
%read images from dataset and convert greyscale
testdata = {'Urban3','Grove3','Urban2','Venus','Dimetrodon','rubberwhale','Grove2','Hydrangea'};%'rubberwhale',{'Hydrangea'};%
scale = 1;
for ti=1:numel(testdata)
    time=datestr(now, 'yyyymmddHHMMSS');
    img1 = imresize(imread(['middlebury/',testdata{ti},'/frame10.png']),scale);
    img2 = imresize(imread(['middlebury/',testdata{ti},'/frame11.png']),scale);
    [gdt_img,gdt_flow,options.minu, options.maxu, options.minv, options.maxv,idxunknow] = ...
        flowToColor(readFlowFile(['middlebury/',testdata{ti},'/flow10.flo']));
    gdt_img = imresize(gdt_img,scale);
    [M,N,~] = size(img1);%m is the image height, n is the width
    U = floor(6*scale)+1;V = U; ft = 3;% windows=[2*wh+1, 2*ww+1]
    filter = Gaussian_filter(2*ft+1,1.7);%gaussan convolution with specified varance
    % filter = ones(2*ft+1);
    % Try median filter or mean filter other Spatial Filters
    %% PRE-PROCCESSING
    %#1. Matching cost computation and convolution
    img_1 = double(rgb2gray(img1));
    img1_ext = zeros(M+2*U,N+2*V);
    img1_ext(U+1:end-U,V+1:end-V) = img_1;
    img_2 = double(rgb2gray(img2));
    costs = zeros(M,N,2*U+1,2*V+1);%based on a slide size of Offset;
    for u = 1 : 2*U+1
        for v = 1 : 2*V+1
            costs(:,:,u,v) = conv2(abs(img_2 - img1_ext(u:M+u-1,v:N+v-1)), filter,'same');%TODO: find a robust cost function; quadratic formulation is not robust to outliers
        end
    end
    [~,idx] = min(reshape(costs,M,N,[]),[],3);
    [fu,fv] = ind2sub([u v],idx);
    umt = U+1-fu;
    vmt= V+1-fv;
    
    options.K = 17;
    options.its = 5000;
    options.sg = 0.0001;
    options.lambdas = 1.7;
    options.lambdad = 0.3;
    options.timestamp = time;
%     mkdir(time);
    [mu, sigma, rou, aepe,bestitr] = gqmap_gpuV2(options,img_1,img_2,gdt_flow);
    writeFlowFile(mu,[time,'/',testdata{ti},'.flo']);
    save([time,'/',testdata{ti},'.mat'],'options','aepe','bestitr','idxunknow');
    fig = figure;
    fig.Position =  [1 41 1920 963];
    subplot(2,2,1), imshow(img1);
    title('Frame#1');
    subplot(2,2,2), imshow(img2);
    title('Frame#1');
    subplot(2,2,3), imshow(gdt_img);
    title('Ground Truth');
    %subplot(2,2,4), imshow(flowToColor(cat(3,vmt,umt)));
    subplot(2,2,4), imshow(flowToColor(mu));
    title('GQMAP');
    % title(sprintf('GQMAP(var=%.1f,gama=%.1f)',options.var,options.gama));
    % matflow = opticalFlow(U+1-fu,V+1-fv); plot(matflow,'DecimationFactor',[10 10],'ScaleFactor',1);%quiver plot
end
