clear;
%read images from dataset and convert greyscale
testdata = {'Urban3'};%{'rubberwhale','Urban2','Urban3','Grove3','Venus','Dimetrodon','Grove2','Hydrangea'};%,{'Hydrangea'};%
for ti=1:numel(testdata)
    time=datestr(now, 'yyyymmddHHMMSS');
    img1 = imread(['middlebury/',testdata{ti},'/frame10.png']);
    img2 = imread(['middlebury/',testdata{ti},'/frame11.png']);
    img_1 = double(rgb2gray(img1));
    img_2 = double(rgb2gray(img2));
    [gdt_img,trueFlow, minu, maxu, minv, maxv,unknownIdx] = ...
        flowToColor(readFlowFile(['middlebury/',testdata{ti},'/flow10.flo']));

    options.K = 11;
    options.its =3000;
    options.epsn = 0.001^2;
    options.lambdas = 5;
    options.lambdad = 1;
    options.dir = time;
%     mkdir(time);
%     [mu, sigma, rou, AEPE,Energy] = gqmap_gpuV2(options,img_1,img_2,trueFlow);
    scales = [1/8, 1/4, 1/2, 1];
    [M,N,~] = size(trueFlow);
    warp = imresize(zeros(M,N,2),scales(1)/2);%initialize warp incremental
    for rd = 1:numel(scales)
        scale = scales(rd);
        I1 = imresize(img_1, scale);
        I2 = imresize(img_2, scale);
        %Warp I1
        warp = imresize(warp,2) .* 2;
        [x, y] = meshgrid(1:size(I1,2), 1:size(I1,1));
        I1_w = interp2(I1, x-warp(:,:,1), y-warp(:,:,2));
        I1_w = fillmissing(fillmissing(I1_w,'nearest',1),'nearest',2);%remove NaN
        [flow, ~, ~, AEPE,Energy] = gqmap_ctf(options,I1_w,I2,trueFlow.*scale);
        warp = warp + flow;
        
    end
    
%     save([options.dir,'/',testdata{ti},'.mat'],'options','AEPE','trueFlow','unknownIdx','mu');
%     fig = figure;
%     fig.Position =  [1 41 1920 963];
%     subplot(2,2,1), imshow(img1);
%     title('Frame#1');
%     subplot(2,2,2), imshow(img2);
%     title('Frame#2');
%     subplot(2,2,3), imshow(gdt_img);
%     title('Ground Truth');
%     subplot(2,2,4), imshow(flowToColor(mu));
%     title('GQMAP');
    % title(sprintf('GQMAP(var=%.1f,gama=%.1f)',options.var,options.gama));
    % matflow = opticalFlow(mu(:,:,1,mu(:,:,2)); plot(matflow,'DecimationFactor',[10 10],'ScaleFactor',1);%quiver plot
end

%         C= imread('pout.tif'); % test image
%         [x, y] = meshgrid(1:size(C,2), 1:size(C,1));
%         % generate synthetic test data, for experimenting
%         vx = 0.1*y;   % an arbitrary flow field, in this case
%         vy = 0.1*x;   % representing shear
%         % compute the warped image - the subtractions are because we're specifying
%         % where in the original image each pixel in the new image comes from
%         D = interp2(double(C), x-vx, y-vy);
%         % display the result
%         imshow(D, []);