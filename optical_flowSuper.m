clear;
%read images from dataset and convert greyscale
testdata ={'Hydrangea'};% {'rubberwhale','Urban2','Urban3','Grove3','Venus','Dimetrodon','Grove2','Hydrangea'};%,{'Hydrangea'};%
scale = 1;
for ti=1:numel(testdata)
    img1 = imresize(imread(['middlebury/',testdata{ti},'/frame10.png']),scale);
    img2 = imresize(imread(['middlebury/',testdata{ti},'/frame11.png']),scale);
    img1 = double(rgb2gray(img1));
    img2 = double(rgb2gray(img2));
%     load(['middlebury/preprocessed/',testdata{ti},'.mat'],'img1','img2');
    [gdt_img,trueFlow,options.minu, options.maxu, options.minv, options.maxv,unknownIdx] = ...
        flowToColor(readFlowFile(['middlebury/',testdata{ti},'/flow10.flo']));
    gdt_img = imresize(gdt_img,scale);

    options.K = 9;
    options.its =50000;
    options.epsn = 0.001^2;
    options.lambdas = 5;
    options.lambdad = 1;
    options.dir = datestr(now, 'yyyymmddHHMMSS');
    mkdir(options.dir);
    [mu, sigma, rou, AEPE,Energy] = gqmap_gpuSuper(options,img1,img2,trueFlow);
    save([options.dir,'/',testdata{ti},'.mat'],'options','AEPE','trueFlow','unknownIdx','mu','Energy');
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
system('shutdown /s')