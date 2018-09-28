clear;
%read images from dataset and convert greyscale
testdata = {'RubberWhale','Hydrangea','Venus','Dimetrodon','Grove2','Grove3','Urban2','Urban3'};%,{'Hydrangea'};,%
scale = 1; preprocessed=false; mix = true;
for ti=1:numel(testdata)
    if ~preprocessed
        img1 = imresize(imread(['middlebury/',testdata{ti},'/frame10.png']),scale);
        img2 = imresize(imread(['middlebury/',testdata{ti},'/frame11.png']),scale);
        img1 = double(rgb2gray(img1));
        img2 = double(rgb2gray(img2));
    else
        load(['middlebury/preprocessed/',testdata{ti},'.mat'],'img1','img2');
    end
    [gdt_img,trueFlow,options.minu, options.maxu, options.minv, options.maxv,unknownIdx] = ...
        flowToColor(readFlowFile(['middlebury/',testdata{ti},'/flow10.flo']));
    gdt_img = imresize(gdt_img,scale);
    
    options.K = 17;
    options.its = 30000;
    options.epsn = 0.001^2;
    options.lambdas = 16;
    options.lambdad = 1;
    options.dir = ['../Results_mix_entropy/',testdata{ti}];
%     options.dir = ['../Results_mix_good/',testdata{ti}];
    mkdir(options.dir);
    if mix==true
        [mu, sigma, alpha, AEPE, Energy] = gqmap_gpuSuper_mix_entropy(options,img1,img2,trueFlow);
        save([options.dir,'/',testdata{ti},'.mat'],'options','AEPE','trueFlow','unknownIdx','mu','sigma','alpha','Energy');
    else
        [mu, sigma, rou, AEPE, Energy] = gqmap_gpuSuper(options,img1,img2,trueFlow);
        save([options.dir,'/',testdata{ti},'.mat'],'options','AEPE','trueFlow','unknownIdx','mu','sigma','Energy');
    end
    
end
% system('shutdown /s /t 60');