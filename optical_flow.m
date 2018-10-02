clear;
%read images from dataset and convert greyscale
testdata = {'RubberWhale','Dimetrodon','Hydrangea','Venus','Grove2','Grove3','Urban2','Urban3'};
scale = 1;
for ti=1:numel(testdata)
    name = testdata{ti};
    % time=datestr(now, 'yyyymmddHHMMSS');
    img1 = imresize(imread(['middlebury/',name,'/frame10.png']),scale);
    img2 = imresize(imread(['middlebury/',name,'/frame11.png']),scale);
    img_1 = double(rgb2gray(img1));
    img_2 = double(rgb2gray(img2));
    [gdt_img,trueFlow,options.minu, options.maxu, options.minv, options.maxv,unknownIdx] = ...
    flowToColor_mex(readFlowFile(['middlebury/',name,'/flow10.flo']));
    gdt_img = imresize(gdt_img,scale);

    options.K = 11;
    options.its =30000;
    options.epsn = 0.001^2;
    options.lambdas = 5;
    options.lambdad = 1;
    options.dir = ['../Results4_full/',name];
    mkdir(options.dir);
    % [mu, sigma, alpha, AEPE,Energy] = gqmap_gpuV2(options,img_1,img_2,trueFlow);
    [mu, sigma, alpha, AEPE, Energy,logP] = gqmap_gpu_mixture(options,img_1,img_2,trueFlow);
    save([options.dir,'/',name,'.mat'],'options','AEPE','trueFlow','unknownIdx','mu', 'sigma','alpha','Energy','logP');

end
