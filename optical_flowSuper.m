clear;
%read images from dataset and convert greyscale
testdata = {'Dimetrodon','Hydrangea','Venus','Grove2','Grove3','Urban2','Urban3'};%'RubberWhale',
scale = 1; preprocessed=false;
for ti=1:numel(testdata)
    name = testdata{ti};
    if ~preprocessed
        img1 = imresize(imread(['middlebury/',name,'/frame10.png']),scale);
        img2 = imresize(imread(['middlebury/',name,'/frame11.png']),scale);
        img1 = double(rgb2gray(img1));
        img2 = double(rgb2gray(img2));
    else
        load(['middlebury/preprocessed/',name,'.mat'],'img1','img2');
    end
    [gdt_img,options.trueFlow,options.minu,options.maxu,options.minv,options.maxv,options.unknownIdx] = ...
        flowToColor_mex(readFlowFile(['middlebury/',name,'/flow10.flo']));
    gdt_img = imresize(gdt_img,scale);
    
    options.K = 7;
    options.its = 30000;
    options.epsn = 0.001^2;
    options.lambdas = 16;
    options.lambdad = 1;
    options.L = 5;                  %number of components of mixture model
    options.temperature = 7;        %initial temperature weight
    options.drate=0.5;                %temperature changing rate
    
    if options.temperature ~=0
        options.dir = ['../Results5_mix_entropy/',name,'_',num2str(preprocessed)];
    else
        options.dir = ['../Results5_mix/',name,'_',num2str(preprocessed)];
    end
    mkdir(options.dir);
    [mu, sigma, alpha, AEPE, Energy,logP] = gqmap_gpuSuper_mix_entropy(options,img1,img2);
    save([options.dir,'/',name,'.mat'],'options','mu','sigma','alpha','AEPE','Energy','logP');
end
system('shutdown /s /t 60');