clear;
%read images from dataset and convert greyscale
testdata = {'Urban3'};%{'Venus','Dimetrodon','Urban2','Urban3'};%{'Grove3','Grove2','rubberwhale','Hydrangea'};%,{'Hydrangea'};%
for ti=1:numel(testdata)
    s = linspace(0.300001,1.0,12);
    bests = 1;
    bestaepe = Inf;
    for i = 1:numel(s)
        img1 = imread(['middlebury/',testdata{ti},'/frame10.png']);
        img2 = imread(['middlebury/',testdata{ti},'/frame11.png']);
        img_1 = double(rgb2gray(img1));
        img_2 = double(rgb2gray(img2));
        [gdt_img,trueFlow,options.minu, options.maxu, options.minv, options.maxv,unknownIdx] = ...
            flowToColor(readFlowFile(['middlebury/',testdata{ti},'/flow10.flo']));

        options.K = 11;
        options.its = 20000;
        options.sg = 0.007^2;
        options.lambdas = s(i);%0.55;(Hydrangea)%;0.45(rubberwhale)
        options.lambdad = 0.1;
        options.dir = [testdata{ti},'/',num2str(s(i),7)];
        mkdir(options.dir);
        [mu, sigma, rou, AEPE] = gqmap_gpuV2(options,img_1,img_2,trueFlow);
        save([options.dir,'/','DATA.mat'],'options','AEPE','trueFlow','unknownIdx','mu');
        aepe = min(AEPE);
        if aepe<bestaepe
            bests = i;
            bestaepe = aepe;
        end
    end
    fileID = fopen([testdata{ti},'/LOG.txt'],'w');
    fprintf(fileID,'Best lambda s = %d\n',s(bests));
    fclose(fileID);
end
