
%%  random to save  new version

clc
clear

for aa=[1 5]
    
    if aa==1
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\close
        s=10;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\close'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\C'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\Validation\C'
    elseif aa==2
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\down
        s=12;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\down'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\D'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\Validation\D'
    elseif aa==3
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\lay
        s=9;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\lay'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\lay'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\lay'
    elseif aa==4
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\left
        s=11;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\left'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\left'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\left'
    elseif aa==5
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\open
        s=12;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\open'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\O'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\Validation\O'
    elseif aa==6
        cd D:\HQ\(2020)AI\NEW_0519\DATAbase\right
        s=9;
        str1='D:\HQ\(2020)AI\NEW_0519\DATAbase\right'
        str2='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\right'
        str3='D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT\right'
    end
    for type=1:2%:2
        for ve=1%:s
            cd(str1)
            %xyloObj = VideoReader('TTT.mp4');
            eval(['xyloObj = VideoReader(''a (',num2str(ve),').mp4'');'])
            nFrames = xyloObj.NumberOfFrames;
            
            if type==1
                cd(str2)
            elseif type==2
                cd(str3)
            end
            
            a = 1;
            b = nFrames;
            r = (b-a).*rand(1,100) + a;
            r=round(r);
            pp=length(r);
            for k=1:pp
                k=r(1,k);
                
                RGB = read(xyloObj,k);
                %RGB = imread(video);
                RGB=rgb2gray(RGB);
                aaa1=800;
                aaa2=600;
                RGB = imresize(RGB,[aaa1 aaa2]);
                se = strel('disk',10);
                %RGB = imclose(RGB,se);
                
                
                %BW=double(RGB);
                %BW= imbinarize(BW);
                RGB = medfilt2(RGB);
                BW = edge(RGB,'canny');
                BW=BW(200:500,:);
                BW= imresize(BW,[aaa1 aaa2]);
                %RGB= imbinarize(RGB);
                
                %BW = im2bw(BW,0.9);
                %BW=double(BW);
                
                %BW=(uint8(BW));
                eval(['filename=''s_',num2str(ve),'_no_',num2str(k),'.jpg'';'])
                imwrite(BW,filename);
                
            end
            
        end
        
        
    end
end
%% network validation ACC, training ACC.


clc
clear
close all

%F=combntns([1:11],2);
%list_ACC=zeros(1,length(F(:,1)));
%FF=1%:11%length(F(:,1))
aaa1=600;
aaa2=400;

digitDatasetPath = fullfile('D:\HQ\(2020)AI\NEW_0519\ana\test1\test4_0521\training\TTT');
%training set


% database for training
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = 200;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');


miniBatchSize = 5;
imdsTrain = shuffle(imdsTrain);
imdsTrain.ReadSize = miniBatchSize;



layers = [
    imageInputLayer([aaa1 aaa2 1])
    
    
    convolution2dLayer(10,10,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(2,2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    
    convolution2dLayer(3,3,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(4,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    
    convolution2dLayer(4,4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    
    convolution2dLayer(2,2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
     convolution2dLayer(2,2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
     convolution2dLayer(2,2,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
 
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
%% vedio check (testing), load vedio and netwrok part 1

clc
clear
aaa1=600;
aaa2=400;
cd D:\HQ\(2020)AI\NEW_0519\ana\test1
load SOBLE_A82
cd D:\HQ\(2020)AI\NEW_0519\DATAbase\close
%load ONEMAN_20200517.mat
xyloObj = VideoReader('a (1).mp4');
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
aaa=640;
TOT=uint8(zeros(aaa1,aaa2,nFrames));
score=zeros(nFrames,1);
% vedio check (testing), load vedio and netwrok part 2
gg=[];
for k = 1 : nFrames
    
    video = read(xyloObj,k);
    RGB = video;
    RGB=rgb2gray(RGB);
    RGB = imresize(RGB,[aaa1 aaa2]);
    se = strel('disk',10);
    %RGB = imclose(RGB,se);
    %RGB= imbinarize(RGB);
    RGB = medfilt2(RGB);
    RGB = edge(RGB,'canny');
    %BW=double(RGB);
    %BW= imbinarize(BW);
    %RGB = medfilt2(RGB);
    
    
    %BW = im2bw(BW,0.9);
    %BW=double(BW);
    BW=RGB;
    %BW=(uint8(BW));
    pred1 = classify(net,BW);
    TOT(:,:,k)=BW;
    score(k,1)=double(pred1);
    %pred1
    
    
    if pred1=='C'
        gg=[gg 1];
    elseif pred1=='D'
        gg=[gg 2];
    elseif pred1=='L'
        gg=[gg 3];
    elseif pred1=='O'
        gg=[gg 4];
    end
    %imwrite(im, ['original_frame',num2str(k),'.bmp'], 'bmp');%把im儲存成圖片，並且儲存的檔名根據序號改變
    
end

samples=length(gg);
double_ACC=length(find(gg==1))/samples;



%% trend testing
clc
clear
cd D:\HQ\(2020)AI\NEW_0519\DATAbase\close
xyloObj = VideoReader('a (1).mp4');   %% you can change it by webcam
nFrames = xyloObj.NumberOfFrames;
vidHeight = xyloObj.Height;
vidWidth = xyloObj.Width;
aaa1=800;
aaa2=600;
TOT=uint8(zeros(aaa1,aaa2,nFrames));
score=zeros(nFrames,1);


%%


%stastics
tot=round(nFrames/100)*100;
warning('off')
GG1 = factor(tot);
GM=max(GG1);
part=tot/max(GG1);
sts_open=zeros(1,part);
for pp=1:part;
    %video = read(xyloObj,(part*(pp-1)+1:1:part(pp)))
    TTTT=zeors(:,:,GG1);
    label=[(pp-1)*GM+1:1:pp*GM];
    
    gg=[];
    for classif=1:max(GG1)
        
        
        RGB = read(xyloObj,k);
        RGB=rgb2gray(RGB);
        aaa1=800;
        aaa2=600;
        RGB = imresize(RGB,[aaa1 aaa2]);
        se = strel('disk',10);
        %RGB = imclose(RGB,se);
        
        
        %BW=double(RGB);
        %BW= imbinarize(BW);
        RGB = medfilt2(RGB);
        BW = edge(RGB,'canny');
        BW=BW(200:500,:);
        BW= imresize(BW,[aaa1 aaa2]);
        %BW=double(RGB);
        %BW= imbinarize(BW);
        %RGB = medfilt2(RGB);
        
        pred1 = classify(net,BW);
        
        score(k,1)=double(pred1);
        %pred1
        
        
        if pred1=='C'
            gg=[gg 1];
        elseif pred1=='D'
            gg=[gg 2];
        elseif pred1=='L'
            gg=[gg 3];
        elseif pred1=='O'
            gg=[gg 4];
        end
        
        
    end
    
    
    samples=length(gg);
    sts_open(1,pp)=length(find(gg==4))/samples;
    
    
    
    
    
    
    
    
    
end


% trend



a = 0.1;
b = 0.6;
low = (b-a).*rand(1,50) + a;
a = 0.7;
b = 1;
high = (b-a).*rand(1,50) + a;

O_C_OC_CO=[];

LH=[low high];
HL=[high high];
LL=[low low];
HH=[high high];


R = corrcoef(sts_open,LL);

O_C_OC_CO=[corrcoef(sts_open,HH) corrcoef(sts_open,LL) corrcoef(sts_open,HL) corrcoef(sts_open,LH)];

ACA=[O_C_OC_CO(1,2) O_C_OC_CO(1,4) O_C_OC_CO(1,6) O_C_OC_CO(1,8)];
ACTION=find(ACA==max(ACA));

%% action