clc
clearvars -except net
close all
load trained_net.mat
%% Training data
digitDatasetPath = fullfile('您的訓練集檔案目錄');
imdsTrain = imageDatastore(digitDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
%% Validation data
Path = '您的驗證集檔案目錄';
imdsVal = imageDatastore(Path,'IncludeSubfolders',true,'LabelSource','foldernames');
%% Data Augmentation
imageSize = [256 256 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-3 3], ...
    'RandYTranslation',[-3 3],...
    'RandXReflection', 1,...
    'RandYReflection', 1)
augimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',imageAugmenter)

%% net
layers = [
    imageInputLayer([256 256 3])
    dropoutLayer(0.1)
    convolution2dLayer(10,40,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(16,40,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,40,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(10,40,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.2)
    convolution2dLayer(5,40,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,40,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,40,'Padding','same')
    batchNormalizationLayer
    reluLayer    
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.2)
    convolution2dLayer(5,40,'Padding','same')
    batchNormalizationLayer
    reluLayer   
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(60)
    reluLayer 
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];
%% options
miniBatchSize = 64;
opts = trainingOptions('adam', ...
    'MaxEpochs',150, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',0.001, ...
    'Verbose',true,'ValidationData',imdsVal);

%% 開train
%net = trainNetwork(augimds,net.Layers,opts); % 使用pretrained_model
net = trainNetwork(augimds,layers,opts); % 訓練新的model

save net





