clc
clearvars -except net
close all
load trained_net.mat
%% test image
rgbImage = imread('您的圖片');
prediction_digit = classify(net,rgbImage)
imshow(rgbImage);



