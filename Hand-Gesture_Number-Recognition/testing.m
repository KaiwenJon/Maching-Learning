clc
clearvars -except net
close all
load trained_net.mat
%% test image
rgbImage = imread('�z���Ϥ�');
prediction_digit = classify(net,rgbImage)
imshow(rgbImage);



