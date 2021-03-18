%% Main
clc
clearvars -except net song class_pitchlong
%%
% load D:\NTU\大三下\科學計算\AIcup\class_pitchlong
% load D:\NTU\大三下\科學計算\AIcup\song_data_with_everything_long.mat % song struct 導入需要耗時間，故可導入一次就好，反正clear不會clear到它
% class_pitchlong = class_pitchlong.';
% testsong = [testsong1 testsong2];
train_song = song;
%%
global Feature_name;
Feature_name = fieldnames(train_song);
global future_num;
future_num = 20 ;%取偶數
%% 加入訓練資料
Xtrain = [];
state = [];
for train_num = 1:450%(class_pitchlong{1}(1:10)
    [Xtrain_temp,state_temp] = Data_Constructor(train_song,train_num,0);
    Xtrain_temp = Xtrain_temp.';
    state_temp = state_temp.';
    % 添加新資料
    Xtrain = [Xtrain;Xtrain_temp];
    state = [state;state_temp];
end
Ytrain = categorical(cellstr(state));
%% 加入val
Xval = [];
val_state = [];
for val_num = 451:500%class_pitchlong{2}(2:2)
    [Xval_temp,val_state_temp] = Data_Constructor(train_song,val_num,0);
    Xval_temp = Xval_temp.';
    val_state_temp = val_state_temp.';
    % 添加新資料
    Xval = [Xval;Xval_temp];
    val_state = [val_state;val_state_temp];
end
Yval = categorical(cellstr(val_state));
%% 進行訓練
numClasses = size(Xtrain{1,1},1);
inputSize = numClasses;
Layers = [ ...
    sequenceInputLayer(inputSize,'Normalization','rescale-zero-one')%'zerocenter' 'zscore' rescale-zero-one
    bilstmLayer(100,'OutputMode','sequence')% 100 64 120
    dropoutLayer(0.3)
    bilstmLayer(80,'OutputMode','sequence')% 100 64 120
    bilstmLayer(70,'OutputMode','sequence')% 100 64 120
    bilstmLayer(64,'OutputMode','last')
    dropoutLayer(0.3)
    fullyConnectedLayer(120)
    reluLayer
    fullyConnectedLayer(90)
    reluLayer
    fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer]  
maxEpochs = 200;
miniBatchSize = 150;
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'L2Regularization',0.000,...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'VerboseFrequency',60,...
    'InitialLearnRate',0.0001, ...
    'Plots','training-progress','CheckpointPath','D:\NTU\大三下\科學計算\AIcup\0614_train_new_model\net_0615new_idea',...
    'ValidationData',{Xval,Yval}, ...
    'ValidationFrequency',60)%0,'ValidationPatience',20)
%% 是否load net
%load net0527_choosetime_dropout02.mat
net = trainNetwork(Xtrain,Ytrain,net.Layers,options);
%net = trainNetwork(Xtrain,Ytrain,Layers,options);

%% 可選擇一首歌畫圖測試
% 挑一首來測試
clear state_pred
Bingo = 0;
%song = testsong;
for test_num = 470%class_pitchlong{3}(1) %class_groundtruth{1}(2)
    [Xtest,state_truth] = Data_Constructor(song,test_num,1);
    %測試有東西的地方
    useful_data_ind = find(song(test_num).vocal_pitch ~= 0 | song(test_num).energy >= 0.00004)
    useless_data_ind = find(~(song(test_num).vocal_pitch ~= 0 | song(test_num).energy >= 0.00004))
    
    for i = 1 : length(useful_data_ind)
        k =useful_data_ind(i);
        if k-future_num/2 <= 0
            state_pred(k) = 'No_predict';
            continue;
        end
        state_pred(k) = classify(net,Xtest{k});
        if strcmp(string(state_pred(k)),state_truth{k})
            Bingo = Bingo + 1;
            disp(k)
        end
    end
end
%% 畫圖 My prediction
% 若沒predict，把結果改成nothing
state_pred(useless_data_ind) = 'No_predict';
state_pred = string(state_pred);
accuracy = sum(strcmp(state_pred(useful_data_ind).',song(test_num).onoff_truth(useful_data_ind)))/length(state_pred(useful_data_ind));
figure;title('My prediction')
plot(song(test_num).time,song(test_num).Dense_truth,'o','LineWidth',1,'Color','black')
hold on
plot(song(test_num).time,song(test_num).vocal_pitch,'o','LineWidth',1,'Color','red')
% %on off truth
% for p = 1 : length(song(test_num).time)
%     x = [song(test_num).time(p)  song(test_num).time(p)];
%     y = [0 80];
%     if strcmp(song(test_num).onoff_truth(p),'on')
%         plot(x,y,'Color','y','LineWidth',5)%黃
%     elseif strcmp(song(test_num).onoff_truth(p),'off')
%         plot(x,y,'Color','c','LineWidth',4)%青
%     elseif strcmp(song(test_num).onoff_truth(p),'both')
%         plot(x,y,'Color','m','LineWidth',6)%粉
% %         elseif strcmp(state_pred(p),'Nothing')
% %             plot(x,y,'Color','black','LineWidth',0.5)
%     end
%     hold on
% end

for p = 1 : length(state_pred)
    x = [song(test_num).time(p)  song(test_num).time(p)];
    y = [0 80];
    if strcmp(state_pred(p),'on')
        plot(x,y,'Color',[0.4660 0.6740 0.1880],'LineWidth',2)%綠
    elseif strcmp(state_pred(p),'off')
        plot(x,y,'Color',[0.6350 0.0780 0.1840],'LineWidth',2)%紅
    elseif strcmp(state_pred(p),'No_predict')
        plot(x,y,'Color','black','LineWidth',0.5)
    elseif strcmp(state_pred(p),'both')
        plot(x,y,'Color',[0.9290 0.6940 0.1250],'LineWidth',3)%黃
%         elseif strcmp(state_pred(p),'Nothing')
%             plot(x,y,'Color','black','LineWidth',0.5)
    end
    hold on
end

%% data constructor
function [X_temp,state_temp] = Data_Constructor(song,num,isTesting)
    global future_num;
    global Feature_name;
    answer = [];
    X_temp = [];
    state_temp = [];
    j = 0;
    norm_energy = normalize(song(num).energy);
    norm_energy_entropy = normalize(song(num).energy_entropy);  
    norm_spec_cen = normalize(song(num).spectral_centroid);
    norm_spec_flux = normalize(song(num).spectral_flux);
    norm_vocal_pitch = normalize(song(num).vocal_pitch);
    norm_zcr = normalize(song(num).zcr);
    norm_spectral_rolloff = normalize(song(num).spectral_rolloff);
    norm_spectral_entropy = normalize(song(num).spectral_entropy);
    sum_chroma = 0;
    for i = 9 : 20
        sum_chroma = sum_chroma + song(num).(Feature_name{i});
    end
    mean_chroma = sum_chroma/12;
    % 用vocal pitch不為0的一系列資料去做 
    % 或energy > 0.0001也要做！
    useful_data = find(song(num).vocal_pitch ~= 0 | song(num).energy >= 0.00004);
    useless_data = find(~(song(num).vocal_pitch ~= 0 | song(num).energy >= 0.00004));
        for i = 1 : length(useful_data)
            k = useful_data(i);
            if k-future_num/2 <= 0
                continue;
            end
            % 第一筆資料(1~5) 第j筆則為(1+5*j)~(5+5*j) 若後面大於筆數，則不用最後那筆 
            if(future_num/2+k>=length(song(num).time))
                break
            end
            %% 選feature    
            c = 0;
            % 加個energy
            c = c + 1;data(c,:) = norm_energy(k-future_num/2:future_num+k-future_num/2);
            % 加個 energy entropy
            c = c + 1;data(c,:) = norm_energy_entropy(k-future_num/2:future_num+k-future_num/2);
            % 加個 spectral_centroid
            c = c + 1;data(c,:) = norm_spec_cen(k-future_num/2:future_num+k-future_num/2);
            % 加個 spectral_flux
            c = c + 1;data(c,:) = norm_spec_flux(k-future_num/2:future_num+k-future_num/2);
            % 加個vocal pitch
            c = c + 1;data(c,:) = norm_vocal_pitch(k-future_num/2:future_num+k-future_num/2);
            % 加個sum of chroma
            c = c + 1;data(c,:) = mean_chroma(k-future_num/2:future_num+k-future_num/2);
            % 加個zcr
            c = c + 1;data(c,:) = norm_zcr(k-future_num/2:future_num+k-future_num/2);
            % 加個spec roll off
            c = c + 1;data(c,:) = norm_spectral_rolloff(k-future_num/2:future_num+k-future_num/2);
            % 加個spec entro
            c = c + 1;data(c,:) = norm_spectral_entropy(k-future_num/2:future_num+k-future_num/2);
            
            answer = song(num).onoff_truth(k);
            if isTesting == 0
                % 隨機丟棄nothing
                rand_remove = rand(1);
                if(strcmp(answer,'Nothing'))
                    if rand_remove < 0.85 %丟棄nothing的機率
                        continue;
                    end
                end
                j = j + 1;
                X_temp{j} = data;
                state_temp{j} = answer.';
            elseif isTesting == 1
                j = j + 1;
                X_temp{k} = data;
                state_temp{k} = answer.';
            end
        end

end
