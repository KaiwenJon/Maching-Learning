%% input data
test_time = time_test;
test_pitch = YPred_test;
load song_data_with_densetruth
groundtruth = textread('400.txt'); %groundtruth的txt要放在工作目錄
%% 後處理
test_pitch = [test_pitch ; zeros(4,1)];
L = size(test_pitch,1);
circle = 10;
while circle > 0
    %ABA
    for i = 1 : L-2
        if test_pitch(i) == test_pitch(i+2) & test_pitch(i) ~= test_pitch(i+1)
            test_pitch(i+1) = test_pitch(i);
        end
    end
    %ABBA
    for i = 1 : L-3
        if test_pitch(i) == test_pitch(i+3) & test_pitch(i+1) == test_pitch(i+2) & test_pitch(i) ~= test_pitch(i+1)
            test_pitch(i+1) = test_pitch(i);
            test_pitch(i+2) = test_pitch(i);
        end
    end
    %ABBBA
    for i = 1 : L-4
        if test_pitch(i) == test_pitch(i+4) & test_pitch(i+1) == test_pitch(i+2) & test_pitch(i+1) == test_pitch(i+3) & test_pitch(i) ~= test_pitch(i+1)
            test_pitch(i+1) = test_pitch(i);
            test_pitch(i+2) = test_pitch(i);
            test_pitch(i+3) = test_pitch(i);
        end
    end
    %ABC
    for i = 1 : L-2
        if test_pitch(i) ~= test_pitch(i+1) & test_pitch(i) ~= test_pitch(i+2) & test_pitch(i+1) ~= test_pitch(i+2)
            diff01 = abs(test_pitch(i) - test_pitch(i+1));
            diff02 = abs(test_pitch(i+1) - test_pitch(i+2));
            if diff01 <= diff02
                test_pitch(i+1) = test_pitch(i);
            else
                test_pitch(i+1) = test_pitch(i+2);
            end
        end
    end
    %ABBC
    for i = 1 : L-3
        if test_pitch(i) ~= test_pitch(i+1) & test_pitch(i) ~= test_pitch(i+3) & test_pitch(i+1) == test_pitch(i+2) & test_pitch(i+1) ~= test_pitch(i+3)
            diff03 = abs(test_pitch(i) - test_pitch(i+1));
            diff04 = abs(test_pitch(i+1) - test_pitch(i+3));
            if diff03 <= diff04
                test_pitch(i+1) = test_pitch(i);
                test_pitch(i+2) = test_pitch(i);
            else
                test_pitch(i+1) = test_pitch(i+3);
                test_pitch(i+2) = test_pitch(i+3);
            end
        end
    end
    % ABBBC
    for i = 1 : L-4
        if test_pitch(i) ~= test_pitch(i+1) & test_pitch(i) ~= test_pitch(i+4) & test_pitch(i+1) ~= test_pitch(i+4) & test_pitch(i+1) == test_pitch(i+2) & test_pitch(i+1) == test_pitch(i+3)
            diff05 = abs(test_pitch(i) - test_pitch(i+1));
            diff06 = abs(test_pitch(i+1) - test_pitch(i+4));
            if diff05 <= diff06
                test_pitch(i+1) = test_pitch(i);
                test_pitch(i+2) = test_pitch(i);
                test_pitch(i+3) = test_pitch(i);
            else
                test_pitch(i+1) = test_pitch(i+4);
                test_pitch(i+2) = test_pitch(i+4);
                test_pitch(i+3) = test_pitch(i+4);
            end
        end
    end
    
    circle = circle - 1;
end
test_time = [test_time ; zeros(4,1)];
test = [test_time test_pitch];
test((end-3 : end),:) = [];
test_pitch = test(:,2);
test_time = test(:,1);
%% 準確率
accuracy = sum(pitch_test == test_pitch)/size(pitch_test,1);
testing_accuracy = accuracy*100
%% 去掉pitch為0的列
del = find(test_pitch == 0);
test(del,:) = [];
%% 畫圖
figure
for i = 1 : size(groundtruth,1)
    groundtruth_time = [groundtruth(i,1) groundtruth(i,2)];
    groundtruth_pitch = [groundtruth(i,3) groundtruth(i,3)];
    plot(groundtruth_time,groundtruth_pitch,'Color','r','Linewidth',10)
    hold on
end
plot(test_time,test_pitch,'o','Color','b')
%% 輸出成 onset offset pitch
onset = test(1,1);
offset = [];
pitch = [];
for i = 1 : size(test,1)-1
    if (test(i,2) - test(i+1,2)) ~= 0
        onset = [onset ; test(i+1,1)];
        offset = [offset ; test(i,1)];
        pitch = [pitch ; test(i,2)];
    end
end
offset = [offset ; test(end,1)];
pitch = [pitch ; test(end,2)];
pred_text = [onset offset pitch];
%% 儲存檔案 檔案存在工作目錄
fid = fopen(['pred_text.txt'],'w');
[r,c] = size(pred_text);
for i = 1 : r
    for j = 1 : c
        fprintf(fid,'%f\t',pred_text(i,j));
    end
    fprintf(fid,'\r\n');
end
fclose(fid);