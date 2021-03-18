%% Main
clc
clearvars -except net song state_pred testsong1 testsong2 testsong
%%
testsong = [testsong1 testsong2];
song = testsong;
global Feature_name;
Feature_name = fieldnames(song);
global future_num;
future_num = 20 ;%������
%% �i��ܤ@���q�e�ϴ���
% �D�@���Ӵ���
Bingo = 0;
%song = testsong;
for test_num = 601:650
    [Xtest,state_truth] = Data_Constructor(song,test_num,1);
    %���զ��F�誺�a��
    useful_data_ind = find(song(test_num).vocal_pitch ~= 0 | song(test_num).energy >= 0.00004)
    useless_data_ind = find(~(song(test_num).vocal_pitch ~= 0 | song(test_num).energy >= 0.00004))
    
    for i = 1 : length(useful_data_ind)
        k =useful_data_ind(i);
        disp(test_num);
        disp(k)
        if k-future_num/2 <= 0
            state_pred(test_num,k) = 'No_predict';
        elseif(future_num/2+k>=length(song(test_num).time))
           state_pred(test_num,k) = 'No_predict';
        else
        state_pred(test_num,k) = classify(net,Xtest{k});
        end
    end
    % �Y�Spredict�A�⵲�G�令nothing
    state_pred(test_num,useless_data_ind) = 'No_predict';
    state_pred = string(state_pred);
end

%% �e�ϴ���
plot_num = test_num;
total_point = length(state_pred(plot_num,:));
for i = 1 : total_point
    t = 0.016 + (i-1)*0.032;
    x = [t t];
    y = [0 80];
    if strcmp(state_pred(plot_num,i),'on')
        plot(x,y,'Color',[0.4660 0.6740 0.1880],'LineWidth',2)%��
    elseif strcmp(state_pred(plot_num,i),'off')
        plot(x,y,'Color',[0.6350 0.0780 0.1840],'LineWidth',2)%��
    elseif strcmp(state_pred(plot_num,i),'No_predict')
        plot(x,y,'Color','black','LineWidth',0.5)
    elseif strcmp(state_pred(plot_num,i),'both')
        plot(x,y,'Color',[0.9290 0.6940 0.1250],'LineWidth',3)%��
%         elseif strcmp(state_pred(p),'Nothing')
%             plot(x,y,'Color','black','LineWidth',0.5)
    end
    hold on
end
plot(testsong(plot_num).time,testsong(plot_num).vocal_pitch,'o','Color','b')
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
    % ��vocal pitch����0���@�t�C��ƥh�� 
    % ��energy > 0.0001�]�n���I
    useful_data = find(song(num).vocal_pitch ~= 0 | song(num).energy >= 0.00004);
    useless_data = find(~(song(num).vocal_pitch ~= 0 | song(num).energy >= 0.00004));
        for i = 1 : length(useful_data)
            k = useful_data(i);
            if k-future_num/2 <= 0
                continue;
            end
            % �Ĥ@�����(1~5) ��j���h��(1+5*j)~(5+5*j) �Y�᭱�j�󵧼ơA�h���γ̫ᨺ�� 
            if(future_num/2+k>=length(song(num).time))
                break
            end
            %% ��feature    
            c = 0;
            % �[��energy
            c = c + 1;data(c,:) = norm_energy(k-future_num/2:future_num+k-future_num/2);
            % �[�� energy entropy
            c = c + 1;data(c,:) = norm_energy_entropy(k-future_num/2:future_num+k-future_num/2);
            % �[�� spectral_centroid
            c = c + 1;data(c,:) = norm_spec_cen(k-future_num/2:future_num+k-future_num/2);
            % �[�� spectral_flux
            c = c + 1;data(c,:) = norm_spec_flux(k-future_num/2:future_num+k-future_num/2);
            % �[��vocal pitch
            c = c + 1;data(c,:) = norm_vocal_pitch(k-future_num/2:future_num+k-future_num/2);
            % �[��sum of chroma
            c = c + 1;data(c,:) = mean_chroma(k-future_num/2:future_num+k-future_num/2);
            % �[��zcr
            c = c + 1;data(c,:) = norm_zcr(k-future_num/2:future_num+k-future_num/2);
            % �[��spec roll off
            c = c + 1;data(c,:) = norm_spectral_rolloff(k-future_num/2:future_num+k-future_num/2);
            % �[��spec entro
            c = c + 1;data(c,:) = norm_spectral_entropy(k-future_num/2:future_num+k-future_num/2);
            
%             answer = song(num).onoff_truth(k);
            if isTesting == 0
                % �H�����nothing
                rand_remove = rand(1);
                if(strcmp(answer,'Nothing'))
                    if rand_remove < 0.85 %���nothing�����v
                        continue;
                    end
                end
                j = j + 1;
                X_temp{j} = data;
%                 state_temp{j} = answer.';
            elseif isTesting == 1
                j = j + 1;
                X_temp{k} = data;
%                 state_temp{k} = answer.';
            end
        end

end
