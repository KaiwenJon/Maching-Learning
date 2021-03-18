clc
clearvars -except net 
load CONVER_ACC9879
mypi = raspi('192.168.137.152','pi','123456');
mycam = cameraboard(mypi,'Resolution','1280x720','Rotation',180);
%mycam = webcam(r);


s_left = servo(mypi,21);
s_right = servo(mypi,20);
writePosition(s_left,53);
writePosition(s_right,110);
while true
    disp('Start photo')
    G_on(mypi); % 綠燈亮開始拍照
for i = 1 : 50
    img = snapshot(mycam);
    img = imrotate(img,90);
    img = imresize(img,[1200 800]);
    img = rgb2gray(img);
    img = medfilt2(img);
    BW = im2bw(img,0.3);
    BW = imresize(BW,[1200 800]);
    BW = ~BW;
    BW = uint8(BW);
    BW = 255*BW;
    BW3(:,:,i) = BW;
    pred_tmp(i) = string(classify(net,BW));
    txt = pred_tmp(i);
    txt2 = string(i);
    BW = insertText(BW,[size(BW,1)*0.5,size(BW,2)*0.5],(txt),'FontSize',50,'BoxOpacity',1);
    BW = insertText(BW,[size(BW,1)*0.5-200,size(BW,2)*0.5-200],(txt2),'FontSize',50,'BoxOpacity',1,'BoxColor','red');
    imshow(BW);
end
nFrames=length(BW3(1,1,:));

tot=round(nFrames/100)*100;
warning('off')
GG1 = factor(tot);
GM=max(GG1);
part=tot/max(GG1);
a=10;
sts_open=zeros(part-a,1);
disp('testing');
Y_on(mypi); %綠暗黃亮 開始計算
G_off(mypi);
for pp=1:(part-a);
    %video = read(xyloObj,(part*(pp-1)+1:1:part(pp)))
    %TTTT=zeors(:,:,GG1);
    label=[(pp-1)*GM+1:1:pp*GM];
    gg=[];
    for classif=1:max(GG1)  
        Y_on(mypi);
        kk=label(classif);
        BW_gotest = BW3(:,:,kk);
        pred1 = classify(net,BW_gotest);     
        if pred1=='C'
            gg=[gg 1];
        elseif pred1=='D'
            gg=[gg 2];
        elseif pred1=='L'
            gg=[gg 3];
        elseif pred1=='O'
            gg=[gg 4];
        end
        Y_off(mypi);
    end
    samples=length(gg);
    sts_open(pp,1)=length(find(gg==4))/samples;
end
length(find(sts_open)==1)/length(sts_open(:,1));
%% 14 part = open;

sts_open2=sts_open(:,1);
sts_open2=sts_open2+rand*0.1;
% trend
KLK=length(sts_open2(:,1));
%
clc
a = 0.01;
b = 0.6;
low = (b-a).*rand(round(KLK/2),1) + a;
a = 0.7;
b = 1;
high = (b-a).*rand(round(KLK/2),1) + a;

O_C_OC_CO=[];

LH=[low;high];
HL=[high;low];
LL=[low;low];
HH=[high;high];

LT=length(HH);
DA=length(sts_open2);
%
if LT>DA
    LH=LH(1:DA,1);
    HL=HL(1:DA,1);
    LL=LL(1:DA,1);
    HH=HH(1:DA,1);
    
else
    sts_open2=sts_open2(1:LT,1);
    
end
R = corrcoef(sts_open2,LL);

O_C_OC_CO=[corrcoef(sts_open2,HH) corrcoef(sts_open2,LL) corrcoef(sts_open2,HL) corrcoef(sts_open2,LH)];
ACA=[O_C_OC_CO(1,2) O_C_OC_CO(1,4) O_C_OC_CO(1,6) O_C_OC_CO(1,8)];
ACTION=find(ACA==max(ACA));

disp('Action start!!!');
disp(pred_tmp)
Y_off(mypi); % 黃暗紅亮 開始行動
R_on(mypi);
pause(2);
R_off(mypi);
switch ACTION
    case 4 % CO 
        disp('CO')
        for i = 1 : 25 % 黃閃
            Y_on(mypi);
            pause(0.1);
            Y_off(mypi);
            pause(0.1);
        end
		dangerous(mypi,s_left,s_right);
        pause(2);
        
	case 3 % OC
        disp('OC')
        for i = 1 : 25 % 黃閃
            Y_on(mypi);
            pause(0.1);
            Y_off(mypi);
            pause(0.1);
        end
        dangerous(mypi,s_left,s_right);
        pause(2);
        
	case 2 % CC
        disp('CC')
        for i = 1 : 25 % 紅閃
            R_on(mypi);
            pause(0.1);
            R_off(mypi);
            pause(0.1);
        end
		hit(mypi,s_left,s_right); % 亮紅打人
        pause(2);
        
        
    case 1 % OO
        disp('OO')
        for i = 1 : 25 % 綠閃
            G_on(mypi);
            pause(0.1);
            G_off(mypi);
            pause(0.1);
        end
		healthy(mypi,s_left,s_right); % 綠燈
        pause(2);
        
end

for j = 1 : 25 %% 重新進行偵測 爆閃
    R_on(mypi);
    pause(0.01);
    G_off(mypi);
    pause(0.01);
    Y_on(mypi);
    pause(0.01);
    R_off(mypi);
    pause(0.01);
    G_on(mypi);
    pause(0.01);
    Y_off(mypi);
    pause(0.01);
end
all_off(mypi);
pause(4);
end

function healthy(mypi,s_left,s_right)
    G_on(mypi);
    Y_off(mypi);
    R_off(mypi);
    writePosition(s_left,53);
    writePosition(s_right,100);
end
function dangerous(mypi,s_left,s_right)
    Y_on(mypi);
    writePosition(s_left,78);
    writePosition(s_right,78);
end
function hit(mypi,s_left,s_right)
    R_on(mypi);
    writePosition(s_left,110);
    writePosition(s_right,53);
end
function all_on(mypi)
    R_on(mypi);
    Y_on(mypi);
    G_on(mypi);
end
function all_off(mypi)
    R_off(mypi);
    Y_off(mypi);
    G_off(mypi);
end
function R_on(mypi)
    writeDigitalPin(mypi,13,1);
end
function R_off(mypi)
    writeDigitalPin(mypi,13,0);
end
function Y_on(mypi)
    writeDigitalPin(mypi,19,1);
end
function Y_off(mypi)
    writeDigitalPin(mypi,19,0);
end
function G_on(mypi)
    writeDigitalPin(mypi,16,1);
end
function G_off(mypi)
    writeDigitalPin(mypi,16,0);
end