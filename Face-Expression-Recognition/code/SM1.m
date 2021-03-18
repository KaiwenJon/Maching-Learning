sts_open=ones(10,1)
sts_open2=sts_open(:,1);

if mean(sts_open2,1)>0.9
fprintf('OO\n')
    
    
elseif mean(sts_open2,1)<0.01
fprintf('CC\n')    
    
    
    
else
    fprintf('Trend\n') 
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

end