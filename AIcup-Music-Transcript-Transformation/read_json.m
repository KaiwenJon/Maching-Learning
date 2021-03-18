clc
clear
for i = 1:500
    str1 = './MIR-ST500/';
    num = int2str(i);
    str2 = '_feature.json';
    fname = [str1,'/',num,'/',num,str2];
    fid = fopen(fname); 
    raw = fread(fid,inf); 
    str = char(raw'); 
    fclose(fid);
    song(i) = jsondecode(str);%%% feature
end
%%
clc
clear
fname = 'finalstr_final6.json';
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid);
upload_result = jsondecode(str);%%% feature



