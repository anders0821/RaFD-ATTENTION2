clc;
clear;
close all;
rng default;

addpath C:\sdk\M\INface_tool;
addpath C:\sdk\M\INface_tool\auxilary;
addpath C:\sdk\M\INface_tool\mex;
addpath C:\sdk\M\INface_tool\histograms;
addpath C:\sdk\M\INface_tool\photometric;
addpath C:\sdk\M\INface_tool\postprocessors;
addpath C:\sdk\M\INface_tool\demos;


% ö���ļ�
fns = dir_recursive('../DATA-CROP-FIX/', '*.png');
%assert(numel(fns)==(13202+(245+241+258+262+259)+(865+400+333))*2);

% ��������ļ��б� ��ֹ�������ݿ�ͼ���С��ͬ ����parfor��������ƽ��
fns = fns(randperm(numel(fns)));
%assert(numel(fns)==(13202+(245+241+258+262+259)+(865+400+333))*2);

% �޸Ĳ����� �ɴ��ڻ�С���߳���
cluster=parcluster;
cluster.NumWorkers=3;
% ���д����ļ�
parfor i=1:numel(fns)
    disp([num2str(i) ' / ' num2str(numel(fns))]);
    main_sub1(fns{i});
    %main_sub2(fns{i});
end
