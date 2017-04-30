clc;
clear;
close all;

% ö���ļ�
fns = dir_recursive('../DATA-CROP-RAW-IN-IS/', '*.png');

% ���д����ļ�
X = zeros(160, 125, numel(fns), 'single');
LBL = zeros(numel(fns), 1, 'uint8');
FOLD = zeros(numel(fns), 1, 'uint8');
for i=1:numel(fns)
    disp([num2str(i) ' / ' num2str(numel(fns))]);
    [im, lbl, cv] = main_sub(fns{i});
    im = rgb2gray(im);% gray
    im = single(im)/256;% 0-1
    X(:,:,i) = im;
    LBL(i) = lbl;
    FOLD(i) = cv;
end

% zscore by scalar
MEAN = mean(X(:))
STD = std(X(:))
X = (X-MEAN)/STD;
mean(X(:))
std(X(:))

% ����Ȼͼ��ͬ���������оֲ�ͳ�����ԣ�by map�ᶪʧ��Ϣ
% zscore by map
% MEAN = mean(IM,3);
% STD = std(IM,0,3);
% IM = (IM - repmat(MEAN, [1,1,size(IM,3)])) ./ repmat(STD, [1,1,size(IM,3)]);
% mean(IM,3)
% std(IM,0,3)

% ���ӻ�
% figure
% subplot(1,2,1)
% imshow(MEAN,[])
% subplot(1,2,2)
% imshow(STD,[])
figure
for i=1:100
    subplot(10,10,i)
    imshow(X(:,:,i),[])
end

% ����mat
save('../DATA-CROP-RAW-IN-IS.mat', 'X', 'LBL', 'FOLD');
