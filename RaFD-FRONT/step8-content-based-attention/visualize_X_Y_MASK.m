% clc;
% clear;
close all;

!copy /B blank.mat+.\100\X-Y-MASK.mat .\100\X-Y-MASK.mat.mat
load ./100/X-Y-MASK.mat.mat
CROP_X_VAL = cat(4, TRAIN_CROP_X_VAL, VAL_CROP_X_VAL);
MASK_VAL = cat(4, TRAIN_MASK_VAL, VAL_MASK_VAL);
Y = cat(2, TRAIN_Y, VAL_Y);
clear TRAIN* VAL*

for i=1:10
    subplot(10,4,(i-1)*4+1);
    x = squeeze(CROP_X_VAL(:,:,:,i));
    imshow(x,[]);
    [~, lbl] = max(Y(:,i));
    title(num2str(lbl));
    
    subplot(10,4,(i-1)*4+2);
    mask = squeeze(MASK_VAL(:,:,:,i));
    imshow(mask,[]);
    
    subplot(10,4,(i-1)*4+3);
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    imshow(mask,[]);
    
    subplot(10,4,(i-1)*4+4);
    imshow(x.*mask,[]);
end


avg_x = {0,0,0,0,0,0,0};
avg_mask = {0,0,0,0,0,0,0};

for i=1:size(CROP_X_VAL,4)
    x = squeeze(CROP_X_VAL(:,:,:,i));
    [~, lbl] = max(Y(:,i));
    
    mask = squeeze(MASK_VAL(:,:,:,i));
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = conv2(mask, ones(3,3)/9, 'same');
    mask = imresize(mask, 2, 'nearest');
    
    avg_x{lbl} = avg_x{lbl}+x;
    avg_mask{lbl} = avg_mask{lbl}+mask;
end

figure
subplot(3,8,(1-1)*8+1+0)
imshow(zeros(1), []);
title('mean of the input')
subplot(3,8,(2-1)*8+1+0)
imshow(zeros(1), []);
title(sprintf('mean of the weight map \\alpha \n (projected back to the input space)'))
subplot(3,8,(3-1)*8+1+0)
imshow(zeros(1), []);
title(sprintf('mean of the input \n \\times \n mean of the weight map \\alpha \n (projected back to the input space)'))
for i=1:7
    subplot(3,8,(1-1)*8+1+i)
    imshow(avg_x{i}, []);
    lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    title(lblStrs{i});
    
    subplot(3,8,(2-1)*8+1+i)
    imshow(avg_mask{i}, []);
    subplot(3,8,(3-1)*8+1+i)
    imshow(avg_x{i}.*avg_mask{i}, []);
end
tightfig