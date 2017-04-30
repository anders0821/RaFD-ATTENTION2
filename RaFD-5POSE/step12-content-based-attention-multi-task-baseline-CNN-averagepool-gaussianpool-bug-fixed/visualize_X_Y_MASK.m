clc;
clear;
close all;
drawnow;

for epoch=90:10:130
    clearvars -except epoch
    close all;
    drawnow;

    disp(epoch);
    eval(['!copy /B blank.mat+.\100\X-Y-MASK-' num2str(epoch) '.mat C:\Users\anders\Desktop\tmp.mat'])
    load C:\Users\anders\Desktop\tmp.mat
    CROP_X_VAL = cat(4, TRAIN_CROP_X_VAL, VAL_CROP_X_VAL);
    MASK_VAL = cat(4, TRAIN_MASK_VAL, VAL_MASK_VAL);
    Y_EXP = cat(2, TRAIN_Y_EXP, VAL_Y_EXP);
    Y_PSE = cat(2, TRAIN_Y_PSE, VAL_Y_PSE);
    clear TRAIN* VAL*

    for i=1:10
        subplot(10,4,(i-1)*4+1);
        x = squeeze(CROP_X_VAL(:,:,:,i));
        imshow(x,[]);
        [~, exp] = max(Y_EXP(:,i));
        title(num2str(exp));

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
    save_current_fig_as_pdf(['.\100\' num2str(epoch) 'a.pdf'])

    avg_x = {0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0};
    avg_mask = {0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0
        0,0,0,0,0,0,0};

    for i=1:size(CROP_X_VAL,4)
        x = squeeze(CROP_X_VAL(:,:,:,i));
        [~, exp] = max(Y_EXP(:,i));
        [~, pse] = max(Y_PSE(:,i));

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

        avg_x{pse,exp} = avg_x{pse,exp}+x;
        avg_mask{pse,exp} = avg_mask{pse,exp}+mask;
    end

    figure
    for i=1:5
        for j=1:7
            subplot(5,7,(i-1)*7+j)
            imshow(avg_x{i,j}, []);
            lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
            title(lblStrs{j});
        end
    end
    save_current_fig_as_pdf(['.\100\' num2str(epoch) 'b.pdf'])

    figure
    for i=1:5
        for j=1:7
            subplot(5,7,(i-1)*7+j)
            imshow(avg_mask{i,j}, []);
            lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
            title(lblStrs{j});
        end
    end
    save_current_fig_as_pdf(['.\100\' num2str(epoch) 'c.pdf'])

    figure
    for i=1:5
        for j=1:7
            subplot(5,7,(i-1)*7+j)
            imshow(avg_x{i,j}.*avg_mask{i,j}, []);
            lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
            title(lblStrs{j});
        end
    end
    save_current_fig_as_pdf(['.\100\' num2str(epoch) 'd.pdf'])

    delete C:\Users\anders\Desktop\tmp.mat
end