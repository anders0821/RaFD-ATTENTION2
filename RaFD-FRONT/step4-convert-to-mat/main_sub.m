function [im, lbl, cv] = main_sub(fn)
    % 读取图片
    im = imread(fn);
    assert(all(size(im)==[160 125 3]));
    
    % 提取标签0-6
    lblStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    lbl = -1;
    for j=0:6
        if numel(strfind(fn, lblStrs{j+1}))>0
            assert(lbl==-1);
            lbl = j;
        end
    end
    lbl = uint8(lbl);
    assert(lbl>=0 && lbl<=6);
    
    % 提取子集号码0-5
    cvStrs = {'Cv1', 'Cv2', 'Cv3', 'Cv4', 'Cv5', 'Cv6', 'Cv7'};
    cv = -1;
    for j=0:4
        if numel(strfind(fn, cvStrs{j+1}))>0
            assert(cv==-1);
            cv = j;
        end
    end
    cv = uint8(cv);
    assert(cv>=0 && cv<=4);
end
