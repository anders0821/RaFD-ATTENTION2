function [im, lblExp, lblPse, cv] = main_sub(fn)
    % ��ȡͼƬ
    im = imread(fn);
    assert(all(size(im)==[284 284 3]));
    
    % ��ȡ�����ǩ0-6
    lblExpStrs = {'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'};
    lblExp = -1;
    for j=0:6
        if numel(strfind(fn, lblExpStrs{j+1}))>0
            assert(lblExp==-1);
            lblExp = j;
        end
    end
    lblExp = uint8(lblExp);
    assert(lblExp>=0 && lblExp<=6);
    
    % ��ȡ��̬��ǩ0-4
    lblPseStrs = {'Rafd000_', 'Rafd045_', 'Rafd090_', 'Rafd135_', 'Rafd180_'};
    lblPse = -1;
    for j=0:4
        if numel(strfind(fn, lblPseStrs{j+1}))>0
            assert(lblPse==-1);
            lblPse = j;
        end
    end
    lblPse = uint8(lblPse);
    assert(lblPse>=0 && lblPse<=4);
    
    % ��ȡ�Ӽ�����0-5
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
