function [] = main_sub1(fn)
    % ��������ļ���
    assert(strcmp(fn(1:17), '../DATA-CROP-FIX/'));
    fn2 = [fn(1:16) '-IN-IS' fn(17:end)];
    [dir2,~,~] = fileparts(fn2);
    orig_state = warning('off');
    mkdir(dir2);
    warning(orig_state);
    disp(['   ' fn]);
    disp(['-> ' fn2]);
    
    % ����ͼ��
    im = imread(fn);
    
    % rgbh->sv
    hsv = rgb2hsv(im);% 0-1 double
    gray = hsv(:,:,3);
    assert(all(size(gray)==[284 284]));
    
    % IN
    gray = gray*255.0;
    gray = isotropic_smoothing(gray);% demo������ͼ���ʽ��0-255 double 128*128
    gray = gray/255.0;
    
    % hsv->rgb
    hsv(:,:,3) = gray;
    im = hsv2rgb(hsv);
    
    % �洢ͼ��
    imwrite(im, fn2);
end
