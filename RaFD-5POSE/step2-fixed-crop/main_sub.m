function [] = main_sub(fn)
    % 计算输出文件名
    [midfn1, midfn2, ~] = fileparts(fn);
    midfn1 = midfn1(9:end);
    fn2 = ['../DATA-CROP-FIX/' midfn1 '/', midfn2 '.png'];
    [dir2,~,~] = fileparts(fn2);
    orig_state = warning('off');
    mkdir(dir2);
    warning(orig_state);
    disp(['   ' fn]);
    disp(['-> ' fn2]);
    
    % 加载图像
    im = imread(fn);
    assert(all(size(im)==[1024 681 3]));
    im = im(1+109:end-300, 1+33:end-33, :);
    assert(all(size(im)==[1024-109-300 681-33-33 3]));
    assert(size(im,1)==size(im,2));
    im = imresize(im, [284 284]);
    assert(all(size(im)==[284 284 3]));
    
    imshow(im);
    imwrite(im(:,:,:), fn2);
end
