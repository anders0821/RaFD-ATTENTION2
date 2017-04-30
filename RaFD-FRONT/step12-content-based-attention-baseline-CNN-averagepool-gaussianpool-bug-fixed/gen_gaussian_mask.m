clc;
clear;
close all;


mask = fspecial('gaussian', [7 9], 2);
sum(mask(:))

fprintf('[\n')
for i=1:size(mask,1)
    fprintf('[')
    for j=1:size(mask,2)
        fprintf('%s, ', mask(i,j))
    end
    fprintf('],\n')
end
fprintf(']')

subplot(1,2,1)
imshow(mask, [])


mask = fspecial('gaussian', [17 17], 4);
sum(mask(:))

fprintf('[\n')
for i=1:size(mask,1)
    fprintf('[')
    for j=1:size(mask,2)
        fprintf('%s, ', mask(i,j))
    end
    fprintf('],\n')
end
fprintf(']')

subplot(1,2,2)
imshow(mask, [])
save_current_fig_as_pdf('gen_gaussian_mask.pdf');