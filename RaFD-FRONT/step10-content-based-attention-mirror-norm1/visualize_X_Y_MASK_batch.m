clc;
clear;
close all;

for I=5100:100:6000
    eval(['!copy 100\X-Y-MASK-' num2str(I) '.mat 100\X-Y-MASK.mat'])
    visualize_X_Y_MASK
    save_current_fig_as_pdf(['100\X-Y-MASK-' num2str(I) '.pdf'])
end
