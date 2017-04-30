function [] = summary_sub(fn)
    f = fopen(fn);
    epoch = nan;
    record = [];
    while(true)
        line = fgetl(f);
        if(feof(f))
            break;
        end
        assert(~str_begin_with(line, 'Traceback'));
        
        if(str_begin_with(line, '---------- epoch '))
            epoch = sscanf(line, '---------- epoch %f');
        end
        if(str_begin_with(line, 'mean train loss '))
            train_loss_train_acc = sscanf(line, 'mean train loss %f, mean train acc %f');
            record(epoch).train_loss = train_loss_train_acc(1);
            record(epoch).train_acc = train_loss_train_acc(2);
        end
        if(str_begin_with(line, 'mean val loss '))
            val_loss_val_acc = sscanf(line, 'mean val loss %f, mean val acc %f, mean val fuseacc %f');
            record(epoch).val_loss = val_loss_val_acc(1);
            record(epoch).val_acc = val_loss_val_acc(2);
            record(epoch).val_fuseacc = val_loss_val_acc(3);
        end
    end
    assert(str_begin_with(line, '---------- epoch '));
    fclose(f);
    
    % ??
    close all;
    figure;
    set(gcf,'Position', get(0,'ScreenSize'));
    set(gcf,'Name',fn)
    
    subplot(2,2,1);
    semilogy([[record.train_loss]' [record.val_loss]']);
    hold on
    [~, bestEpoch] = min([record.val_loss]);
    tmp = [record.val_loss];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('loss');
    legend('train loss','val loss');
    grid on;
    
    subplot(2,2,2);
    plot([[record.train_acc]' [record.val_acc]' [record.val_fuseacc]']);
    hold on
    [~, bestEpoch] = max([record.val_fuseacc]);
    tmp = [record.val_fuseacc];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    [~, bestEpoch] = max([record.val_acc]);
    tmp = [record.val_acc];
    scatter(bestEpoch, tmp(bestEpoch));
    text(bestEpoch, tmp(bestEpoch), ['(' num2str(bestEpoch) ', ' num2str(tmp(bestEpoch)) ')']);
    hold off
    title('acc');
    legend('train acc','val acc','val fuseacc');
    ylim([0 1])
    grid on;
    set(gca,'ytick',0:0.05:1);
    
    subplot(2,2,4);
    smooth_span = 10;
    plot([smooth([record.train_acc]', smooth_span) smooth([record.val_acc]', smooth_span) smooth([record.val_fuseacc]', smooth_span)]);
    title('smooth acc');
    legend('train acc', 'val acc', 'val fuseacc');
    ylim([0 1])
    grid on;
    set(gca,'ytick',0:0.05:1);
    
    drawnow;
    
    save_current_fig_as_pdf([fn '.pdf'])
end
