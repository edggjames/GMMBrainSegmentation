% Student Number = 14062340
% Step 2

clc
clearvars
close all

%% Resample segmentation maps into mean space using transformations obtained
% from GroupWiseReg.m

operation = 'reg_resample -inter 0 -pad 0';
% choose 0 order interpolation to maintain binary classification (i.e. only
% 0, 1, 2, or 3 labels - no inbetween values for NN interpolation). 

% use mean template image from GroupWiseReg as reference image
ref = ' -ref Step_1_Images/average_image_step_12.nii'; 
% loop through all 10 images
tic
for image_num = 0:9
    % define floating image
    flo = [' -flo Step_1_Images/template_',num2str(image_num),'_seg.nii.gz'];
    % define filename of where resampled segementation image should be saved
    res = [' -res Step_1_Images/seg_image_mean_space_',num2str(image_num),'.nii'];
    % define the file containing the final transformation parameterisation
    % from GWR
    trans = [' -trans Step_1_Images/transform_image_',num2str(image_num),'_step_12.nii'];
    command = [operation, ref, flo, res, trans];
    dos(command)
end

%% All the segementated images are now the same size as the mean image
% template and in the same space - load them to calculate tissue probability
% maps - and concatenate into one 4D matrix

% get dimensions of mean space image
mean_image = load_untouch_nii('Step_1_Images/average_image_step_12.nii');
[a, b, c] = size(mean_image.img);
class(mean_image.img);
% use this as a template structure for TPMs
TPM_0 = mean_image;
TPM_1 = mean_image;
TPM_2 = mean_image;
TPM_3 = mean_image;
clear mean_image
d = 10; % number of images
% initialise dimensions of 4D matrix to hold all mean images in
stacked_image = zeros(a,b,c,d);

% loop through all 10 images and assign to stacked_image
for image_num = 0:9
    seg_image_mean_space = ...
        load_untouch_nii(['Step_1_Images/seg_image_mean_space_',num2str(image_num),'.nii']);
    stacked_image(:,:,:,image_num+1) = seg_image_mean_space.img;
    size(seg_image_mean_space.img)
    class(seg_image_mean_space.img)
    clear seg_image_mean_space
end
    
    
%% Initialise matrices to hold 4 TPMs in and calculate them

TPM_0.img = single(zeros(a,b,c)); % - non-brain
TPM_1.img = single(zeros(a,b,c)); % - CSF
TPM_2.img = single(zeros(a,b,c)); % - GM
TPM_3.img = single(zeros(a,b,c)); % - WM

for x = 1:a
    for y = 1:b
        for z = 1:c
            % Initialise counters at each voxel
            sum_0 = 0; sum_1 = 0; sum_2 = 0; sum_3 = 0;
            % count all 0s and divide by d = prob of 0 at this voxel, do
            % for other 3 labels
            for image_num = 1:d
                value = stacked_image(x,y,z,image_num);
                if value == 0
                    sum_0 = sum_0 + 1;
                elseif value == 1
                    sum_1 = sum_1 + 1;
                elseif value == 2
                    sum_2 = sum_2 + 1;
                elseif value == 3
                    sum_3 = sum_3 + 1;
                end
            end
            TPM_0.img(x,y,z) = sum_0/d;
            TPM_1.img(x,y,z) = sum_1/d;
            TPM_2.img(x,y,z) = sum_2/d;
            TPM_3.img(x,y,z) = sum_3/d;
        end
    end
end
time = toc;
clear stacked_image

%% save all 4 TPMs in correct format
save_untouch_nii(TPM_0,'Step_1_Images/TPM_0_step_1.nii')
clear TPM_0_struct
save_untouch_nii(TPM_1,'Step_1_Images/TPM_1_step_1.nii')
clear TPM_1_struct
save_untouch_nii(TPM_2,'Step_1_Images/TPM_2_step_1.nii')
clear TPM_2_struct
save_untouch_nii(TPM_3,'Step_1_Images/TPM_3_step_1.nii')
clear TPM_3_struct


%% Show a slice from each TPM
fs = 20; % fontsize for plots
figure('units','normalized','outerposition',[0 0 1 1])

subplot(2,2,1)
TPM_0 = load_untouch_nii('Step_1_Images/TPM_0_step_1.nii');
size(TPM_0.img)
class(TPM_0.img)
slice_0 = squeeze(TPM_0.img(:,83,:));
imagesc(slice_0)
colormap gray
axis off
title('Non-Brain','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,2)
TPM_1 = load_untouch_nii('Step_1_Images/TPM_1_step_1.nii');
slice_1 = squeeze(TPM_1.img(:,83,:));
imagesc(slice_1)
colormap gray
axis off
title('CSF','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,3)
TPM_2 = load_untouch_nii('Step_1_Images/TPM_2_step_1.nii');
slice_2 = squeeze(TPM_2.img(:,83,:));
imagesc(slice_2)
colormap gray
axis off
title('GM','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,4)
TPM_3 = load_untouch_nii('Step_1_Images/TPM_3_step_1.nii');
slice_3 = squeeze(TPM_3.img(:,83,:));
imagesc(slice_3)
colormap gray
axis off
title('WM','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])
saveas(gcf,'Step_1_Images/TPM_step_1.m')
saveas(gcf,'Step_1_Images/TPM_step_1.jpeg')

% check all four volumes add to 1 at each voxel
total_vol = TPM_0.img + TPM_1.img + TPM_2.img + TPM_3.img;
disp('Mean total voxel value = ');
mean(total_vol(:))

%% Check alignment of TPMs with template
GWR_template = load_untouch_nii('Step_1_Images/average_image_step_9.nii');
GWR_template = GWR_template.img;
GWR_template (GWR_template < 0) = 0; % convert to all zero or greater
GWR_template = GWR_template./max(GWR_template(:)); % normalise

%overlay with TPMs
overlay_0 = TPM_0.img + GWR_template;
overlay_1 = TPM_1.img + GWR_template;
overlay_2 = TPM_2.img + GWR_template;
overlay_3 = TPM_3.img + GWR_template;

fs = 20; % fontsize for plots
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
slice_0 = squeeze(overlay_0(:,83,:));
slice_0 (isnan(slice_0)) = 1; % convert boundary NaNs to 1s.  
slice_0 (slice_0 > 1) = 1; % convert to all 1 or less

imagesc(slice_0)
colormap gray
axis off
title('Non-Brain','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,2)
slice_1 = squeeze(overlay_1(:,83,:));
imagesc(slice_1)
colormap gray
axis off
title('CSF','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,3)
slice_2 = squeeze(overlay_2(:,83,:));
imagesc(slice_2)
colormap gray
axis off
title('GM','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])

subplot(2,2,4)
slice_3 = squeeze(overlay_3(:,83,:));
imagesc(slice_3)
colormap gray
axis off
title('WM','FontSize',fs+1,'FontWeight','bold');
daspect([1 1 1])
saveas(gcf,'Step_1_Images/TPM_step_1_overlay.m')
saveas(gcf,'Step_1_Images/TPM_step_1_overlay.jpeg')
