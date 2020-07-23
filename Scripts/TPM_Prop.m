% Student Number = 14062340
% Step 2

clc
clearvars
close all

%% Form binary mask for float in floating image space
% Then use the actual warped template image as a mask after binarising it in
% subsequent registration steps. specify nifty reg tool
operation = 'reg_tools.exe -bin';
% specify input image and output image
in = ' -in Step_1_Images/average_image_step_12.nii';
out = ' -out Step_2_Images/binarised_mask.nii'; 
command = [operation, in, out];
dos(command)

%% The first step is to globally (affine) register each image to the groupwise template
flo = ' -flo Step_1_Images/average_image_step_12.nii';
fmask = ' -fmask Step_2_Images/binarised_mask.nii';
NCC = zeros(20, 2);
operation = 'reg_aladin'; 
tic
for image_num = 0:19
    % use new image as reference
    ref = [' -ref Step_2_Images/img_',num2str(image_num),'.nii.gz'];
    % define filename of where to save the output transformation
    aff = [' -aff Step_2_Images/transform_seg_prop_',num2str(image_num),'_aff_step.txt'];
    % define filename of where to save output warped image
    res = [' -res Step_2_Images/warped_image_seg_prop_',num2str(image_num),'_step_1.nii'];
    command = [operation, ref, flo, aff, res, fmask];
    dos(command)
    NCC(image_num+1,1) = CalcNCC(image_num,1);
end
    

%% Then perform non-linear registration
operation = 'reg_f3d -pad 0';

for image_num = 0:19
    ref = [' -ref Step_2_Images/img_',num2str(image_num),'.nii.gz'];
    % define filename of where to get input transformation
    aff = [' -aff Step_2_Images/transform_seg_prop_',num2str(image_num),'_aff_step.txt'];
    % define filemane of where to save output warped image
    res = [' -res Step_2_Images/warped_image_seg_prop_',num2str(image_num),'_step_2.nii'];
    % define filename of where to write the output transformation
    cpp = [' -cpp Step_2_Images/transform_seg_prop_',num2str(image_num),'_nrr_step.nii'];
    command = [operation, ref, flo, res, aff, cpp, fmask];
    dos(command)
    NCC(image_num+1,2) = CalcNCC(image_num,2);
end

%% The non-linear transformation parametrisation can now be used to propagate 
% the TPMS (x4) into the space of each of the 20 images 
% Since the values are probabilities, use a first order linear
% interpolation scheme. Use 0 padding for non-brain prior, and use 1
% padding for all others
for image_num = 0:19
    % use new image as reference
    ref = [' -ref Step_2_Images/img_',num2str(image_num),'.nii.gz'];
    % define the file containing the final transformation parameterisation
    trans = [' -trans Step_2_Images/transform_seg_prop_',num2str(image_num),'_nrr_step.nii'];
    for labels = 0:3
        if labels ~= 0
            operation = 'reg_resample -inter 1 -pad 0';
        elseif labels == 0
            operation = 'reg_resample -inter 1 -pad 1';
        end
        % define filename of where resampled TPM image should be saved
        res = [' -res Step_2_Images/TPM_',num2str(labels),'_image_',num2str(image_num),'_step_2.nii'];
        % Use relevant TPM label from step 1 as floating image
        flo = [' -flo Step_1_Images/TPM_',num2str(labels),'_step_1.nii'];
        command = [operation, ref, flo, res, trans];
        dos(command)
    end
end
% This will produce 4 TPMs for each image (i.e. 80 in total)
time = toc; time = time/60;

%% Show stepwise results of registration for each image
fs = 20; % fontsize for plots
LW = 1.5; % linewidth for plots

for image_num = 0:19
    figure('name',['Image - ',num2str(image_num)],'units','normalized','outerposition',[0 0 1 1])
    
    subplot(2,2,1)
    original_template = niftiread('Step_1_Images/average_image_step_12.nii');
    slice_disp = squeeze(original_template(:,83,:));
    imagesc(slice_disp)
    colormap gray
    axis off
    title('Float','FontSize',fs+1,'FontWeight','bold');
    daspect([1 1 1])
    
    subplot(2,2,2)
    original_vol = niftiread(['Step_2_Images/img_',num2str(image_num),'.nii.gz']);
    slice_disp = squeeze(original_vol(:,83,:));
    imagesc(slice_disp)
    colormap gray
    axis off
    title('Reference','FontSize',fs+1,'FontWeight','bold');
    daspect([1 1 1])
    
    subplot(2,2,3)
    vol_step_1 = niftiread(['Step_2_Images/warped_image_seg_prop_',num2str(image_num),'_step_1.nii']);
    slice_disp = squeeze(vol_step_1(:,83,:));
    imagesc(slice_disp)
    colormap gray
    axis off
    title('Warped - Affine','FontSize',fs+1,'FontWeight','bold');
    daspect([1 1 1])
    
    subplot(2,2,4)
    vol_step_2 = niftiread(['Step_2_Images/warped_image_seg_prop_',num2str(image_num),'_step_2.nii']);
    slice_disp = squeeze(vol_step_2(:,83,:));
    imagesc(slice_disp)
    colormap gray
    axis off
    title('Warped - NRR','FontSize',fs+1,'FontWeight','bold');
    daspect([1 1 1])
    
    saveas(gcf,['Step_2_Images/Seg_Prop_image_',num2str(image_num),'.m'])
    saveas(gcf,['Step_2_Images/Seg_Prop_image_',num2str(image_num),'.jpeg'])
end

%% Plot NCC values
figure('units','normalized','outerposition',[0 0 1 1])
plot(0:19,NCC(:,2),'LineWidth',LW)
hold on
plot(0:19,NCC(:,1),'LineWidth',LW)
hold off
grid minor 
xlabel('Image Number','FontSize',fs-1,'FontWeight','bold');
ylabel('NCC','FontSize',fs-1,'FontWeight','bold');
ax = gca;
ax.FontSize = fs-3;
ylim ([min(min(NCC)) 1])
xlim ([0 19])
legend('NRR','Affine')
saveas(gcf,'Step_2_Images/NCC_fig_step_2.m')
saveas(gcf,'Step_2_Images/NCC_fig_step_2.jpeg')

%% Check alignment of TPMs with each image in an overlay figure
for image_num = 0:19
    
    image = load_untouch_nii(['Step_2_Images/img_',num2str(image_num),'.nii.gz']);
    TPM_0 = load_untouch_nii(['Step_2_Images/TPM_0_image_',num2str(image_num),'_step_2.nii']);
    TPM_1 = load_untouch_nii(['Step_2_Images/TPM_1_image_',num2str(image_num),'_step_2.nii']);
    TPM_2 = load_untouch_nii(['Step_2_Images/TPM_2_image_',num2str(image_num),'_step_2.nii']);
    TPM_3 = load_untouch_nii(['Step_2_Images/TPM_3_image_',num2str(image_num),'_step_2.nii']);
    image = single(image.img);
    image (image < 0) = 0; % convert to all zero or greater
    image = image./max(image(:)); % normalise
    TPM_0 = TPM_0.img; TPM_1 = TPM_1.img; TPM_2 = TPM_2.img; TPM_3 = TPM_3.img; 
    
    % check all four volumes add to 1 at each voxel
    disp('Mean voxel value for this volume = ');
    total_vol = TPM_0 + TPM_1 + TPM_2 + TPM_3;
    mean(total_vol(:))
    
    %overlay with TPMs
    overlay_0 = TPM_0 + image;
    overlay_1 = TPM_1 + image;
    overlay_2 = TPM_2 + image;
    overlay_3 = TPM_3 + image;

    figure('name',['Image - ',num2str(image_num)],'units','normalized','outerposition',[0 0 1 1])
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
     
    saveas(gcf,['Step_2_Images/TPM_overlay_image_',num2str(image_num),'.jpeg'])
    saveas(gcf,['Step_2_Images/TPM_overlay__image_',num2str(image_num),'.m'])
end
