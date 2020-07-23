% Student Number = 14062340
% Step 1

clc
clearvars
close all

%% 1) Define number of iterations for GWR
num_it_affine = 6; % informed by previous execution of this script
num_it_non_linear = 6; % informed by previous execution of this script
total_it_num = num_it_non_linear + num_it_affine;
time = zeros(total_it_num,1);
NCC_values = zeros(total_it_num,10);

%% 2) Initialisation Step - Perform rigid registration of all 10 images to 1st image
% perform a rigid only registration to transform images into space of one image
operation = 'reg_aladin -rigOnly';
% use first image as reference image (arbitrary choice - has to be
% one of the 10 images - doesn't matter which one)
ref = ' -ref Step_1_Images/template_0_img.nii.gz';
it_num = 1;
tic
% loop through all 10 images
for image_num = 0:9
    % define floating image as appropriate image
    flo = [' -flo Step_1_Images/template_',num2str(image_num),'_img.nii.gz'];
    % define filename of where to save warped image
    res = [' -res Step_1_Images/warped_image_',num2str(image_num),'_step_1.nii'];
    % define filename of where to save output transformation
    aff = [' -aff Step_1_Images/transform_image_',num2str(image_num),'_step_1.txt'];
    command = [operation, ref, flo, res, aff];
    dos(command)
end
CalcMeanImage(it_num);
time(it_num) = toc;

%% Then perform rigid and affine registration
operation = 'reg_aladin';
for it_num = 2 : num_it_affine
    tic
    % define previous average image as reference image
    ref = [' -ref Step_1_Images/average_image_step_',num2str(it_num-1),'.nii']; 
    % loop through all 10 images
    for image_num = 0:9
        % define floating image as appropriate image
        flo = [' -flo Step_1_Images/template_',num2str(image_num),'_img.nii.gz'];
        % define filename of where to save output warped image
        res = [' -res Step_1_Images/warped_image_',num2str(image_num),'_step_',num2str(it_num),'.nii'];
        % define filename of where to retrieve input transformation
        inaff = [' -inaff Step_1_Images/transform_image_',num2str(image_num),'_step_',num2str(it_num-1),'.txt'];
        % define filename of where to save output transformation
        aff = [' -aff Step_1_Images/transform_image_',num2str(image_num),'_step_',num2str(it_num),'.txt'];
        command = [operation, ref, flo, res, inaff, aff];
        dos(command)
    end
    CalcMeanImage(it_num);
    time(it_num) = toc;
end

%% Then perform non-rigid registration (with zero padding)
operation = 'reg_f3d -pad 0';
for it_num = num_it_affine + 1 : total_it_num
    tic
    % define previous average image as reference image
    ref = [' -ref Step_1_Images/average_image_step_',num2str(it_num-1),'.nii']; 
    % loop through all 10 images
    for image_num = 0:9
        % define floating image as appropriate image
        flo = [' -flo Step_1_Images/template_',num2str(image_num),'_img.nii.gz'];
        % define filename of where to save warped image
        res = [' -res Step_1_Images/warped_image_',num2str(image_num),'_step_',num2str(it_num),'.nii'];        
        % define filename of where to retrieve input transformation from
        % last affine step
        aff = [' -aff Step_1_Images/transform_image_',num2str(image_num),'_step_',num2str(num_it_affine),'.txt'];
        if it_num == total_it_num
            % define filename of where to save output transformation
            cpp = [' -cpp Step_1_Images/transform_image_',num2str(image_num),'_step_',num2str(it_num),'.nii'];
            command = [operation, ref, flo, res, aff, cpp]; 
        else
            command = [operation, ref, flo, res, aff]; 
        end
            dos(command)
    end
    CalcMeanImage(it_num);
    time(it_num) = toc;
end

% calculate total time in minutes
total_min = sum(time)/60;

%% Calc mean NCC at end of each step
for it_num = 1:total_it_num
    % loop through all 10 images
    for image_num = 0:9
            NCC_values(it_num,image_num+1) = CalcMeanNCC(it_num, image_num);
    end
end
%To set line widths and font size for plots
fs=20; LW=1.5;
NCC_value_mean = mean(NCC_values,2);
figure('units','normalized','outerposition',[0 0 1 1])
plot(1:total_it_num,NCC_value_mean,'LineWidth',LW)
xlabel('Iteration Number','FontSize',fs-1,'FontWeight','bold');
ylabel('Mean NCC','FontSize',fs-1,'FontWeight','bold');
grid minor
ax = gca;
ax.FontSize = fs-3;
ylim ([NCC_value_mean(1) 1])
xlim ([1 total_it_num])
saveas(gcf,'Step_1_Images/NCC_fig.m')
saveas(gcf,'Step_1_Images/NCC_fig.jpeg')

%% load central slice 83 of each of 20 average registrations to show progress
% (transverse/axial/horizontal/transaxial plane)

figure('units','normalized','outerposition',[0 0 1 1])
for i = 1:total_it_num
    subplot(3,4,i)
    mean_image= load_untouch_nii(['Step_1_Images/average_image_step_',num2str(i),'.nii']);
    slice = squeeze(mean_image.img(:,83,:));
    imagesc(slice)
    colormap gray
    axis off
    daspect([1 1 1])
    title(num2str(i))
end
saveas(gcf,'Step_1_Images/GWR_progression.m')
saveas(gcf,'Step_1_Images/GWR_progression.jpeg')


%% view all 10 original images
figure('units','normalized','outerposition',[0 0 1 1])
for image_num = 0:9
    image = load_untouch_nii(['Step_1_Images/template_',num2str(image_num),'_img.nii.gz']);
    if image_num == 8 || image_num == 9
        subplot(3,4,image_num+2)
    else
        subplot(3,4,image_num+1)
    end
    imagesc(squeeze(image.img(:,83,:)))
    colormap gray
    daspect([1 1 1])
    axis off
    title(image_num)
end
saveas(gcf,'Step_1_Images/10_images.m')
saveas(gcf,'Step_1_Images/10_images.jpeg')