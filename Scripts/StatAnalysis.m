% Student Number = 14062340
% Step 3

clc
close all
clearvars

% Enter template number that was used to optimise segmentation parameters (0:9)
template = 2;
% Enter implementation parameter values that were used to segment the 20 images
MRF_beta = 0.3;
bias_order = 3;

%To set line widths and font size for plots
fs=20;
LW=1.5;

%% Create data matrix
% 1) first column is age of subject
data_matrix(:,1) = [29.6;38.0;42.2;53.7;54.6;63.0;63.1;63.2;65.0;68.1;70.4;
                    71.5;72.3;73.8;76.1;79.8;80.7;83.5;84.1;84.4];
                
% 2,3,4) load and loop through all 19 segmented volumes and sum each of
% tissue label, and calculate voxel dimensions of each image
for image_num = 0:19
    % load categorical segmentation and reshape into 1D vector
    vol = load_untouch_nii(['Segmented_Volumes/categorical_segmentation_image_',...
            num2str(image_num),'_beta_',num2str(MRF_beta*10),'_bias_',...
            num2str(bias_order),'_template_',num2str(template),'.nii']);
    seg_vol = reshape(vol.img,[],1);
    % calculate volume of each voxel and store in data matrix
    vox_dim = vol.hdr.dime.pixdim(2:4);
    vox_vol = prod(vox_dim)/1000; % convert from mm^3 to cm^3

    % count number of voxels and number of labels of each class: [0,1,2,3]
    num_vox = length(seg_vol);
    num_1 = zeros(num_vox,1); num_2 = zeros(num_vox,1); num_3 = zeros(num_vox,1);
    num_1(seg_vol==1) = 1; num_1 = sum(num_1);
    num_2(seg_vol==2) = 1; num_2 = sum(num_2);
    num_3(seg_vol==3) = 1; num_3 = sum(num_3);
    
    % assign numbers to appropriate element of data matrix
    % NB multiply result by volume of each voxel here
    % put CSF label in 2nd column
    data_matrix(image_num+1,2) = num_1*vox_vol;
    % put GM label in 3rd column
    data_matrix(image_num+1,3) = num_2*vox_vol;
    % put WM label in 4th column
    data_matrix(image_num+1,4) = num_3*vox_vol;
end
 
%% Add column of brain volume (i.e. GM + WM only) and scatter plot
data_matrix(:,5) = data_matrix(:,3) + data_matrix(:,4);
%plot brain volume versus age
figure('units','normalized','outerposition',[0 0 1 1])
X_data = data_matrix(:,1);
Y_data = data_matrix(:,5);
scatter(X_data,Y_data,'+','r','LineWidth',LW)
xlabel('Age (years)','FontSize',fs-1,'FontWeight','bold')
ylabel('Brain Volume (cm^3)','FontSize',fs-1,'FontWeight','bold')
grid minor
hold on
% calculate linear regression coefficients using linearFit function
[m, c, r, p] = linearStatAnalysis(X_data,Y_data);
% use this to calculate and plot a line of best fit 
Y_fit = m.*X_data + c;
plot(X_data,Y_fit,'b','LineWidth',LW)
hold off
ax = gca;
ax.FontSize = fs-3;
xlim([29 85])
% insert r and p value into title of plot
string_1 = ['r = ',num2str(r,'%.2f'),', '];
string_2 = ['p-value = ',num2str(p,'%.3f')];
title([string_1, string_2],'FontSize',fs-1,'FontWeight','bold')
saveas(gcf,'Step_3_Images/brain_volumes.m')
saveas(gcf,'Step_3_Images/brain_volumes.jpeg')

%% add column of GM/WM ratio and scatter plot
data_matrix(:,6) = data_matrix(:,3) ./ data_matrix(:,4);
%plot brain GM/WM ratio versus age
figure('units','normalized','outerposition',[0 0 1 1])
Y_data = data_matrix(:,6);
scatter(X_data,Y_data,'+','r','LineWidth',LW)
hold on
% calculate linear regression coefficients using linearFit function
[m, c, r, p] = linearStatAnalysis(X_data,Y_data);
% use this to calculate and plot a line of best fit 
Y_fit = m.*X_data + c;
plot(X_data,Y_fit,'b','LineWidth',LW)
hold off
xlabel('Age (years)','FontSize',fs-1,'FontWeight','bold')
ylabel('GM/WM Ratio','FontSize',fs-1,'FontWeight','bold')
grid minor
ax = gca;
ax.FontSize = fs-3;
xlim([29 85])
% insert r and p value into title of plot
string_1 = ['r = ',num2str(r,'%.2f'),', '];
string_2 = ['p-value = ',num2str(p,'%.3f')];
title([string_1, string_2],'FontSize',fs-1,'FontWeight','bold')
saveas(gcf,'Step_3_Images/GM_WM.m')
saveas(gcf,'Step_3_Images/GM_WM.jpeg')

%% repeat for normalised data
% compute normalised total brain volume - i.e. divide TBV by TIV ( GM + WM + CSF)
data_matrix(:,7) = data_matrix(:,5) ./ ...
    (data_matrix(:,2) + data_matrix(:,3) + data_matrix(:,4));
% plot normalised brain volume versus age
figure('units','normalized','outerposition',[0 0 1 1])
Y_data = data_matrix(:,7);
scatter(X_data,Y_data,'+','r','LineWidth',LW)
xlabel('Age (years)','FontSize',fs-1,'FontWeight','bold')
ylabel('Brain Volume Normalised by TIV','FontSize',fs-1,'FontWeight','bold')
grid minor
hold on
% calculate linear regression coefficients using linearFit function
[m, c, r, p] = linearStatAnalysis(X_data,Y_data);
% use this to calculate and plot a line of best fit 
Y_fit = m.*X_data + c;
plot(X_data,Y_fit,'b','LineWidth',LW)
hold off
ax = gca;
ax.FontSize = fs-3;
xlim([29 85])
% insert rsq, r and p value into title of plot
string_1 = ['r = ',num2str(r,'%.2f'),', '];
string_2 = ['p-value = ',num2str(p,'%.3f')];
title([string_1, string_2],'FontSize',fs-1,'FontWeight','bold')
saveas(gcf,'Step_3_Images/normalised_brain_volumes.m')
saveas(gcf,'Step_3_Images/normalised_brain_volumes.jpeg')