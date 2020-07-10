% Student Number = 14062340
% Step 2

clc                 
close all 
clearvars

%To set line widths and font size for plots
fs=20;
LW=1.5;

% define maximum number of iterations
max_it = 30;
% set convergence tolerance
tol = 1e-5;
% number of classes to produce during segmentation
numclass = 4;
% Select one already segmented image to use as template (from [0:9])
template = 2;

%% load ground truth segmentation, get size and reshape into 1D
GT_seg = load_untouch_nii(['Step_1_Images/template_',num2str(template),'_seg.nii.gz']);
GT_seg_data = GT_seg.img; [dim1, dim2, dim3] = size(GT_seg_data);
GT_seg_data = reshape(GT_seg_data,[],1);

% calculate number of pixels in ground truth segmentation
num_vox = length(GT_seg_data); dim4 = num_vox;

% count number of zeros in ground truth segmentation (i.e. number of non-brain labels) 
GT_0 = zeros(num_vox,1); GT_0(GT_seg_data==0) = 1; GT_num_0 = sum(GT_0);

% count number of ones in ground truth segmentation (i.e. number of CSF labels)
GT_1 = zeros(num_vox,1); GT_1(GT_seg_data==1) = 1; GT_num_1 = sum(GT_1);

% count number of twos in ground truth segmentation (i.e. number of GM labels) 
GT_2 = zeros(num_vox,1); GT_2(GT_seg_data==2) = 1; GT_num_2 = sum(GT_2);

% count number of threes in ground truth segmentation (i.e. number of WM labels) 
GT_3 = zeros(num_vox,1); GT_3(GT_seg_data==3) = 1; GT_num_3 = sum(GT_3);

% count number of brain volume labels and weights accordingly (i.e. exclude
% non-brain labels)
num_vox_brain = sum(GT_1 + GT_2 + GT_3);
weight_1_brain = sum(GT_1/num_vox_brain);
weight_2_brain = sum(GT_2/num_vox_brain);
weight_3_brain = sum(GT_3/num_vox_brain);

%% Produce and save priors to use for test segmentation
% 1) affine registration step - define float mask
fmask = ' -fmask Step_2_Images/binarised_mask.nii';
% define operation to perform
operation = 'reg_aladin'; 
% define floating image
flo = ' -flo Step_1_Images/average_image_step_12.nii';
% use new image as reference
ref = [' -ref Step_1_Images/template_',num2str(template),'_img.nii.gz'];
% define filename of where to save the output transformation
aff = [' -aff Optimisation/transform_TPM_prop_',num2str(template),'_aff_step.txt'];
% define filename of where to save output warped image
res = [' -res Optimisation/warped_image_TPM_prop_',num2str(template),'_step_1.nii'];
command = [operation, ref, flo, aff, res, fmask];
dos(command)

% 2) Perform non-linear registration - define operation
operation = 'reg_f3d -pad 0';
% define filename of where to retrieve the input transformation
aff = [' -aff Optimisation/transform_TPM_prop_',num2str(template),'_aff_step.txt'];
% define filename of where to save output warped image
res = [' -res Optimisation/warped_image_TPM_prop_',num2str(template),'_step_2.nii'];
% define filename of where to write the output transformation
cpp = [' -cpp Optimisation/transform_TPM_prop_',num2str(template),'_nrr_step.nii'];
command = [operation, ref, flo, res, aff, cpp, fmask];
dos(command)

% 3) The non-linear transformation parametrisation can now be used to propagate 
% the groupwise TPMS (x4) into the space of each of the image to be segmented. 
% Since the values are probabilities, use a first order linear interpolation
% scheme. Use 0 padding for non-brain prior, and use 1 padding for all others

% define the file containing the final transformation parameterisation
trans = [' -trans Optimisation/transform_TPM_prop_',num2str(template),'_nrr_step.nii'];
for labels = 0:3
    if labels ~= 0
        operation = 'reg_resample -inter 1 -pad 0';
    elseif labels == 0
        operation = 'reg_resample -inter 1 -pad 1';
    end
    % define filename of where resampled TPM image should be saved
    res = [' -res Optimisation/TPM_',num2str(labels),'_image_',num2str(template),'.nii'];
    % Use relevant TPM label from step 1 as floating image
    flo = [' -flo Step_1_Images/TPM_',num2str(labels),'_step_1.nii'];
    command = [operation, ref, flo, res, trans];
    dos(command)
end

%% Load priors for the segmentation of image

Other_Prior = load_untouch_nii(['Optimisation/TPM_0_image_',num2str(template),'.nii']);
Other_Prior = reshape(Other_Prior.img,[],1);
CSF_Prior = load_untouch_nii(['Optimisation/TPM_1_image_',num2str(template),'.nii']);
CSF_Prior = reshape(CSF_Prior.img,[],1);
GM_Prior = load_untouch_nii(['Optimisation/TPM_2_image_',num2str(template),'.nii']);
GM_Prior = reshape(GM_Prior.img,[],1);
WM_Prior = load_untouch_nii(['Optimisation/TPM_3_image_',num2str(template),'.nii']);
WM_Prior = reshape(WM_Prior.img,[],1);

% initialise and allocate priors
classPrior = ones(dim4, numclass);
classPrior(:, 1) = GM_Prior;
classPrior(:, 2) = WM_Prior;
classPrior(:, 3) = CSF_Prior;
classPrior(:, 4) = Other_Prior;

%% Then segment template image with a given set of beta and bias order values
% using these priors, in order to produce a test categorical segmentation.
% For each pair of beta and bias order values obtain a vector of DICE scores
% by comparing with the ground truth segmentation from above.

% Enter range of parameter values to test here
beta = 0:0.1:2.0;
bias = 0:1:4;

% load volume to segment
vol_data = load_untouch_nii(['Step_1_Images/template_',num2str(template),'_img.nii.gz']);
vol_data = double(vol_data.img);
% reshape into 1D vector to speed up implementation and log transform after
% adding eps (so that there are no -Inf intensity values)
vol_data_vector = log(reshape(vol_data,[],1)+eps);

% Initialise holder for DICE values (one for each of 4 labels and 1 overall = 5)
DICE_values = zeros(length(beta),length(bias),5);
    
%% Calculate DICE score for each pair of parameter values
tic
for k = 1:length(bias)
    
    bias_order = bias(k);
    % Calculate number of polynomials in a 3D basis function of this order
    J = ((bias_order+1)/1)*((bias_order+2)/2)*((bias_order+3)/3);
    % Calculate matrix A of basis polynomials. Only do this once per volume
    % per bias order as this is very memory intensive.
    A = getMatrixA(dim1,dim2,dim3,dim4,bias_order); 
    
    for i = 1:length(beta)
        
        MRF_beta = beta(i);

        disp(['Testing Beta = ',num2str(MRF_beta),' and Bias Order = ', ...
            num2str(bias_order),' ...']);

        % initialiase probability and parameter estimates using priors and image data
        mu = zeros(1,numclass); var = zeros(1,numclass);
        for classIndex = 1:numclass
            p_ik = classPrior(:, classIndex);
            % where p_ik is the current probabiity estimate that pixel i belongs to class k
            mu(classIndex) = sum(p_ik.*vol_data_vector) / sum(p_ik);
            var(classIndex) = sum((p_ik .* ((vol_data_vector-mu(classIndex)).^2))) / sum(p_ik);
        end
        
        % Initialise some more variables
        logLik = -1e9;
        oldLogLik = -1e9;
        iteration = 0;
        didNotConverge = 1;

        % allocate space for the posteriors
        classProb = zeros(dim4, numclass);
        classProbSum = zeros(dim4,1);
        % initialise MRF parameters to all ones for first iteration
        MRF = ones(dim4, numclass);
        % initialise bias field, allocate voxel weights and bias-corrected data
        BF = zeros(dim4,1); % NB initialise to all zeros for first iteration
        w_ik = zeros(dim4,numclass); w_i = zeros(dim4,1);
        y_bar_numerator = zeros(dim4,numclass); y_bar = zeros(dim4,1);

        % Define iterative process in while loop - initialise vector to
        % hold convergence criteria in
        convergence_criteria = zeros(max_it,1);

        % Run iterations 
        while didNotConverge == 1
            iteration = iteration + 1;

            % Calculate expectation for all voxels and classes - i.e. update
            % the expectation step    
            classProbSum(:) = 0;
            for classIndex = 1:numclass 
                % calculate Gaussian PDF for this class with appropriate mean and
                % variance
                gauss_PDF = (1 / sqrt(2*pi*var(classIndex))) ...
                    * exp( -0.5 * ((vol_data_vector - mu(classIndex) - BF).^2) ...
                    / var(classIndex) );
                % calculate overall probability density for full image
                classProb(:, classIndex) = (gauss_PDF + eps) .* classPrior(:, classIndex) ...
                    .* MRF(:, classIndex);
                % sum all four class probabilities together
                classProbSum(:) = classProbSum(:) + classProb(:, classIndex); 
            end

            % Normalise the posterior class probability
            for classIndex = 1:numclass 
                classProb(:, classIndex) = classProb(:, classIndex) ./ classProbSum(:);
            end
            
            w_i(:) = 0;
            y_bar(:) = 0;
            % Maximization - update mean, variance, MRF and BF parameters of each class
            for classIndex = 1:numclass 
                p_ik = classProb(:, classIndex);
                mu(classIndex) = sum(p_ik .* (vol_data_vector - BF)) / sum(p_ik);
                var(classIndex) = sum(p_ik.*(vol_data_vector - mu(classIndex) - BF).^2) ...
                    / sum(p_ik);
                % then also update bias field parameters
                w_ik(:,classIndex) = p_ik ./ var(classIndex);
                % sum all four class probabilities together
                w_i(:) = w_i(:) + w_ik(:, classIndex);
                % calculate y_bar
                y_bar_numerator(:,classIndex) = w_ik(:,classIndex) .* mu(classIndex);
                y_bar(:) = y_bar(:) + y_bar_numerator(:,classIndex);
                % update MRF field too, first reshape to four 3D matrices
                classProb = reshape(classProb,dim1,dim2,dim3,numclass);
                % Then call UMRF function
                field = reshape(UMRF(classProb,classIndex),[],1);
                MRF(:,classIndex) = exp( -MRF_beta .* field );
                % Then reshape back to four 1D vectors
                classProb = reshape(classProb,dim4,numclass);
            end
            
            % Update bias field parameters too - calculate y_bar
            y_bar = y_bar./w_i;
            % calculate W (i.e. put w_i on the main diagonal of sparse matrix W)
            W = spdiags(w_i,0,dim4,dim4);
            % calculate vector r
            r = vol_data_vector - y_bar;
            % calculate column vector c
            c = (A'*W*A)\(A'*W*r);
            % calculate BF by multiplying c by basis function and summing
            % across the number of basis functions
            BF(:) = 0;
            for j = 1:J
                BF_j = c(j) .* A(:,j);
                BF = BF + BF_j;
            end
            
            % Convergence criteria - update the cost function parameters
            oldLogLik = logLik;
            logLik = sum(log(classProbSum));
            convergence_criteria(iteration) = abs((logLik/oldLogLik) -1);
            if convergence_criteria(iteration) <=  tol || iteration >= max_it
                didNotConverge = 0;
            end    
        end

        % reshape classProb into 4D after convergence
        classProb = reshape(classProb,dim1,dim2,dim3,numclass);

        % Convert 4D probabilistic data to 4D categorical data by taking MAP label
        % at each voxel (i.e. find the tissue label with the highest probability per voxel)

        % Reshape class probabilities back to four 1D vectors
        classProb = reshape(classProb,dim4,numclass);
        % This is now a 2D matrix of four columns (corresponding to number of
        % classes) and number of rows corresponding to number of voxels.
        % Take the maximum value along each row and keep a record of original
        % index in vector I.
        [~,I] = max(classProb,[],2);
        % I is now a vector of value either 1 (GM), 2(WM), 3(CSF) or 4(Non-brain). 
        % Therefore convert so compatible for comparison with ground truth
        % segmentation
        labels = zeros(dim4,1);
        labels(I==1) = 2; labels(I==2) = 3; labels(I==3) = 1; labels(I==4) = 0;
        % reshape back to a 3D volume
        labels = reshape(labels,dim1,dim2,dim3);         
        
        % Then to compare with ground truth segmentation reshape labels 
        % (test segmentation) into 1D vector
        labels = reshape(labels,[],1);

        % count number of zeros in test segmentation (i.e. number of non-brain labels) 
        test_0 = zeros(num_vox,1); test_0(labels==0) = 1; test_num_0 = sum(test_0);

        % count number of ones in test segmentation (i.e. number of CSF labels)
        test_1 = zeros(num_vox,1); test_1(labels==1) = 1; test_num_1 = sum(test_1);

        % count number of twos in test segmentation (i.e. number of GM labels) 
        test_2 = zeros(num_vox,1); test_2(labels==2) = 1; test_num_2 = sum(test_2);

        % count number of threes in test segmentation (i.e. number of WM labels) 
        test_3 = zeros(num_vox,1); test_3(labels==3) = 1; test_num_3 = sum(test_3);

        % Assign one to all overlapping voxel values between two volumes
        overlap = zeros(num_vox,1); overlap(GT_seg_data==labels)=1;

        % Calculate DICE score for label 0
        DICE_0 = overlap .* GT_0; DICE_0 = (2 * sum(DICE_0)) / (GT_num_0 + test_num_0);
        % Calculate DICE score for label 1
        DICE_1 = overlap .* GT_1; DICE_1 = (2 * sum(DICE_1)) / (GT_num_1 + test_num_1);
        % Calculate DICE score for label 2
        DICE_2 = overlap .* GT_2; DICE_2 = (2 * sum(DICE_2)) / (GT_num_2 + test_num_2);
        % Calculate DICE score for label 3
        DICE_3 = overlap .* GT_3; DICE_3 = (2 * sum(DICE_3)) / (GT_num_3 + test_num_3);

        DICE_values(i,k,1) = DICE_0;
        DICE_values(i,k,2) = DICE_1;
        DICE_values(i,k,3) = DICE_2;
        DICE_values(i,k,4) = DICE_3;

        % Multiply each DICE score by weighting factor from ground truth segmentation
        % Then add all DICE scores up to achieve final DICE score between 2 images 
        
        % Method (2) - only takes into account brain voxels - unequal weighting
        DICE_overall = (DICE_1 * weight_1_brain) + (DICE_2 * weight_2_brain) + ...
            (DICE_3 * weight_3_brain);
       
        DICE_values(i,k,5) = DICE_overall;

        % This is an overlap metric between each label (e.g. overlap between 
        % GM_GT/GM_Seg, WM_GT/WM_Seg ...) and overall (weighted by relative 
        % abundance of given label in ground truth segmentation) to estimate
        % how good the test segmentation is.
        
        disp('Converged with the following DICE scores:')
        disp(['DICE CSF = ',num2str(DICE_1)])
        disp(['DICE GM = ',num2str(DICE_2)])
        disp(['DICE WM = ',num2str(DICE_3)])
        disp(['DICE Overall = ',num2str(DICE_overall)])
        disp(' ')
        
    end
end
time = toc; time = time/60;

save DICE_Values DICE_values bias beta LW fs time

%% Plot DICE scores

load DICE_values 
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
% to loop through bias field orders
for j = 1:5
    semilogy(beta,squeeze(DICE_values(:,j,2)),'LineWidth',LW)
    hold on
end
hold off
title('CSF Label','FontSize',fs+1,'FontWeight','bold')
xlabel('MRF Beta','FontSize',fs-1,'FontWeight','bold')
ylabel('DICE score','FontSize',fs-1,'FontWeight','bold')
grid minor
ax = gca;
ax.FontSize = fs-3;
legend('0','1','2','3','4')

subplot(2,2,2)
% to loop through bias field orders
for j = 1:5
    semilogy(beta,squeeze(DICE_values(:,j,3)),'LineWidth',LW)
    hold on
end
hold off
title('GM Label','FontSize',fs+1,'FontWeight','bold')
xlabel('MRF Beta','FontSize',fs-1,'FontWeight','bold')
ylabel('DICE score','FontSize',fs-1,'FontWeight','bold')
grid minor
ax = gca;
ax.FontSize = fs-3;
legend('0','1','2','3','4')

subplot(2,2,3)
% to loop through bias field orders
for j = 1:5
    semilogy(beta,squeeze(DICE_values(:,j,4)),'LineWidth',LW)
    hold on
end
hold off
title('WM Label','FontSize',fs+1,'FontWeight','bold')
xlabel('MRF Beta','FontSize',fs-1,'FontWeight','bold')
ylabel('DICE score','FontSize',fs-1,'FontWeight','bold')
grid minor
ax = gca;
ax.FontSize = fs-3;
legend('0','1','2','3','4')
   
subplot(2,2,4)
% to loop through bias field orders
for j = 1:5
    semilogy(beta,squeeze(DICE_values(:,j,5)),'LineWidth',LW)
    hold on
end
hold off
title('Overall','FontSize',fs+1,'FontWeight','bold')
xlabel('MRF Beta','FontSize',fs-1,'FontWeight','bold')
ylabel('DICE score','FontSize',fs-1,'FontWeight','bold')
grid minor
ax = gca;
ax.FontSize = fs-3;
legend('0','1','2','3','4')

saveas(gcf,['Optimisation/DICE_plots_',num2str(template),'.jpeg'])
saveas(gcf,['Optimisation/DICE_plots_',num2str(template),'.m'])