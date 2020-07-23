% Student Number = 14062340
% Step 2

clc                 
close all 
clearvars

% to prevent MATLAB crashing due to low-level graphics error
opengl('save', 'software')

%To set line widths and font size for plots
fs=20;
LW=1.5;

% save 2D images - on or off?
% save_2D = 'on';
save_2D = 'off';
% save 3D volumes (categorical segmentation only) - on or off?
% save_3D = 'on';
save_3D = 'off';

%%  Allocate mode for GMMEM
% mode = 'none'; % no priors, no MRF, no bias field
% mode = 'prior'; % use priors only
% mode = 'MRF'; % use priors and MRF
mode = 'bias'; % use priors, MRF and bias field

% define maximum number of iterations
max_it = 30;
% set convergence tolerance
tol = 1e-5;
% number of classes to produce during segmentation
numclass = 4;
% enter template number that was used to optimise segmentation parameters (0:9)
template = 2;
% allocate MRF beta value
MRF_beta = 0.3; % acquired from GMM_EM_Optimise.m
% allocate bias field order term
bias_order = 3; % acquired from GMM_EM_Optimise.m

%% Loop through all 20 images to segment

for image_num = 0:19

    % load volume to segment and initialise variables
    vol_data = load_untouch_nii(['Step_2_Images/img_',num2str(image_num),'.nii.gz']);
    % intialise holder for segmented volume
    seg_data = vol_data;
    vol_data = double(vol_data.img);
    [dim1, dim2, dim3] = size(vol_data);
    
    % reshape into 1D vector to speed up implementation
    vol_data_vector = reshape(vol_data,[],1);
    dim4 = length(vol_data_vector);
    
    % initialise mean and variances to be random if no priors are used
    if strcmp(mode,'none')
        mu  = rand(numclass,1)*max(vol_data_vector(:));
        var = (rand(numclass,1)*10)+200;
    else
        % load priors (four TPMs) here
        Other_Prior = load_untouch_nii(['Step_2_Images/TPM_0_image_',...
            num2str(image_num),'_step_2.nii']);
        CSF_Prior = load_untouch_nii(['Step_2_Images/TPM_1_image_',...
            num2str(image_num),'_step_2.nii']);
        GM_Prior = load_untouch_nii(['Step_2_Images/TPM_2_image_',...
            num2str(image_num),'_step_2.nii']);
        WM_Prior = load_untouch_nii(['Step_2_Images/TPM_3_image_',...
            num2str(image_num),'_step_2.nii']);
        Other_Prior = double(Other_Prior.img);
        CSF_Prior = double(CSF_Prior.img);
        GM_Prior = double(GM_Prior.img);
        WM_Prior = double(WM_Prior.img);
    end
    
    % allocate space for the posteriors
    classProb = zeros(dim4, numclass);
    classProbSum = zeros(dim4,1);
    if strcmp(mode,'MRF') || strcmp(mode,'bias')
        % initialise MRF
        MRF = ones(dim4, numclass);
    end
    if  strcmp(mode,'bias')
        % log transform the data (add eps to ensure log(0) does not occur)
        vol_data_vector = log(vol_data_vector + eps);
        % initialise bias field parameters
        BF = zeros(dim4,1); % initialise to all zeros
        w_ik = zeros(dim4,numclass);
        w_i = zeros(dim4,1);
        y_bar_numerator = zeros(dim4,numclass);
        y_bar = zeros(dim4,1);
        % calculate number of polynomials in a 3D basis function 
        J = ((bias_order+1)/1)*((bias_order+2)/2)*((bias_order+3)/3);
        % Calculate A matrix of basis polynomials. Only do this once per volume
        % per bias order as this is very memory intensive.
        A = getMatrixA(dim1,dim2,dim3,dim4,bias_order); 
    end
    
    % allocate space for the priors
    classPrior = ones(dim4, numclass);

    % show priors if used
    if strcmp(mode,'none') == false
        figure('name',['Image - ',num2str(image_num)],'units','normalized',...
            'outerposition',[0 0 1 1])
        for i = 1:numclass
            if i == 1
                subplot(2,2,i)
                image = squeeze(GM_Prior(:,83,:));
                imagesc(image)
                title('GM Prior','FontSize',fs+1,'FontWeight','bold');
                colormap gray
                axis off
                daspect([1 1 1])
            elseif i == 2
                subplot(2,2,i)
                image = squeeze(WM_Prior(:,83,:));
                imagesc(image)
                title('WM Prior','FontSize',fs+1,'FontWeight','bold');
                colormap gray
                axis off
                daspect([1 1 1])
            elseif i == 3
                subplot(2,2,i)            
                image = squeeze(CSF_Prior(:,83,:));
                imagesc(image)           
                title('CSF Prior','FontSize',fs+1,'FontWeight','bold');
                colormap gray
                axis off
                daspect([1 1 1])
            elseif i == 4
                subplot(2,2,i)           
                image = squeeze(Other_Prior(:,83,:));
                imagesc(image)           
                title('Non-Brain Prior','FontSize',fs+1,'FontWeight','bold');      
                colormap gray
                axis off
                daspect([1 1 1])       
            end
           
        end
        pause(0.1) % give time for figure to display

        % reshape priors to 1D
        GM_Prior = reshape(GM_Prior,[],1);
        CSF_Prior = reshape(CSF_Prior,[],1);
        WM_Prior = reshape(WM_Prior,[],1);
        Other_Prior = reshape(Other_Prior,[],1);

        % allocate priors
        classPrior(:, 1) = GM_Prior;
        classPrior(:, 3) = CSF_Prior;
        classPrior(:, 2) = WM_Prior;
        classPrior(:, 4) = Other_Prior;
        
        % initialiase probability and parameter estimates using priors and image data
        for classIndex = 1:numclass
            p_ik = classPrior(:, classIndex);
            % where p_ik is the current probabiity estimate that pixel i belongs to class k
            mu(classIndex) = sum(p_ik.*vol_data_vector) / sum(p_ik);
            var(classIndex) = sum((p_ik .* ((vol_data_vector-mu(classIndex)).^2))) / sum(p_ik);
        end
    end

    % Initialise some more variables
    logLik = -1e9;
    oldLogLik = -1e9;
    iteration = 0;
    didNotConverge = 1;
    
    % Define iterative process in while loop
    % initialise vector to hold convergence criteria in
    convergence_criteria = zeros(max_it,1);

    % Run iterations 
    tic
    while didNotConverge == 1
        iteration = iteration + 1;
        disp(['Running iteration ',num2str(iteration),' ...']);

        % Calculate expectation for all voxels and classes - i.e. update the expectation
        % step    
        classProbSum(:) = 0;
        for classIndex = 1:numclass 
            % calculate Gaussian PDF for this class with appropriate mean and
            % variance
            if strcmp(mode,'bias')
                gauss_PDF = (1 / sqrt(2*pi*var(classIndex))) ...
                    * exp( -0.5 * ((vol_data_vector - mu(classIndex) - BF).^2) ...
                    / var(classIndex) );
            else
                gauss_PDF = (1 / sqrt(2*pi*var(classIndex))) ...
                    * exp( -0.5 * ((vol_data_vector-mu(classIndex)).^2) / var(classIndex) );
            end
            % calculate overall probability density for full image
            if     strcmp(mode,'none')
                classProb(:, classIndex) = (gauss_PDF + eps); 
            elseif strcmp(mode,'prior')
                classProb(:, classIndex) = (gauss_PDF + eps) .* classPrior(:, classIndex);
            elseif strcmp(mode,'MRF') || strcmp(mode,'bias')
                classProb(:, classIndex) = (gauss_PDF + eps) .* classPrior(:, classIndex) ...
                    .* MRF(:, classIndex);
            end

            % sum all four class probabilities together
            classProbSum(:) = classProbSum(:) + classProb(:, classIndex); 

        end

        % Normalise the posterior class probability
        for classIndex = 1:numclass 
            classProb(:, classIndex) = classProb(:, classIndex) ./ classProbSum(:);
        end

        % Maximization - update mean and and variance of the class
        w_i(:) = 0;
        y_bar(:) = 0;
        for classIndex = 1:numclass 
            p_ik = classProb(:, classIndex);
            if strcmp(mode,'bias')
                mu(classIndex) = sum(p_ik .* (vol_data_vector - BF)) / sum(p_ik);
                var(classIndex) = sum(p_ik.*(vol_data_vector - mu(classIndex) - BF).^2) ...
                    / sum(p_ik);
                % then also update other bias field parameters
                w_ik(:,classIndex) = p_ik ./ var(classIndex);
                % sum all four class probabilities together
                w_i(:) = w_i(:) + w_ik(:, classIndex);
                % calculate y_bar
                y_bar_numerator(:,classIndex) = w_ik(:,classIndex) .* mu(classIndex);
                y_bar(:) = y_bar(:) + y_bar_numerator(:,classIndex);
            else
                mu(classIndex) = sum(p_ik.*vol_data_vector) / sum(p_ik);
                var(classIndex) = sum(p_ik.*(vol_data_vector-mu(classIndex)).^2) ...
                    / sum(p_ik);
            end
            if strcmp(mode,'MRF') || strcmp(mode,'bias')
                % update MRF field too, first reshape to four 3D matrices
                classProb = reshape(classProb,dim1,dim2,dim3,numclass);
                % Then call UMRF function
                field = reshape(UMRF(classProb,classIndex),[],1);
                MRF(:,classIndex) = exp( -MRF_beta .* field );
                % Then reshape back to four 1D vectors
                classProb = reshape(classProb,dim4,numclass);
                disp(['    MRF calculation ',num2str(classIndex),'/4 completed']);
            end
        end

        % update bias field parameters too
        if strcmp(mode,'bias')
            % calculate y_bar
            y_bar = y_bar./w_i;
            % calculate W (i.e. put w_i on the main diagonal of sparse matrix W)
            W = spdiags(w_i,0,dim4,dim4);
            % calculate vector r
            r = vol_data_vector - y_bar;
            % calculate c
            c = (A'*W*A)\(A'*W*r);
            % calculate BF by multiplying c by basis function and summing
            % across the number of basis functions
            BF(:) = 0;
            for j = 1:J
                BF_j = c(j) .* A(:,j);
                BF = BF + BF_j;
            end
            disp('    Bias Field updated');
        end
        % Convergence criteria - update the cost function parameters
        oldLogLik = logLik;
        logLik = sum(log(classProbSum));
        convergence_criteria(iteration) = abs((logLik/oldLogLik) -1);
        if convergence_criteria(iteration) <=  tol || iteration >= max_it
            didNotConverge = 0;
            time = toc;
            disp(['Converged after ',num2str(iteration),' iterations in ', ...
                num2str(time),' seconds']);
        end    
    end

    % reshape classProb into 4D after convergence and exponentially t
    classProb = (reshape(classProb,dim1,dim2,dim3,numclass));
    % check that the probabilities/segmentations at each voxel sums to one 
    % across all four classes
    total = sum(classProb,4);
    check = mean(mean(mean(total))) %#ok<NOPTS>

    % Plot results - show probabilisitic segmented images
    figure('name',['Image - ',num2str(image_num)],'units',...
        'normalized','outerposition',[0 0 1 1])
    % show slice from volume to segment
    subplot(2,3,1)
    imagesc(squeeze(vol_data(:,83,:)))
    colormap gray
    title('Volume to Segment','FontSize',fs+1,'FontWeight','bold');      
    axis off
    daspect([1 1 1])
    for i = 2:numclass+1
        subplot(2,3,i)
        image = squeeze(classProb(:,83,:,i-1));
        imagesc(image)
        colormap gray
        axis off
        if i == 2
            if strcmp(mode,'none')
                title('Segmentation #1','FontSize',fs+1,'FontWeight','bold');
            else
                title('GM Segmentation','FontSize',fs+1,'FontWeight','bold');
            end
        elseif i == 3
            if strcmp(mode,'none')
                title('Segmentation #2','FontSize',fs+1,'FontWeight','bold');
            else
                title('WM Segmentation','FontSize',fs+1,'FontWeight','bold');
            end
        elseif i == 4
            if strcmp(mode,'none')
                title('Segmentation #3','FontSize',fs+1,'FontWeight','bold');
            else        
                title('CSF Segmentation','FontSize',fs+1,'FontWeight','bold');
            end
        elseif i == 5
            if strcmp(mode,'none')
                title('Segmentation #4','FontSize',fs+1,'FontWeight','bold');
            else               
                title('Non-Brain Segmentation','FontSize',fs+1,'FontWeight','bold');   
            end
        end
        daspect([1 1 1])
    end
    subplot(2,3,6)
    % show convergence behaviour
    semilogy(1:iteration,convergence_criteria(1:iteration),'LineWidth',LW)
    grid minor
    ax = gca;
    ax.FontSize = fs-3;
    xlabel('Iteration Number','FontSize',fs-1,'FontWeight','bold')
    ylabel('$\frac{\partial \mathrm{log} L(\Phi)}{\partial(\Phi)}$','Interpreter',...
        'LaTex','FontSize',fs-1,'FontWeight','bold')
    title('Cost Function','FontSize',fs+1,'FontWeight','bold'); 
    xlim([1 iteration])
    ylim([max(tol,min(convergence_criteria)) convergence_criteria(1)])
    
    if strcmp(save_2D,'on')
        if strcmp(mode,'none')
            saveas(gcf,['Step_2_Images/GMMEM_output_none_image_',num2str(image_num),'.m'])
            saveas(gcf,['Step_2_Images/GMMEM_output_none_image_',num2str(image_num),'.jpeg']) 
        elseif strcmp(mode,'prior')
            saveas(gcf,['Step_2_Images/GMMEM_output_prior_image_',num2str(image_num),'.m'])
            saveas(gcf,['Step_2_Images/GMMEM_output_prior_image_',num2str(image_num),'.jpeg'])    
        elseif strcmp(mode,'MRF')
            saveas(gcf,['Step_2_Images/GMMEM_output_MRF_image_',num2str(image_num),'_beta_',...
                num2str(MRF_beta*10),'.m'])
            saveas(gcf,['Step_2_Images/GMMEM_output_MRF_image_',num2str(image_num),'_beta_',...
                num2str(MRF_beta*10),'.jpeg'])  
        elseif strcmp(mode,'bias')
            saveas(gcf,['Step_2_Images/GMMEM_output_bias_image_', ...
                num2str(image_num),'_beta_',num2str(MRF_beta*10),'_bias_',...
                num2str(bias_order),'.m'])
            saveas(gcf,['Step_2_Images/GMMEM_output_bias_image_', ...
                num2str(image_num),'_beta_',num2str(MRF_beta*10),'_bias_',...
                num2str(bias_order),'.jpeg'])  
        end
    end
    
    % Convert 4D probabilistic data to 4D categorical data by taking MAP label
    % at each voxel (i.e. find the tissue label with the highest probability per voxel)
    
    % Reshape class probabilities back to four 1D vectors
    classProb = reshape(classProb,dim4,numclass);
    % This is now a 2D matrix of four columns (corresponding to number of
    % classes) and number of rows corresponding to number of pixels.
    % Take the maximum value along each row and keep a record of original
    % index in vector I.
    [~,I] = max(classProb,[],2);
    % I is now a vector of value either 1 (GM), 2(WM), 3(CSF) or 4(Non-brain). 
    % Therefore convert 4 values to 0 values
    labels = zeros(dim4,1);
    labels(I==1) = 2; labels(I==2) = 3; labels(I==3) = 1; labels(I==4) = 0;
    % reshape back to a 3D volume
    labels = reshape(labels,dim1,dim2,dim3);

    % show a slice of this labelled image
    figure('name',['Image - ',num2str(image_num)],'units',...
        'normalized','outerposition',[0 0 1 1])
    image = squeeze(labels(:,83,:));
    imagesc(image)
    colormap gray
    axis off
    daspect([1 1 1])
    title('Categorical Segmentation','FontSize',fs+1,'FontWeight','bold')

    % Also show, if present, bias field, residue image, voxel weights, and
    % estimated bias corrected data
    if strcmp(mode,'bias')
        figure('name',['Image - ',num2str(image_num)],'units',...
            'normalized','outerposition',[0 0 1 1])
        subplot(2,2,1)
        BF_3D = reshape(BF,dim1,dim2,dim3);
        image = squeeze(BF_3D(:,83,:));
        imagesc(image)
        title('Bias Field','FontSize',fs+1,'FontWeight','bold')
        colormap gray
        axis off
        daspect([1 1 1])
        
        subplot(2,2,2)
        w_i_3D = reshape(w_i,dim1,dim2,dim3);
        image = squeeze(w_i_3D(:,83,:));
        imagesc(image)
        title('Voxel Weights','FontSize',fs+1,'FontWeight','bold')
        colormap gray
        axis off
        daspect([1 1 1])
        
        subplot(2,2,3)
        y_bar_3D = reshape(y_bar,dim1,dim2,dim3);
        image = squeeze(y_bar_3D(:,83,:));
        imagesc(image)
        title('Bias Corrected Data','FontSize',fs+1,'FontWeight','bold')
        colormap gray
        axis off
        daspect([1 1 1])
        
        subplot(2,2,4)
        r_3D = reshape(r,dim1,dim2,dim3);
        image = squeeze(r_3D(:,83,:));
        imagesc(image)
        title('Residue','FontSize',fs+1,'FontWeight','bold')
        colormap gray
        axis off
        daspect([1 1 1])           
        
    end

    if strcmp(save_2D,'on')
        saveas(gcf,['Step_2_Images/BF_image_', num2str(image_num),'_beta_', ...
            num2str(MRF_beta*10),'_bias_',num2str(bias_order),'.m'])
        saveas(gcf,['Step_2_Images/BF_image_', num2str(image_num),'_beta_', ...
            num2str(MRF_beta*10),'_bias_',num2str(bias_order),'.jpeg'])         
    end
            
    if strcmp(save_3D,'on')
        % Save categorical segmentation volume as .nii file
        seg_data.img = labels;
        save_untouch_nii(seg_data,['Segmented_Volumes/categorical_segmentation_image_',...
            num2str(image_num),'_beta_',num2str(MRF_beta*10),'_bias_',...
            num2str(bias_order),'_template_',num2str(template),'.nii'])
    end
    
    pause(0.1) % give time for figure to display before moving to next segmentation
            
end

%% Display a slice of each of 20 segmented volumes in 1 plot

figure('units','normalized','outerposition',[0 0 1 1])
for image_num = 0:19
    subplot(4,5,image_num+1)
    image = load_untouch_nii(['Segmented_Volumes/categorical_segmentation_image_',...
            num2str(image_num),'_beta_',num2str(MRF_beta*10),'_bias_',...
            num2str(bias_order),'_template_',num2str(template),'.nii']);
    imagesc(squeeze(image.img(:,83,:)))
    colormap gray
    daspect([1 1 1])
    axis off
    title(image_num)
end
saveas(gcf,['Step_2_Images/categorical_segmentations_template',num2str(template),'.m'])
saveas(gcf,['Step_2_Images/categorical_segmentations_template',num2str(template),'.jpeg'])