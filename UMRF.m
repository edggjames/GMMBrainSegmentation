% Student Number = 14062340
% Used by scripts GMM_EM.m and GMM_EM_Optimise.m - Step 2

function umrf = UMRF(p_ik,k)
% Function to calculate UMRF for class k given class probabilities p_ik

% Initialise UMRF with zeros matrix the same size as p_ik
umrf = zeros(size(p_ik,1),size(p_ik,2),size(p_ik,3),size(p_ik,4));
% Then initialise a zeros 3x3x3 kernel with the 6 nearest neighbours
% of the centre voxel set to 1
kernel = zeros(3,3,3);
% First plane of kernel
kernel(2,2,1) = 1;
% Second plane of kernel
kernel(1,2,2) = 1; kernel(3,2,2) = 1; kernel(2,1,2) = 1; kernel(2,3,2) = 1;
% Third plane of kernel
kernel(2,2,3) = 1;
% convolve this kernel with the class probabilities of appropriate class index
for j = 1: size(p_ik, 4)
    if j ~= k
        % this skips the need for an explicit G energy functional and 
        % also speeds up the implementation by decreasing number of convolutions
        % performed from 4 to 3 each time the function is called.
        umrf(:,:,:,j) = convn(p_ik(:,:,:,j),kernel,'same');
        % specifying same here returns the central part of the convolution
        % that is the same size as p_ik.
    end
end
% take the same of the umrf across the 4th dimension (i.e. across all
% classes)
umrf = sum(umrf,4);
end