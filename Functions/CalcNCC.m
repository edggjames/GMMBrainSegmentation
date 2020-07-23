% Student Number = 14062340
% Used by script TPM_Prop.m - Step 2
    
function NCC = CalcNCC(image_num,it_num)
% Function to calculate NCC value at the end of each registration step
% between appropriate reference image and warped image.


operation = 'reg_measure.exe';
out = ' -out NCCresult.txt';
% define reference image
ref = [' -ref Step_2_Images/img_',num2str(image_num),'.nii.gz'];
% define floating image
flo = [' -flo Step_2_Images/warped_image_seg_prop_',num2str(image_num),'_step_',num2str(it_num),'.nii -ncc'];
command = [operation, ref, flo, out];
dos(command)
% read in result and return (and display in command window)
NCC = csvread('NCCresult.txt') %#ok<NOPRT>
end

