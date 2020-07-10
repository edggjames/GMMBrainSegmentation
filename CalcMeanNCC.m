% Student Number = 14062340
% Used by script GroupWiseReg.m - Step 1

function NCCvalue = CalcMeanNCC(it_num, image_num)
% Function to calculate NCC value at the end of each registration step
% between appropriate reference image and warped image.
% Used by script GroupWiseReg.m

operation = 'reg_measure.exe';
out = ' -out NCCresult.txt';
if it_num == 1
    % define reference image
    ref = ' -ref Step_1_Images/template_0_img.nii.gz';
    % define floating image
    flo = [' -flo Step_1_Images/warped_image_',num2str(image_num),'_step_',num2str(it_num),'.nii -ncc'];
else
    % define reference image
    ref = [' -ref Step_1_Images/average_image_step_',num2str(it_num-1),'.nii'];
    % define warped image and ncc mehtod
    flo = [' -flo Step_1_Images/warped_image_',num2str(image_num),'_step_',num2str(it_num),'.nii -ncc'];
end
command = [operation, ref, flo, out];
dos(command)
% read in result and return (and display in command window)
NCCvalue = csvread('NCCresult.txt') %#ok<NOPRT>
end

