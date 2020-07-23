% Student Number = 14062340
% Used by script GroupWiseReg.m - Step 1

function CalcMeanImage(it_num)
% Define function to calculate mean warped image at step it_num of
% groupwise registration process.
% This mean image is then saved appropriately for later reference.


operation = 'reg_average';

image_0 = [' -avg Step_1_Images/warped_image_0_step_',num2str(it_num),'.nii'];
image_1 = [' Step_1_Images/warped_image_1_step_',num2str(it_num),'.nii'];
image_2 = [' Step_1_Images/warped_image_2_step_',num2str(it_num),'.nii'];
image_3 = [' Step_1_Images/warped_image_3_step_',num2str(it_num),'.nii'];
image_4 = [' -avg Step_1_Images/warped_image_4_step_',num2str(it_num),'.nii'];
image_5 = [' Step_1_Images/warped_image_5_step_',num2str(it_num),'.nii'];
image_6 = [' Step_1_Images/warped_image_6_step_',num2str(it_num),'.nii'];
image_7 = [' Step_1_Images/warped_image_7_step_',num2str(it_num),'.nii'];
image_8 = [' -avg Step_1_Images/warped_image_8_step_',num2str(it_num),'.nii'];
image_9 = [' Step_1_Images/warped_image_9_step_',num2str(it_num),'.nii'];

%take the average of images 0,1,2 and 3 --> av_image_1
out_1 = ' Step_1_Images/av_image_1.nii';
command = [operation, out_1, image_0, image_1, image_2, image_3];
dos(command)

%take the average of images 4,5,6 and 7 --> av_image_2
out_2 = ' Step_1_Images/av_image_2.nii';
command = [operation, out_2 image_4, image_5, image_6, image_7];
dos(command)

%take the average of images 8 and 9  --> av_image_3
out_3 = ' Step_1_Images/av_image_3.nii';
command = [operation, out_3, image_8, image_9];
dos(command)

%take the average of av_image_1, av_image_2 and av_image_3 -->
%average_image_step_it_num
out_4 = [' Step_1_Images/average_image_step_',num2str(it_num),'.nii -avg'];
command = [operation, out_4, out_1, out_2, out_3];
dos(command)
end
