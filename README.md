# GMMBrainSegmentation

This repository contains the code that I have written to assess the correlation   
between age and brain volumetric measurements in 20 MRI images.

Step 1
------

Implementation of a groupwise registration pipeline to create a groupwise space of 10  
previously segmented images. This registration task was implemented using NiftyReg  
tools (see http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_documentation and  
http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install). This groupwise space  
was then used to generate mean tissue probability maps for non brain tissue,  
cerebrospinal fluid (CSF), white matter (WM) and grey matter (GM) for this population.

Step 2
------

The tissue probability maps were then propagated into the space of 20 unsegmented  
images, which were used as a priori information for their segmentation. A Gaussian  
Mixture Model (GMM) was implemented to segment these images, optimised through an   
Expectation-Maximisation scheme. A Markov random field was embedded into the  
segmentation framework to introduce a spatial smoothness in the label estimation  
process. See "Automated Model-Based Tissue Classification of MR Images of the Brain"  
in the 'Papers' folder.

As MRI acquisition usually suffers from magnetic field intensity non-uniformity,  
the robustness of the GMM framework was improved by adding a bias field correction  
component to the probabilistic model. See "Automated Model-Based Bias Field Correction  
of MR Images of the Brain" in the 'Papers' folder.

Optimisation of segmentation implementation parameters was achieved by using a DICE  
image similarity metric to compare test segmentations to a ground truth segmentation.

Step 3
------

Statistical analysis was then performed to assess the relationship between brain  
volume (having normalised by total intracranial volume) and age, as well as the  
relationship between GM/WM and age in the 20 segmented MRI images.  
