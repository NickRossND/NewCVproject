Process beginning to end:
Extract data from previous student's files using justheatmaps.m code
Perform data augmentation using CVproject_Data_NoiseBrightnessContrast.py
Split data/files into tumor and no tumor (this was done manually according to 'measurement transparencies' folder)
Tumor and no tumor data placed into two folders under 'Test' folder 
Run NeuralNetwork.py code for training and validation. Testing functionality also included

############################################################################
Code Descriptions:
CV...NoiseBrightness...- USED FOR ALL IMAGE AUGMENTATION. code for adding noise, changing brightness, and altering contrast to raw data 			 obtained from imaging for MULTIPLE files in a directory. 
CV...NoiseBright...woutForLoop - code for adding noise, changing brightness, and altering contrast to raw data obtained from 
				 imaging for ONE file
CV...IndividualLoops - same as "CV...NoiseBrightness..." but a single image alteration is performed for every patient prior 
		       to making the next alteration as opposed to making all alterations to one file and then moving to the 
		       next. (i.e., add noise to every file, then add brightness for each, and so on)
NeuralNet - runs the neural network code

############################################################################
Code for Unit Testing:
AddFolders  - test code for adding new folders to directories and subdirectories
BilinearInt - test code for interpolating data after normalization
Convolution - test code for performing 2D convolutions for sharpening and blurring
CV...ImageChanges - tet code for adding noise, changing brightness, and altering contrast from images generated
RemoveFiles - test code for removing unwanted files
RenameFiles - test code for renaming unwanted files
heatmaps and   - code to extract data from old files created by previous student. This is how I extracted the original matrices that
justheatmaps.m 	 were then reduced or enlarged to 6x6 matrices

############################################################################
Patient Data File Naming Convention: 
BreastSide-Chromophore-ChangesMade%%added.jpg
Example: Right-TOI-Noise10.jpg (represents 10% noise added to dataset Right-TOI)
Example: Right-THC-Noise7-Brt10-Cnt60.jpg (represents 7% noise with increase of brightness by .10 and contrast alteration of 
	 60% added to dataset Right-THC)

############################################################################
Miscellaneous Files/Folders:
CVProject_FirstNN_Results - Results from first neural network mdoel with unbalanced datasets for training and testing
CVProject_SecondNN_Results- Results from second neural network mdoel with balanced datasets for training and testing
CVProject_ThirdNN_Results - Results from third neural network mdoel with balanced subject-disjointed datasets for training and testing
CV_Summary     - FIRST ASSIGNMENT DOCUMENT giving overview of project
DataDescript   - txt file discussing what the project will be and what the data is referencing 
DataIssues     - issues experienced when processing the data for this project and any resolutions
DK1_TestFolder - used as a test folder when running code to avoid altering original data. Data is from 
		 "PatientPlots" > "DKU01_140502" 
PatientPlots   - contains all the raw data from breast cancer imaging sessions
Test 	       - USED FOR NEURAL NETWORK CODE. Contains all augmented and non-augmented data for tumor and no tumor images/files 
############################################################################
Images:
Accuracy - model accuracy images
	Naming: Accuracy_DatasetUsed_DenseLyrActivationFunctionLossFunction.png
ConfusionMatrix- Image of the confusion matrix saved from after running the neural network
	Naming: ConfusionMatrix_DatasetUsed_DenseLyrActivationFunctionLossFunction.png
Loss- Image of the the loss performance saved from after running the neural network
	Naming: Loss_DatasetUsed_DenseLyrActivationFunctionLossFunction.png
ROC - receiver operating characteristic (ROC) curve for datasets used for testing
	Naming: ROC_DatasetUsed_DenseLyrActivationFunctionLossFunction.png
