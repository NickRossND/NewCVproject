Code Descriptions:
CV...NoiseBrightness...- code for adding noise, changing brightness, and altering contrast to raw data obtained from imaging 
				 for MULTIPLE files in a directory
CV...NoiseBright...woutForLoop - code for adding noise, changing brightness, and altering contrast to raw data obtained from 
					   imaging for ONE file
CV...IndividualLoops - same as "CV...NoiseBrightness..." but a single image alteration is performed for every patient prior 
			     to making the next alteration as opposed to making all alterations to one file and then moving to the 
			     next. (i.e., add noise to every file, then add brightness for each, and so on)

Code for Unit Testing:
CV...ImageChanges - tet code for adding noise, changing brightness, and altering contrast from images generated
AddFolders - test code for adding new folders to directories and subdirectories
Convolution - test code for performing 2D convolutions for sharpening and blurring
RemoveFiles - test code for removing unwanted files
RenameFiles - test code for renaming unwanted files

File Naming Convention: 
BreastSide-Chromophore-ChangesMade%%added.jpg
Example: Right-TOI-Noise10.jpg (represents 10% noise added to dataset Right-TOI)
Example: Right-THC-Noise7-Brt10-Cnt60.jpg (represents 7% noise with increase of brightness by .10 and contrast alteration of 
	   60% added to dataset Right-THC)

Miscellaneous Files/Folders:
DK1_TestFolder - used as a test folder when running code to avoid altering original data. Data is from 
		     "PatientPlots" > "DKU01_140502" 
