# AI-FAce-Blurring-App
Web App that blurs the faces in images or videos that are uploaded


How to train and use the model:


In order to re-train the model, you may have to resize your training images to a fixed size. To do so:

1) Go to the process_image.py file and enter the correct folder path for both the segmented and normal images. If you're using the same dataset as the CCIHP project, no need to change anything other than the folder paths.

2) Create the target coordinates file. Run the find_color_coordinates.py by changing the file path under the find_coordinates function to that of your processed segmented images folder. This will output a new csv file that is easily readable for the human eye. However, to change it into the correct format to train the model, paste the file path to the new csv file that was created from the above into the vector function right under it. The new csv file that comes from it is the one that will be used to train the model

3) At main.py, change the code at the beginning such that in the end, your training_folder and training_path lead to the processed training image folder and the target coordinate csv file respectively. Make sure to change the output path for the saved weights according to your preference. Feel free to tune the hyperparameters for better performance.


To use the model with the saved weights, do the following:

1) Go to blur_img.py and change all the strings where it mentions "path_to_weights_here" to the path to the saved weights on your computer.

2) Change the image folder to that of the images you want to censor.

3) Modify the for loop below to change which image you want to see censored. A series of images will be shown processed by the model strictly from the censor_img_test function. The two functions above censor images and videos on the website of this project.

To check the accuracy of this project, do the following:

1) Head to accuracy.py and tune the default input variables of the censor_img_test function. Read the DOCSTRING for instructions on what each input variable does and their effect on the final processed image.

2) Change the variables 'validation', 'weights' and 'targets_path' to the validation image folder (of any image dimensions), the model saved weights file, and the target coordinates of the images in the validation folder.

3) After running the file, the output will be a message indicating the average IoU, the ratio of Target Area/Predicted Area, and the percent of times the model strictly covered all the faces, regardless of how big the boxes are.


If I figured out how to upload large files to Github, all the files and images to do the above should be in the Github as well.
