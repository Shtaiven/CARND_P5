## Project 5 Writeup by Steven Eisinger

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/feature_visualization.png
[image2]: ./output_images/sliding_windows_no_filter.png
[image3]: ./output_images/sliding_windows_heat.png
[image4]: ./output_images/full_pipeline_issues.png
[video1]: ./find_vehicles.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG, color spatial, and color histogram feature extraction methods are located in the 3rd cell of the IPython notebook `vehicle_detection.ipynb`  

I started by reading in all the `vehicle` and `non-vehicle` images into a glob, then used the methods `get_hog_features()`, `bin_spatial()`, and `color_hist()` to extract the features of a random image from each set and display them with matplotlib.

The methods `get_hog_features()`, `bin_spatial()`, and `color_hist()` use `skimage.feature.hog()`, `np.ravel()`, and `np.histogram()` respectively to extract the corresponding features. These three methods were used collectively in both `extract_features()` and `single_image_features()` and concatenated to create the feature set to be used by the classifier in the next section. Example ouput can be seen below:

![alt text][image1]

#### 2. Explain how you settled on your final choice of HOG parameters.

The hyperparameters of 

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

were chosen for these methods based on their accuracy score when training the LinearSVC classifier. These parameters yeilded an accuracy of 98.93%.

The selection process was a matter of trial and error and educated guessing to see which combinations yeilded the highest accuracy while not making the feature array too large, as docker couldn't handle the amount of memory while training the SVM in the next section. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 4th cell of the IPython notebook, I trained a linear SVM using the `LinearSVC()` classifier with its default parameter of `C=1.0` by fitting it with features extracted by `extract_features()` on both the vehicles and non-vehicles sets scaled using `StandardScaler()` from sklearn.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Window search is performed in the 5th cell of the IPython notebook. I slide a 96x96 window across all x from y in the range (360, 720), meaning the entire bottom half of thd image, with 80% overlap. The 96x96 window caught most cars at various distances from the camera and combined with the 80% overlap did very well identifying the same car multiple times, which is necessary for heatmapping and removing duplicates in the 6th and 7th cells of the IPython notebook. Sliding window search output is shown below.

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Below are some images generated by the full pipeline seen in cell 8. I optimized my pipeline by testing the results on the test images provided and tuning hyperparameters until I got satisfactory results, particularly on `test5.jpg`, which has shadows and views into the highway going in the opposite direction, which generally confuses my pipeline.

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./find_vehicles.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video:

![alt text][image3]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

