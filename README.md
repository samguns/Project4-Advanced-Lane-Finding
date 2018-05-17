## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistort.png "Undistorted"
[image2]: ./images/undistort_test1.png "Road Transformed"
[image3]: ./images/binary_combo.png "Binary Examples"
[image4]: ./images/warped.png "Warp Example"
[image5]: ./images/lane_identifier_visualization.png "Fit Visual"
[image6]: ./images/output.png "Output"
[video1]: ./track_project_video.mp4 "Video"
[video2]: ./track_challenge_video.mp4 "Video"


---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in cal_cam.py.  

I start by preparing `objpoints` and `imgpoints` from the provided images in 'camera_cal' folder. This step was defined in function 'collect_points', in which it iterates through all images and using cv2.findChessboardCorners to find out the 9x6 corners in them, thus I obtained the real world space points and its 2d space mapping points.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function in function 'calibrate_camera'.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I put all the color space filtering functions, gradients thresholded methods and warp transformation in the 'utils.py'. Then I worked on different combinations to get a satisfied binary image for the following lane line detection. Of all the results I've tried, I found out that by combining (180, 255) thresholded output of V binary in HSV space, and (170, 255) output of R binary in RGB space, gave me an acceptable binary to get through the 'project_video' and 'challenge_video'. Below is an example output binary.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 85 through 104 in the file `utils.py`.  The `warp()` function takes as inputs an image (`img`), as well as desired trapezoid region defined as ('bottom_width, mid_width' and 'height_pct').

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 512, 489.6      | 320, 0        |
| 768, 489.6      | 960, 0      |
| 1126.4, 673.2     | 960, 720      |
| 153.6, 673.2      | 320, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I wrote a class 'LaneIdentifier' in 'lane_identifier.py' to find lane line pixels and generate corresponding second order polynomial fit.

I adopted the sliding window method from the course material as my basic implementation. Because both 'project_video' and 'challenge_video' don't have sharp turns, the identifier just does one sliding window search blindly for the first frame. For the following frames, it just search a marginal areas based on previously recognized polynomial fit. It not only helps identifying lane lines efficiently, but plays an outlier filter as well. Below is the code snippet from line 116 to 128 of 'lane_identifier.py' for left lane line points processing:
```python
if self.left_fit is not None:
    leftx_trend = self.left_fit[0]*lefty*lefty + self.left_fit[1]*lefty + self.left_fit[2]
    range = abs(leftx - leftx_trend)
    indices = (range > self.filter).nonzero()
    leftx = np.delete(leftx, indices)
    lefty = np.delete(lefty, indices)
```
After the identifier collected all potential Y coordinate points of the left lane line (lefty), fed them into the polynomial fit to get a trending X points. The configurable parameter 'self.filter' checks if there are points that seem to be far away from previous lane line. One thing I should point it out here is this is a rough implementation that handles mild curve well enough, but falls short when facing sharp turns.

On some occasions, the sliding window will not gather any points or the points are totally discarded by the filter. Instead of going back to blind searching, I let the identifier to extrapolate a temporary lane fit, by adding a points stack, e.g. 'self.leftx', to store some previously collected points. The size of the stack is governed by parameter 'self.smooth_factor'.

This is a visualization of the output from 'LaneIdentifier':
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 136 through 145 in my code in `utils.py`. They're almost copied from the course material and Q&A session.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's two videos I got by running 'process_video.py'.

[project_video](./track_project_video.mp4)

[challenge_video](./track_challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When I started to work on this project, I found it's hard to make a good thresholded binary that could cover almost any situation. For example, the fine tuned thresholds works well for images taken on road under sunny weather. But when it comes to deal with images that have shadows, it often resulted in a disastrous binary. I think it might be worth trying to make the thresholds adaptively.

Likewise, the shape of source trapezoid to be warped isn't easy to decide. When I tried my current code on the 'harder_challenge_video', the selected region was too large, it included too much noises. However, if the region is not big enough, many useful pixels will be blocked out. I also would like to try an adaptive approach on this one.

The most difficult problem I encountered and I'm still not satisfied with, is to smooth the lane lines. I tried different ideas, such as maintaining a mean polynomial fit, combining current learned pixels with pixels of last frame, and so on. None worked well, so I ended up with this pixel stack that mentioned in item 4 above. I believe there's some methods to handle this problem, but I think it's beyond my knowledge at the moment. If possible, I'd like to learn more and go deeper into dealing with this issue.
