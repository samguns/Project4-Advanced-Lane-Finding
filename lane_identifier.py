import numpy as np
import matplotlib.pyplot as plt
import cv2

class LaneIdentifier:
    def __init__(self, smooth_factor, filter):
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.lane_gap = []
        self.binary_warped = None
        self.window_height = None
        self.leftx_current = 0
        self.rightx_current = 0
        self.nonzeroy = None
        self.nonzerox = None
        self.left_fit = None
        self.right_fit = None
        self.margin = 100
        self.nwindows = 9
        self.minpix = 50
        self.leftx = []
        self.lefty = []
        self.rightx = []
        self.righty = []
        self.smooth_factor = smooth_factor
        self.filter = filter
        return


    def identify_lanes(self, binary):
        self.binary_warped = binary
        self.window_height = np.int(self.binary_warped.shape[0] // self.nwindows)

        nonzero = binary.nonzero()
        self.nonzeroy = np.array(nonzero[0])
        self.nonzerox = np.array(nonzero[1])

        if self.left_fit is None or self.right_fit is None:
            self.blind_sliding_window_search()
        else:
            self.selective_window_search()

        ret = self.extract_lane_lines()
        if ret is False:
            return False, None, None

        return True, self.left_fit, self.right_fit


    def blind_sliding_window_search(self):
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        l_lane_inds = []
        r_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = self.binary_warped.shape[0] - (window + 1) * self.window_height
            win_y_high = self.binary_warped.shape[0] - window * self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((self.nonzeroy >= win_y_low) &
                              (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) &
                              (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) &
                              (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xright_low) &
                              (self.nonzerox < win_xright_high)).nonzero()[0]
            l_lane_inds.append(good_left_inds)
            r_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        self.left_lane_inds = np.concatenate(l_lane_inds)
        self.right_lane_inds = np.concatenate(r_lane_inds)
        return


    def selective_window_search(self):
        self.left_lane_inds = ((self.nonzerox >
                                (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy +
                                 self.left_fit[2] - self.margin)) &
                               (self.nonzerox <
                                (self.left_fit[0] * (self.nonzeroy ** 2) + self.left_fit[1]*self.nonzeroy +
                                 self.left_fit[2] + self.margin)))
        self.right_lane_inds = ((self.nonzerox >
                                (self.right_fit[0] * (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy +
                                 self.right_fit[2] - self.margin)) &
                               (self.nonzerox <
                                (self.right_fit[0] * (self.nonzeroy ** 2) + self.right_fit[1] * self.nonzeroy +
                                 self.right_fit[2] + self.margin)))

        return


    def extract_lane_lines(self):
        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        if leftx.size == 0 or rightx.size == 0:
            if self.left_fit is None or self.right_fit is None:
                return False

        # Outliers filter, delete those that far away from previous
        # recognized lane curve.
        if self.left_fit is not None:
            leftx_trend = self.left_fit[0]*lefty*lefty + self.left_fit[1]*lefty + self.left_fit[2]
            range = abs(leftx - leftx_trend)
            indices = (range > self.filter).nonzero()
            leftx = np.delete(leftx, indices)
            lefty = np.delete(lefty, indices)

        if self.right_fit is not None:
            rightx_trend = self.right_fit[0]*righty*righty + self.right_fit[1]*righty + self.right_fit[2]
            range = abs(rightx - rightx_trend)
            indices = (range > self.filter).nonzero()
            rightx = np.delete(rightx, indices)
            righty = np.delete(righty, indices)

        # Take previous identified pixels into 2nd order polynomial
        # calculation, in order to alleviate oscillation.
        self.leftx = np.append(self.leftx, leftx)
        self.lefty = np.append(self.lefty, lefty)
        self.rightx = np.append(self.rightx, rightx)
        self.righty = np.append(self.righty, righty)

        self.leftx = self.leftx[-self.smooth_factor:]
        self.lefty = self.lefty[-self.smooth_factor:]
        self.rightx = self.rightx[-self.smooth_factor:]
        self.righty = self.righty[-self.smooth_factor:]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        return True


    def visualization(self):
        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0] - 1, self.binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped)) * 255
        fit_img = np.zeros_like(out_img)
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(fit_img, 1, window_img, 0.3, 0)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(out_img)
        ax1.set_title('Detected Lane Points', fontsize=30)

        ax2.imshow(result)
        ax2.set_title('Lane Lines', fontsize=30)

        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)