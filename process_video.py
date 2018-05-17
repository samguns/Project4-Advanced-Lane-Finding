from moviepy.editor import VideoFileClip
from utils import *
from lane_identifier import LaneIdentifier


def warp_binary_pipe(image, mtx, dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    v_binary = hsv_select(undist, v_thresh=(180, 255))
    r_binary = red_select(undist, r_thresh=(170, 255))
    binary = np.zeros_like(r_binary)
    binary[((r_binary == 1) & (v_binary == 1))] = 1

    bottom_width = 0.76
    mid_width = 0.2
    height_pct = 0.68
    warped, Minv = warp(binary, bottom_width, mid_width, height_pct)

    return warped, Minv


def process_image(image):
    binary_warped, Minv = warp_binary_pipe(image, mtx, dist)

    ret, left_fit, right_fit = lane_identifier.identify_lanes(binary_warped)
    if ret is True:
        img_size = (image.shape[1], image.shape[0])
        window_width = 25

        yvals = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fitx = right_fit[0] * yvals * yvals + right_fit[1] * yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2),
                                                     axis=0),
                                      np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width / 2, right_fitx[::-1] + window_width / 2),
                                                     axis=0),
                                      np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        inner_lane = np.array(
            list(zip(np.concatenate((left_fitx + window_width / 2, right_fitx[::-1] + window_width / 2),
                                    axis=0),
                     np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
        road = np.zeros_like(image)
        road_bkg = np.zeros_like(image)
        cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
        cv2.fillPoly(road_bkg, [left_lane], color=[255, 0, 255])
        cv2.fillPoly(road_bkg, [right_lane], color=[255, 0, 255])

        road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
        road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)
        base = cv2.addWeighted(image, 1, road_warped_bkg, 0.8, 0)
        result = cv2.addWeighted(base, 1, road_warped, 0.4, 0)

        curverad, center_offset = calculate_curvature(yvals, left_fitx, right_fitx)
        side_pos = 'left'
        if center_offset <= 0:
            side_pos = 'right'

        cv2.putText(result, 'Radius of Curvature = ' + str(round(curverad, 3)) + '(m)',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)
        cv2.putText(result, 'Vehicle is ' + str(abs(round(center_offset, 3))) + '(m) ' + side_pos + ' of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 2)

        return result

    else:
        return image


if __name__ == "__main__":
    videos = ['challenge_video.mp4', 'project_video.mp4']

    for input_video in videos:
        lane_identifier = LaneIdentifier(smooth_factor=100000, filter=85)
        output_video = 'track_' + input_video
        input_clip = VideoFileClip(input_video)
        output_clip = input_clip.fl_image(process_image)
        output_clip.write_videofile(output_video, audio=False)