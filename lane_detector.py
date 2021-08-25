import cv2
import numpy as np
import matplotlib.pyplot as plt
from lane import Lane

     
def detect_lane(filename):
     
  # Load a frame
  original_frame = cv2.imread(filename)
 
  # Create a Lane object
  lane_obj = Lane(orig_frame=original_frame)
 
  # Perform thresholding to isolate lane lines
  lane_line_markings = lane_obj.get_line_markings()
  #cv2.imshow("Image", lane_line_markings)
  # Display the window until any key is pressed
  #cv2.waitKey(0) # Close all windows
  #cv2.destroyAllWindows() 
 
  # Plot the region of interest on the image
  lane_obj.plot_roi(plot=True)

  # Perform the perspective transform to generate a bird's eye view
  warped_frame = lane_obj.perspective_transform(plot=True)

  # Generate the image histogram to serve as a starting point
  histogram = lane_obj.calculate_histogram(plot=True)

  # Find lane line pixels using the sliding window method 
  left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=True)
 
  # Fill in the lane line
  lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=True)
     
  # Overlay lines on the original frame
  frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
 
  # Calculate lane line curvature (left and right lane lines)
  lane_obj.calculate_curvature(print_to_terminal=False)
 
  # Calculate center offset                                                                 
  car_position = lane_obj.calculate_car_position(print_to_terminal=False)
     
  # Display curvature and center offset on image
  frame_with_lane_lines2 = lane_obj.display_curvature_offset(frame=frame_with_lane_lines, plot=False)

  return frame_with_lane_lines, car_position