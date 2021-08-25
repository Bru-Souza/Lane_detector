
import cv2
from lane_detector import detect_lane


def main():
    image = 'test.png'
    
    lane_marking, center = detect_lane(image)
    print(f'{center}'+ ' cm')
    
    # Display the image 
    cv2.imshow("Image", lane_marking)
    # Display the window until any key is pressed
    cv2.waitKey(0) # Close all windows
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    main()