import pyzed.sl as sl
import cv2

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    image = sl.Mat()

    runtime_parameters = sl.RuntimeParameters()
    i = 0
    key = ""
    while key != 113: # for 'q' key
        # Grab an image, a RuntimeParameters object must be given to grab()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT)
            cv2.imshow("ZED", image.get_data())
            key = cv2.waitKey(100)
            if key == ord('s'):
                cv2.imwrite("left/" + str(i) + ".jpg", image.get_data())

                zed.retrieve_image(image, sl.VIEW.RIGHT)
                cv2.imwrite("right/" + str(i) + ".jpg", image.get_data())
                i = i+1
    # Close the camera
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
