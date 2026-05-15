import cv2

def test_usb_cam():
    print("Opening Index 2...")
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("Failed to open Index 2")
        return

    # Force the codec and resolution BEFORE reading anything
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera opened. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            break
            
        cv2.imshow("DroidCam USB Fix", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_usb_cam()