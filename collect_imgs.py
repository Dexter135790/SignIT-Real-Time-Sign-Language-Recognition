import os
import cv2

def collect_images(data_dir, number_of_classes=49, dataset_size=100):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    for j in range(41, number_of_classes):
        class_dir = os.path.join(data_dir, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print(f'Collecting data for class {j}')
        print('Press "Q" to start collecting images')

        # Wait for user to press 'Q' to start collecting images
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            cv2.putText(frame, 'Ready? Press "Q" to Start', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            image_path = os.path.join(class_dir, f'{counter}.jpg')
            success = cv2.imwrite(image_path, frame)
            if not success:
                print(f"Error: Failed to save image {image_path}.")
            counter += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    DATA_DIR = './data1'
    collect_images(DATA_DIR)