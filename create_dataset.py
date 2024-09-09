import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

def process_image(img_path, hands):
    """Process an image to extract hand landmarks."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]

            # Normalize and pad landmarks
            min_x, min_y = min(x_coords), min(y_coords)
            for i in range(len(hand_landmarks.landmark)):
                x = (hand_landmarks.landmark[i].x - min_x)
                y = (hand_landmarks.landmark[i].y - min_y)
                data_aux.append(x)
                data_aux.append(y)

            # Ensure consistent length
            data_aux = data_aux[:landmark_length] + [0] * (landmark_length - len(data_aux))
    else:
        # No hand landmarks found, return empty data
        data_aux = [0] * landmark_length

    return data_aux

def create_dataset(data_dir, hands):
    """Create dataset from images in the specified directory."""
    data = []
    labels = []

    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if os.path.isdir(dir_path):
            for img_path in os.listdir(dir_path):
                img_full_path = os.path.join(dir_path, img_path)
                data_aux = process_image(img_full_path, hands)
                data.append(data_aux)
                labels.append(dir_)

    return data, labels

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)
    DATA_DIR = './data1'

    # Define a fixed length for each hand landmark (assuming there are 21 landmarks per hand)
    global landmark_length
    landmark_length = 21 * 2  # 21 (x, y) coordinates

    data, labels = create_dataset(DATA_DIR, hands)

    # Save the data and labels to a pickle file
    with open('data1.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

if __name__ == "__main__":
    main()