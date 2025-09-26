import cv2
import os
import numpy as np
from datetime import datetime
from PIL import Image

# ------------------------------
# Step 1: Dataset Creator
# ------------------------------
def create_dataset():
    name = input("Enter your name: ").strip()
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    count = 0
    print("Collecting images... Press 'q' to quit early.")
    while True:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_path, f"{name}_{count}.jpg"), face_img)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Dataset Creator", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection completed for {name} with {count} images.")


# ------------------------------
# Step 2: Train Model
# ------------------------------
def train_model():
    dataset_path = "dataset"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []
    label_dict = {}
    current_id = 0

    print("Training model...")

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "_").lower()

                if label not in label_dict:
                    label_dict[label] = current_id
                    current_id += 1

                img = Image.open(path).convert("L")  # grayscale
                img_numpy = np.array(img, "uint8")
                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(label_dict[label])

    recognizer.train(face_samples, np.array(ids))
    recognizer.write("trainer.yml")

    with open("labels.txt", "w") as f:
        for label, idx in label_dict.items():
            f.write(f"{idx},{label}\n")

    print("Model training completed and saved as trainer.yml & labels.txt.")


# ------------------------------
# Step 3: Attendance System
# ------------------------------
def mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = {}
    with open("labels.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split(",")
            labels[int(idx)] = label

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"
    attendance = set()

    print("Press 'q' to quit attendance system.")

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 80:
                name = labels[id_]
                cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                attendance.add(name)
            else:
                cv2.putText(img, "Unknown", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Attendance System", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    with open(filename, "w") as f:
        f.write("Name,Date,Time\n")
        for name in attendance:
            now = datetime.now().strftime("%Y-%m-%d,%H:%M:%S")
            f.write(f"{name},{now}\n")

    print(f"Attendance saved in {filename}")


# ------------------------------
# Main Menu
# ------------------------------
while True:
    print("\n--- Face Recognition Attendance System ---")
    print("1. Create Dataset")
    print("2. Train Model")
    print("3. Mark Attendance")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        create_dataset()
    elif choice == "2":
        train_model()
    elif choice == "3":
        mark_attendance()
    elif choice == "4":
        print("Exiting program...")
        break
    else:
        print("Invalid choice. Please try again.")
