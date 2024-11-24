import sys
import cv2
from ultralytics import YOLO
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
from cryptography.fernet import Fernet
from threading import Thread
from queue import Queue
from datetime import datetime
import arweave
import requests


def load_model(model_path):
    """Load the YOLO model."""
    return YOLO(model_path)


def load_encryption_key():
    """Load the encryption key for credentials."""
    key_file = "encryption.key"
    if not os.path.exists(key_file):
        raise FileNotFoundError("Encryption key not found. Please set up credentials.")
    with open(key_file, "rb") as file:
        return file.read()


def load_email_credentials():
    """Load and decrypt email credentials."""
    credentials_file = "email_credentials.enc"
    if not os.path.exists(credentials_file):
        raise FileNotFoundError("Email credentials not found. Please set them up in the UI.")

    encryption_key = load_encryption_key()
    with open(credentials_file, "rb") as file:
        encrypted_data = file.read()
    fernet = Fernet(encryption_key)
    return json.loads(fernet.decrypt(encrypted_data).decode())


def send_email_with_frame(frame_path, receiver_email):
    """Send an alert email with the detected frame attached."""
    try:
        credentials = load_email_credentials()
        sender_email = credentials["sender_email"]
        sender_password = credentials["sender_password"]

        subject = "Surveillance Alert"
        body = "A detection event occurred in the live feed. See the attached image for details."

        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.attach(MIMEText(body, "plain"))

        # Attach the image file
        try:
            with open(frame_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f"attachment; filename={os.path.basename(frame_path)}",
            )
            msg.attach(part)
        except Exception as e:
            print(f"Error attaching image: {e}")
            return

        # Send the email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Email with frame sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


def run_detection(input_source, send_email, receiver_email, models):
    """Run detection on the selected input using the selected models."""
    yolo_models = [load_model(model_path) for model_path in models]
    cap = cv2.VideoCapture(0 if input_source == "Webcam" else input_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Create output folder
    output_folder = "detected_frames"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize queues and threads
    email_queue = Queue()
    arweave_queue = Queue()  # Initialize Arweave queue
    if send_email == "True":
        email_thread = Thread(target=email_worker, args=(email_queue, receiver_email), daemon=True)
        email_thread.start()

    arweave_thread = Thread(target=arweave_worker, args=(arweave_queue,), daemon=True)  # Initialize Arweave thread
    arweave_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            email_sent = False  # Flag to track if email has been sent for this frame

            for model in yolo_models:
                results = model(frame)
                frame = results[0].plot()

                # Check if any objects are detected
                if len(results[0].boxes) > 0:
                    # Save the frame locally
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    frame_path = os.path.join(output_folder, f"frame_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)

                    # Add frame to Arweave queue
                    arweave_queue.put((frame_path, results[0]))

                    # Optionally send email
                    if send_email == "True" and not email_sent:
                        email_queue.put(frame_path)
                        email_sent = True  # Set the flag to avoid duplicate emails

            cv2.imshow("YOLO Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Stop the threads
        arweave_queue.put(None)  # Signal Arweave worker to stop
        arweave_queue.join()

        if send_email == "True":
            email_queue.put(None)  # Signal email worker to stop
            email_queue.join()

    print("Detection, email, and Arweave threads completed.")



def arweave_worker(arweave_queue):
    """Worker thread for uploading frames to ArLocal."""
    wallet_path = "6pminIkHfO4UbtPgGGFJJI7-8f5Sn9Y4WmdWBTE_aLQ.json"  # Replace with your wallet path
    try:
        wallet = arweave.Wallet(wallet_path)
        wallet.api_url = 'http://localhost:1984'  # Point to ArLocal
    except Exception as e:
        print(f"Error loading Arweave wallet: {e}")
        return

    while True:
        task = arweave_queue.get()
        if task is None:  # Sentinel to stop the thread
            break

        frame_path, detection_results = task
        arweave_url = None

        try:
            # Read the frame data
            with open(frame_path, "rb") as f:
                data = f.read()

            # Create a transaction
            transaction = arweave.Transaction(wallet, data=data)
            transaction.add_tag("Content-Type", "image/jpeg")
            transaction.add_tag("App-Name", "SurveillanceSystem")
            transaction.sign()
            transaction.send()

            # Mine the transaction in ArLocal
            mine_url = f"{wallet.api_url}/mine"
            response = requests.get(mine_url)
            if response.status_code == 200:
                # Generate Arweave URL
                arweave_url = f"{wallet.api_url}/{transaction.id}"
                
                status_url = f'http://localhost:1984/tx/{transaction.id}/status'
                response = requests.get(status_url)

                if response.status_code == 200:
                    status = response.json()
                    if status.get('confirmed'):
                        print("Transaction is confirmed.")
                    else:
                        print("Transaction is not yet confirmed.")
                else:
                    print(f"Error checking transaction status: {response.text}")
                print(f"Frame uploaded to ArLocal: {arweave_url}")
            else:
                print(f"Error mining transaction: {response.text}")
        except Exception as e:
            print(f"Error uploading to ArLocal: {e}")

        # Log detection with Arweave URL
        log_detection(frame_path, detection_results, arweave_url)

        arweave_queue.task_done()



def log_detection(frame_path, detection_results, arweave_url=None):
    """Log detection information to a JSON file."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frame_path": frame_path,
        "arweave_url": arweave_url,  # Add Arweave URL
        "detections": []
    }

    for box in detection_results.boxes:
        log_entry["detections"].append({
            "class": box.cls.tolist(),  # Detected class ID
            "confidence": box.conf.tolist(),  # Confidence score
            "coordinates": box.xyxy.tolist(),  # Bounding box coordinates
        })

    # Append to JSON log file
    log_file = "detection_log.json"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error logging detection: {e}")


def email_worker(email_queue, receiver_email):
    """Worker thread for sending emails."""
    while True:
        frame_path = email_queue.get()  # Wait for a task
        if frame_path is None:  # Sentinel value to signal exit
            break
        send_email_with_frame(frame_path, receiver_email)  # Process the email
        email_queue.task_done()  # Mark the task as done


if __name__ == "__main__":
    try:
        input_source = sys.argv[1]
        send_email = sys.argv[2]
        receiver_email = sys.argv[3]
        models = sys.argv[4:]
        run_detection(input_source, send_email, receiver_email, models)
    except IndexError:
        print("Usage: python detection_backend.py <input_source> <send_email> <receiver_email> <models...>")
    except Exception as e:
        print(f"An error occurred: {e}")
