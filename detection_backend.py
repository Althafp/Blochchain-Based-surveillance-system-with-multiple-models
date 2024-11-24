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
from web3 import Web3


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
    if send_email == "True":
        email_thread = Thread(target=email_worker, args=(email_queue, receiver_email), daemon=True)
        email_thread.start()

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

                    # Log detection to the smart contract
                    log_detection_to_smart_contract(frame_path, results[0])

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

        if send_email == "True":
            email_queue.put(None)  # Signal email worker to stop
            email_queue.join()

    print("Detection and email threads completed.")


def log_detection_to_smart_contract(frame_path, detection_results):
    """Log detection information to the Ethereum smart contract."""
    # Connect to Ethereum network
    eth_url = "http://127.0.0.1:8545"  # Update with your Ethereum RPC URL
    web3 = Web3(Web3.HTTPProvider(eth_url))

    if not web3.is_connected():
        print("Failed to connect to Ethereum network.")
        return

    # Load contract ABI and address
    with open("detetion.abi", "r") as abi_file:
        contract_abi = json.load(abi_file)
    with open("detection.address", "r") as address_file:
        contract_address = address_file.read().strip()

    # Get the contract
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    # Use the first Ethereum account
    default_account = web3.eth.accounts[0]

    # Prepare detection data
    detections = [
        {
            "class": box.cls.tolist(),
            "confidence": box.conf.tolist(),
            "coordinates": box.xyxy.tolist(),
        }
        for box in detection_results.boxes
    ]
    detection_data = json.dumps(detections)

    # Add log to smart contract
    try:
        tx_hash = contract.functions.addLog(frame_path, detection_data).transact({"from": default_account})
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Log added to smart contract. Transaction hash: {receipt.transactionHash.hex()}")
    except Exception as e:
        print(f"Error logging detection to smart contract: {e}")


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
