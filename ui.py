import sys
import os
import json
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QScrollArea,
    QFrame,
    QGridLayout,
    QWidget,
    QMessageBox,
    QCheckBox,
    QFileDialog,
    QLineEdit,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from cryptography.fernet import Fernet
import subprocess
from functools import partial


class LoginPage(QWidget):
    def __init__(self, switch_to_dashboard, encryption_key, credentials_file):
        super().__init__()
        self.switch_to_dashboard = switch_to_dashboard
        self.encryption_key = encryption_key
        self.credentials_file = credentials_file
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Welcome to Surveillance Platform")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #81A1C1;")
        layout.addWidget(title_label)

        subtitle_label = QLabel("Please enter your credentials to proceed")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #D8DEE9;")
        layout.addWidget(subtitle_label)

        # Email Inputs
        self.sender_email_input = QLineEdit()
        self.sender_email_input.setPlaceholderText("Enter Sender Email")
        self.sender_email_input.setStyleSheet(self.input_style())
        self.sender_password_input = QLineEdit()
        self.sender_password_input.setPlaceholderText("Enter Sender Password")
        self.sender_password_input.setEchoMode(QLineEdit.Password)
        self.sender_password_input.setStyleSheet(self.input_style())
        self.receiver_email_input = QLineEdit()
        self.receiver_email_input.setPlaceholderText("Enter Receiver Email")
        self.receiver_email_input.setStyleSheet(self.input_style())
        layout.addWidget(self.sender_email_input)
        layout.addWidget(self.sender_password_input)
        layout.addWidget(self.receiver_email_input)

        # Save Button
        save_button = QPushButton("Save & Continue")
        save_button.setStyleSheet(self.button_style())
        save_button.clicked.connect(self.save_email_credentials)
        layout.addWidget(save_button)

        layout.addStretch()
        self.setLayout(layout)

    def save_email_credentials(self):
        sender_email = self.sender_email_input.text()
        sender_password = self.sender_password_input.text()
        receiver_email = self.receiver_email_input.text()

        if not sender_email or not sender_password or not receiver_email:
            QMessageBox.warning(self, "Error", "All fields are required.")
            return

        credentials = {
            "sender_email": sender_email,
            "sender_password": sender_password,
            "receiver_email": receiver_email,
        }

        encrypted_data = self.encrypt_data(json.dumps(credentials))
        with open(self.credentials_file, "wb") as file:
            file.write(encrypted_data)

        QMessageBox.information(self, "Success", "Credentials saved successfully.")
        self.switch_to_dashboard()

    def encrypt_data(self, data):
        fernet = Fernet(self.encryption_key)
        return fernet.encrypt(data.encode())

    @staticmethod
    def input_style():
        return """
            QLineEdit {
                background-color: #4C566A;
                border: 1px solid #81A1C1;
                border-radius: 10px;
                padding: 10px;
                color: #ECEFF4;
            }
        """

    @staticmethod
    def button_style():
        return """
            QPushButton {
                background-color: #81A1C1;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
        """


class DashboardPage(QWidget):
    def __init__(self, encryption_key, credentials_file):
        super().__init__()
        self.encryption_key = encryption_key
        self.credentials_file = credentials_file
        self.selected_models = set()  # To track selected models
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Title
        title_label = QLabel("Surveillance Dashboard")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #81A1C1;")
        layout.addWidget(title_label)

        subtitle_label = QLabel("Select options below to start detection")
        subtitle_label.setFont(QFont("Arial", 14))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #D8DEE9;")
        layout.addWidget(subtitle_label)

        layout.addStretch()

        # Input Source
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Source:")
        input_label.setFont(QFont("Arial", 12))
        input_label.setStyleSheet("color: #ECEFF4;")
        self.input_combo = QComboBox()
        self.input_combo.addItem("Webcam")
        self.input_combo.addItem("Video File")
        self.input_combo.currentIndexChanged.connect(self.handle_input_selection)
        self.video_path = None
        self.input_combo.setStyleSheet(self.combo_style())
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_combo)
        layout.addLayout(input_layout)

        # Model Selection as Blocks
        model_label = QLabel("Select Models:")
        model_label.setFont(QFont("Arial", 12))
        model_label.setStyleSheet("color: #ECEFF4;")
        layout.addWidget(model_label)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        model_container = QWidget()
        grid_layout = QGridLayout()

        self.model_paths = {
            "Detect Accident": "accident_detection.pt",
            "Fire Accident Detection": "activity_detection.pt",
            "Shop Lift": "shoplift.pt",
            "Human Activity": "yolov8n-pose.pt",
            "Precrime Detection": "precrime.pt",
            "Weapon Detection": "weapon_detection.pt",
        }

        for i, (model_name, model_path) in enumerate(self.model_paths.items()):
            model_box = QFrame()
            model_box.setStyleSheet(self.unselected_block_style())
            model_box.setFixedSize(230, 150)

            model_label = QLabel(model_name)
            model_label.setFont(QFont("Arial", 10, QFont.Bold))
            model_label.setAlignment(Qt.AlignCenter)
            model_label.setStyleSheet("color: #ECEFF4;")
            model_box_layout = QVBoxLayout()
            model_box_layout.addWidget(model_label)
            model_box.setLayout(model_box_layout)

            # Use functools.partial to bind arguments
            model_box.mousePressEvent = partial(self.toggle_model_selection, model_name, model_box)
            grid_layout.addWidget(model_box, i // 3, i % 3)

        model_container.setLayout(grid_layout)
        scroll_area.setWidget(model_container)
        layout.addWidget(scroll_area)

        # Send Email Checkbox
        self.message_checkbox = QCheckBox("Enable Email Notifications")
        self.message_checkbox.setFont(QFont("Arial", 12))
        self.message_checkbox.setStyleSheet("color: #ECEFF4;")
        layout.addWidget(self.message_checkbox)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        run_button = QPushButton("Run Detection")
        run_button.setStyleSheet(self.button_style())
        run_button.clicked.connect(self.run_detection)
        quit_button = QPushButton("Quit")
        quit_button.setStyleSheet(self.button_style())
        quit_button.clicked.connect(QApplication.quit)
        button_layout.addWidget(run_button)
        button_layout.addWidget(quit_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def toggle_model_selection(self, model_name, model_box, event):
        if model_name in self.selected_models:
            self.selected_models.remove(model_name)
            model_box.setStyleSheet(self.unselected_block_style())
        else:
            self.selected_models.add(model_name)
            model_box.setStyleSheet(self.selected_block_style())


    def handle_input_selection(self):
        if self.input_combo.currentText() == "Video File":
            self.video_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
            )
            if not self.video_path:
                QMessageBox.warning(self, "Input Selection", "No video file selected.")
                self.input_combo.setCurrentIndex(0)

    def decrypt_data(self, encrypted_data):
        fernet = Fernet(self.encryption_key)
        return json.loads(fernet.decrypt(encrypted_data).decode())

    def run_detection(self):
        input_source = (
            "Webcam" if self.input_combo.currentText() == "Webcam" else self.video_path
        )
        if not input_source:
            QMessageBox.warning(self, "Error", "Please select an input source.")
            return

        if not self.selected_models:
            QMessageBox.warning(self, "Error", "No models selected.")
            return

        with open(self.credentials_file, "rb") as file:
            credentials = self.decrypt_data(file.read())
        receiver_email = credentials["receiver_email"]

        send_email = "True" if self.message_checkbox.isChecked() else "False"

        subprocess.Popen(
            [
                "python",
                "detection_backend.py",
                input_source,
                send_email,
                receiver_email,
            ]
            + [self.model_paths[model] for model in self.selected_models]
        )

    @staticmethod
    def combo_style():
        return """
            QComboBox {
                background-color: #4C566A;
                border: 1px solid #81A1C1;
                border-radius: 10px;
                padding: 5px;
                color: #ECEFF4;
            }
        """

    @staticmethod
    def button_style():
        return """
            QPushButton {
                background-color: #81A1C1;
                color: #ECEFF4;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5E81AC;
            }
            QPushButton:pressed {
                background-color: #4C566A;
            }
        """

    @staticmethod
    def selected_block_style():
        return """
            QFrame {
                background-color: #5E81AC;
                border: 2px solid #81A1C1;
                border-radius: 10px;
            }
        """

    @staticmethod
    def unselected_block_style():
        return """
            QFrame {
                background-color: #4C566A;
                border: 1px solid #81A1C1;
                border-radius: 10px;
            }
        """


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surveillance")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("background-color: #2E3440;")
        self.showFullScreen()

        self.encryption_key = self.load_or_generate_key()
        self.credentials_file = "email_credentials.enc"

        self.initUI()

    def initUI(self):
        if not os.path.exists(self.credentials_file):
            self.show_login_page()
        else:
            self.show_dashboard_page()

    def show_login_page(self):
        login_page = LoginPage(self.show_dashboard_page, self.encryption_key, self.credentials_file)
        self.setCentralWidget(login_page)

    def show_dashboard_page(self):
        dashboard_page = DashboardPage(self.encryption_key, self.credentials_file)
        self.setCentralWidget(dashboard_page)

    def load_or_generate_key(self):
        key_file = "encryption.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as file:
                return file.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as file:
                file.write(key)
            return key


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
