import flwr as fl
import numpy as np
import sqlite3
import time
import threading
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# Directory where the PCAP files are stored.
PCAP_DIRECTORY = "/app/data/pcaps"

# Path to the anomalies SQLite database.
DB_PATH = "anomalies.db"

# Mapping of class labels to attack names.
class_mapping = {
    0: "Benign",
    1: "Port Scan",
    2: "SSH Brute Force",
    3: "MQTT Brute Force"
}

def initialize_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            message TEXT,
            pcap_file TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_anomaly_to_db(anomalies, pcap_path):
    if not anomalies:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for idx, anomaly in anomalies:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "INSERT INTO anomaly_logs (timestamp, message, pcap_file) VALUES (?, ?, ?)",
            (timestamp, anomaly, pcap_path)
        )
    conn.commit()
    conn.close()

def extract_features_from_pcap(pcap_path, label):
    packets = rdpcap(pcap_path)
    features = []
    for packet in packets:
        feature_dict = {}
        if IP in packet:
            feature_dict["ip_len"] = packet[IP].len
            feature_dict["ip_ttl"] = packet[IP].ttl
        if TCP in packet or UDP in packet:
            feature_dict["src_port"] = packet.sport
            feature_dict["dst_port"] = packet.dport
            feature_dict["protocol"] = 6 if TCP in packet else 17
        feature_dict["label"] = label
        features.append(feature_dict)
    return pd.DataFrame(features).fillna(0)

def load_data():
    initialize_db()
    data_frames = []
    # Load and combine training data from four PCAP files.
    labeled_pcaps = {
        "benign.pcap": 0,
        "port_scan.pcap": 1,
        "ssh_brute_force.pcap": 2,
        "mqtt_brute_force.pcap": 3
    }
    for file, label in labeled_pcaps.items():
        df = extract_features_from_pcap(f"{PCAP_DIRECTORY}/{file}", label)
        data_frames.append(df)
    data = pd.concat(data_frames, ignore_index=True)
    X = data.drop("label", axis=1).values
    y = data["label"].values
    return X, y

def create_model(input_shape):
    """
    Create a neural network model with three layers:
      - Input layer with 32 neurons (ReLU activation)
      - Hidden layer with 16 neurons (ReLU activation)
      - Output layer with 4 neurons (softmax activation for 4 classes)
    
    Parameters:
    - input_shape: Number of features in the input.
    
    Returns:
    - A compiled Keras model.
    """
    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_shape,)),
        Dense(16, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

class FLClient(fl.client.NumPyClient):
    """
    Federated Learning client that:
      - Loads training data from PCAP files.
      - Creates and trains a neural network model.
      - Applies differential privacy by adding Gaussian noise to model weights.
    """
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.X, self.y = load_data()
        self.model = create_model(self.X.shape[1])
        self.epsilon = epsilon
        self.delta = delta
        self.noise_scale = self.calculate_noise_scale()

    def calculate_noise_scale(self):
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, weights):
        """
        Add Gaussian noise to each weight array.
        
        Parameters:
        - weights: List of numpy arrays representing model weights.
        
        Returns:
        - List of numpy arrays with added noise.
        """
        noise = [np.random.normal(0, self.noise_scale, size=w.shape) for w in weights]
        return [w + n for w, n in zip(weights, noise)]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """
        Update the model with received parameters, train the model on local data,
        add noise for differential privacy, and return the updated weights.
        """
        self.model.set_weights(parameters)
        self.model.fit(self.X, self.y, epochs=5, batch_size=32, verbose=0)
        weights = self.model.get_weights()
        noisy_weights = self.add_noise(weights)
        return noisy_weights, len(self.X), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X, self.y, verbose=0)
        return loss, len(self.X), {"accuracy": accuracy}

def detect_attack_on_file(client_instance, pcap_file):
    """
    Load a PCAP file (for detection purposes), extract its features, and use the model to predict.
    
    Parameters:
    - client_instance: An instance of FLClient with a trained model.
    - pcap_file: The filename of the PCAP to analyze (e.g., "test.pcap").
    
    Returns:
    - predicted_class: The predicted class (0 for Benign; 1, 2, or 3 for attack types).
    """
    
    df = extract_features_from_pcap(f"{PCAP_DIRECTORY}/{pcap_file}", -1)
    # Remove the label column to get only the features.
    X = df.drop("label", axis=1).values
     # Predict the class for each packet and average the predictions.
    predictions = client_instance.model.predict(X)
    avg_pred = np.mean(predictions, axis=0)
    predicted_class = np.argmax(avg_pred)
    return predicted_class

def detection_loop(client_instance):
    """
    Continuously check (every 60 seconds) a PCAP file ("test.pcap") for attacks.
    
    If the predicted class is not 0 (Benign), print a message to the terminal
    and log the detection in the anomalies database.
    """
    while True:
        time.sleep(60)
        try:
            predicted_class = detect_attack_on_file(client_instance, "test.pcap")
            if predicted_class != 0:
                attack_type = class_mapping.get(predicted_class, "Unknown")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Attack detected: {attack_type}")
                log_anomaly_to_db([(0, attack_type)], f"{PCAP_DIRECTORY}/test.pcap")
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No attack detected in test.pcap.")
        except Exception as e:
            print("Error during detection:", e)

if __name__ == "__main__":
    print("Starting FL client...")
    client_instance = FLClient(epsilon=100.0, delta=1e-5)
    
    # Start a separate thread for continuous attack detection on "test.pcap".
    detection_thread = threading.Thread(target=detection_loop, args=(client_instance,), daemon=True)
    detection_thread.start()
    
    # Start the federated learning client to connect to the FL server.
    fl.client.start_numpy_client(
        server_address="server_container:8080",
        client=client_instance,
    )
    
    # Keep the container running (for debugging purposes).
    while True:
        time.sleep(60)
