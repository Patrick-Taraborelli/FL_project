import flwr as fl

def start_server():
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(),
    )

if __name__ == "__main__":
    print("Starting FL server...")
    start_server()
