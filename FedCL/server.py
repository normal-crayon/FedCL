import flwr as fl
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            # np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights

strategy = SaveModelStrategy()

if __name__ == '__main__':
    fl.server.start_server(server_address = 'localhost:5000', config={'num_rounds': 2}, strategy=strategy)