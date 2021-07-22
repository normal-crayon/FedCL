#!usr/bin/myenv python
import flwr as fl

fl.server.start_server( "localhost:5000",config={"num_rounds": 2})
