import sumolib
import torch

from helper import get_edge_index, choose_action
from models.GNN_model import GNNModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(model_file="saved_models/GNN/multi_GNN_offline_15.07.2021-15:57.pt",
         net_file="scenarios/medium_grid/u_map.net.xml"):
    model = GNNModel(
        input_dim=18,
        output_dim=2,
        num_layers=1,
        dropout=0.05
    ).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    net = sumolib.net.readNet(net_file)
    edge_index = torch.LongTensor(get_edge_index(net).T).to(device)

    change_obs = torch.zeros(100, 18)
    change_obs[17] = torch.FloatTensor([5]*9 + [0]*9)
    change_obs[18] = torch.FloatTensor([0]*9 + [5]*9)
    action = choose_action(model, change_obs, edge_index)

    assert action[17][1] == 1
    assert action[18][0] == 1
    print(action[17])
    print(action[18])


if __name__ == "__main__":
    main()
