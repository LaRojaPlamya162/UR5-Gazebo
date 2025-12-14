import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state):
        return self.net(state)
def cut_back_attribute(example, col_name, n):
#def cut_back_attribute(example):
        example[col_name] = example[col_name][:n]
        #example["action"] = example["action"][:6]
        return example 

if __name__ == '__main__':
    dataset = load_dataset("lerobot/berkeley_autolab_ur5", split = "train")
    dataset = dataset.with_format("torch")
    dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "action", "n" : 6})
    dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "observation.state", "n" : 6})
    #dataset = dataset.map(cut_back_atatribute, "action")
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = 256,
        shuffle = True,
        num_workers = 2,
        pin_memory = True,
        drop_last = True
    )
    model = BCPolicy(state_dim = len(dataset[0]['observation.state']),
                     action_dim = len(dataset[0]['action']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
        #for i in range(len(dataset)):
            state = batch['observation.state']
            action = batch['action']

            #state = dataset[i]['observation.state']
            #action = dataset[i]['action']
            pred = model(state)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(dataloader):.6f}")
    
    print("Training finised")
    #torch.save(model.state_dict(),"ur_gazebo/src/Universal_Robots_ROS2_Gazebo_Simulation/ur_simulation_gazebo/model/param/BCmodel.pth")
    #torch.save(model.state_dict(), './my_controller/bc_model.pth')
    torch.save(model.state_dict(), 'bc_model_v2.pth')
    #torch.save(model.state_dict(), "my_controller/bc_model/bc_model.pth")
    print("Model loaded!")