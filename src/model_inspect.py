import torch

from agent import DQN
from environment import SnakeEnv

path = "../models/dqn_1.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env =  SnakeEnv()
model = DQN(env, double_dqn=True).online_model
ckpt = torch.load(path, map_location=device)

# ckpt might be just a state_dict OR a dict with keys
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict)

model.to(device)
model.eval()

state = [1,1,1,1,1,1,1,1,1,1,1]
pred = model(torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0))[0]
print(pred.detach().cpu().numpy())