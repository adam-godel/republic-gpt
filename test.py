import torch
import gpt

model = gpt.GPT()
m = model.to(gpt.device)
m.load_state_dict(torch.load('republic_gpt.pt', map_location=gpt.device))
context = torch.zeros((1, 1), dtype=torch.long, device=gpt.device)
m.generate(context, max_new_tokens=500)