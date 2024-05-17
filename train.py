import torch
import gpt

model = gpt.GPT()
m = model.to(gpt.device)

optimizer = torch.optim.Adam(model.parameters(), lr=gpt.learning_rate)
for iter in range(gpt.max_iters):
    if iter % gpt.eval_interval == 0 or iter == gpt.max_iters-1:
        losses = gpt.estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = gpt.get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), 'republic_gpt.pt')