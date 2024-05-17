import os
import time
import tokenizer as tk

text = open("input.txt", "r", encoding="utf-8").read()
os.makedirs("models", exist_ok=True)

t0 = time.time()
tokenizer = tk.Tokenizer()
tokenizer.train(text, 768, verbose=True)
tokenizer.save('tokenizer')
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
