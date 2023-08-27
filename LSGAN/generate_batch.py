from models import Generator
import torch
import cv2 
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Generator().to(device)
states = torch.load('ckpt_latest.pth')
model.load_state_dict(states['generator'])
model.eval()
img_dir = 'generated_batch'
os.makedirs(img_dir,exist_ok=True)
with torch.inference_mode():
    z = torch.randn(16,1024).to(device)
    outputs = model(z).cpu()
    for index,output in enumerate(outputs):
        cv2.imwrite(f'{img_dir}\\gen_{index}.png', output.permute(1, 2, 0).numpy() * 255)

