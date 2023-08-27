import numpy as np
import torch
import torch.nn as nn
from models import Generator, Discriminator, gen_loss, disc_loss
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import AHE, collate
import lazy_dataset
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
sw = SummaryWriter()
#load the files
def load_img(example):
    img = cv2.imread(example['image_path'])
    example['image'] = img.astype(np.float32)
    return example

def prepare_dataset(dataset,batch_size=16):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(collate)
    return dataset

path = 'ckpt_latest.pth'
def main():
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5

    db = AHE()
    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    steps = 0
    gen = Generator().to(device)
    disc = Discriminator().to(device)
    gen.train()
    disc.train()
    g_optim = torch.optim.Adam(gen.parameters(),lr=1e-4,betas=(0.5,0.999))
    d_optim = torch.optim.Adam(disc.parameters(),lr=1e-4, betas=(0.5,0.999))

    min_val = 10e7
    running_g_loss = 0
    running_d_loss = 0 
    for epoch in range(250):
        train_ds = prepare_dataset(t_ds)
        valid_ds = prepare_dataset(v_ds,batch_size=1)
        for index,batch in enumerate(tqdm(train_ds)):
            g_optim.zero_grad()
            d_optim.zero_grad()
            images = batch['image']
            images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2) # cv2 loads images as (h,w,3), models take(3,h,w)
            noise = torch.randn(images.shape[0], 1024).to(device)
            fakes = gen(noise)
            p_images = images #for visualisation
            d_fake = disc(fakes)

            #******************* 
            # generator step
        
            #*******************

            loss_g = gen_loss(d_fake)
            loss_g.backward()
            g_optim.step()
            running_g_loss += loss_g.item()
            #*********************  
            
            # discriminator step

            #*********************
            d_real = disc(images)
            d_fake = disc(fakes.detach())
            loss_d = disc_loss(d_real, d_fake)
            loss_d.backward()
            d_optim.step()
            running_d_loss += loss_d.item()
           
            if steps % 1000 == 0:
                gen.eval()
                disc.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_ds):
                        images = batch['image']
                        images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2)
                        noise = torch.randn(images.shape[0], 1024).to(device)

                        fakes = gen(noise)
                        d_fake = disc(fakes.detach())
                        d_real = disc(images)
                        g_loss = gen_loss(d_fake)
                        d_loss = disc_loss(d_real, d_fake)

                print(f'training losses after {steps} batches  {running_g_loss/ (steps + 1)} |--| {running_d_loss/ (steps + 1)}  ')
                sw.add_scalar("training/generator_loss",running_g_loss/(steps + 1),steps)
                sw.add_scalar("training/discriminator_loss",running_d_loss/(steps + 1),steps)


                print(f'validation loss after {steps} batches: {g_loss.item()} |---| {d_loss.item()}')
                sw.add_scalar("validation/generator_loss",g_loss,steps)
                sw.add_scalar("validation/fake_image_prediction",d_fake,steps)
                sw.add_scalar("validation/real_image_prediction",d_real,steps)

                sw.add_scalar("validation/discriminator_loss",d_loss,steps)
                z = torch.randn(16,1024).to(device)
                outs = gen(z)
                sw.add_images("validation/generated_images", outs,steps)
                sw.add_images("validation/real_images", torch.div(p_images,255.0),steps)


                
                torch.save({
                    'steps': steps,
                    'generator': gen.state_dict(),
                    'generator_optimizer': g_optim.state_dict(),
                    'discriminator': disc.state_dict(),
                    'discriminator_optimizer': d_optim.state_dict(),
                    'generator_loss': g_loss,
                    'discriminator_loss': d_loss
                    }, path)
                
            steps +=1
            gen.train()
            disc.train()
  

if __name__== '__main__':
    main()

