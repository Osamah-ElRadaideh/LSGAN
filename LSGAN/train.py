import numpy as np
import torch
import torch.nn as nn
from models import Generator, Discriminator, gen_loss, disc_loss, compute_gp
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2
from utils import AHE, collate
import torch.nn.functional as F
import lazy_dataset
from sacred import Experiment
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device set to : {device}')

ex = Experiment('LSGAN', save_git_info=False)
sw = SummaryWriter()

@ex.config
def defaults():
    batch_size = 16
    g_lr = 0.0001
    d_lr = 0.00005
    steps_per_eval = 1000
    max_steps = 150_000
    latent_size = 256
#load the files
def load_img(example):
    img = cv2.imread(example['image_path'])
    example['image'] = img.astype(np.float32) / 255.0
    return example

@ex.capture
def prepare_dataset(dataset,batch_size):
    if isinstance(dataset,list):
        dataset = lazy_dataset.new(dataset)
    dataset = dataset.map(load_img)
    dataset = dataset.shuffle()
    dataset = dataset.batch(batch_size=batch_size, drop_last=True)
    dataset = dataset.map(collate)
    return dataset

path = 'ckpt_latest.pth'
@ex.automain
def main(batch_size,d_lr,g_lr, steps_per_eval, latent_size,max_steps):
    #model hyperparamters
    #per the LSGAN paper, beta1 os set to 0.5

    db = AHE()
    t_ds = db.get_dataset('training_set')
    v_ds = db.get_dataset('validation_set')
    steps = 0
    gen = Generator(latent_size=latent_size).to(device)
    disc = Discriminator().to(device)
    gen.train()
    disc.train()
    g_optim = torch.optim.Adam(gen.parameters(),lr=g_lr)
    d_optim = torch.optim.Adam(disc.parameters(),lr=d_lr)

    for epoch in range(10000):
        epoch_g_loss = 0
        epoch_d_loss = 0
        train_ds = prepare_dataset(t_ds, batch_size=batch_size)
        valid_ds = prepare_dataset(v_ds, batch_size=1)
        for _,batch in enumerate(tqdm(train_ds)):
            g_optim.zero_grad()
            d_optim.zero_grad()
            images = batch['image']
            images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2) # cv2 loads images as (h,w,3), models take(3,h,w)
            noise = torch.randn(images.shape[0], latent_size).to(device)
            p_images = images #for visualisation
            fakes = gen(noise)

            #******************* 
            # generator step
        
            #*******************
            fakeouts, g_fm = disc(fakes)
            _, d_fm = disc(images)
            fm_loss = 0
            for g,d in zip(g_fm,d_fm):

                fm_loss += torch.mean(torch.abs(g - d))
            loss_g =  fm_loss * 10 + gen_loss(fakeouts)
            loss_g.backward()
            g_optim.step()
            epoch_g_loss += loss_g.item()
               

            #*********************  
            
            # discriminator step

            #*********************
            d_fake ,_ = disc(fakes.detach())
            d_real,_ = disc(images)
            loss_d = disc_loss(d_real, d_fake) 
            loss_d.backward()
            d_optim.step()
            epoch_d_loss += loss_d.item()
   


            
        
           
            if steps % steps_per_eval == 0:
                gen.eval()
                disc.eval()
                with torch.no_grad():
                    for batch in tqdm(valid_ds[0:1]):
                        images = batch['image']
                        images = torch.tensor(np.array(images)).to(device).permute(0,3,1,2)
                        noise = torch.randn(images.shape[0], latent_size).to(device)

                        fakes = gen(noise)
                        d_fake, _ = disc(fakes.detach())
                        d_real, _ = disc(images)
                        g_loss = gen_loss(d_fake)
                        d_loss  = disc_loss(d_real, d_fake)

                


                    print(f'validation loss after {steps} batches: {g_loss.item()} |---| {d_loss.item()}')
                    sw.add_scalar("validation/generator_loss",g_loss,steps)
                    sw.add_scalar("validation/fake_image_prediction",d_fake,steps)
                    sw.add_scalar("validation/real_image_prediction",d_real,steps)

                    sw.add_scalar("validation/discriminator_loss",d_loss,steps)
                    z = torch.randn(16,latent_size).to(device)
                    outs = gen(z)
                    sw.add_images("validation/generated_images", outs,steps)
                    sw.add_images("validation/real_images", p_images,steps)


                
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
        sw.add_scalar("training/generator_loss",epoch_g_loss/len(train_ds),epoch)
        sw.add_scalar("training/discriminator_loss",epoch_d_loss/len(train_ds),epoch)
        if steps ==   max_steps:
            print('maximum steps reached....stopping training loop')
            break
