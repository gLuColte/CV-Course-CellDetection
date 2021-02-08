import cv2 as cv
import numpy as np
import os
import torch
import torch.nn as tnn

from datetime import datetime
from torch.utils.data import DataLoader

from src.unet import UNetLite
from src.dataset import random_transform, CellDataset
from src.preprocess import preprocess_dic, preprocess_fluo, preprocess_phc


def status_string(it, start_time, loss):
    s = f"Iteration: {it + 1}    " \
        f"Time: {datetime.now() - start_time}".split(".")[0] + "    " \
        f"Loss: {loss}"

    return s


def train(model_path=None, max_iter=2000, lr=0.0007, save_interval=500, 
          report_interval=50, prep_func=None):
    # Choose the device used to train the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Open the log file
    f = open("log.txt", "w")

    # Load the training set
    train_set = CellDataset(transform=random_transform, preprocess=prep_func)
    train_loader = DataLoader(train_set, batch_size=3, shuffle=True, 
                              num_workers=1, drop_last=True)

    # Specify the network parameters, loss function, and optimizer
    net = UNetLite(scale=0.6, double=True)
    seg_loss = tnn.BCEWithLogitsLoss(reduction='none')
    bnd_loss = tnn.BCEWithLogitsLoss(reduction='none')
    dst_loss = tnn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # If a model was provided, train on that
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    
    # Send the model to the training device
    net.to(device)
    net.train()

    # Print the total number of iterations to train on
    print(f"Total iterations: {max_iter}")
    
    # Initialize some variables
    start_time = datetime.now()

    running_loss_save = 0
    running_loss_report = 0
    it = 0
    
    while (it < max_iter):
        for _, batch in enumerate(train_loader): 
            # Increment the iteration count
            it += 1

            # Reset all gradients to 0
            optimizer.zero_grad()
            
            # Load the sample data
            nrm = batch["nrm"].to(device)
            seg = batch["seg"].to(device)
            msk = batch["msk"].to(device)
            bnd = batch["bnd"].to(device)
            dst = batch["dst"].to(device)

            # Calculate the network output
            out = net(nrm)
            out_seg = out[:, 0, :, :].unsqueeze(1)
            out_dst = out[:, 1, :, :].unsqueeze(1)

            # Calculate the loss
            loss1 = seg_loss(out_seg, seg) * msk
            loss2 = bnd_loss(out_seg, seg) * bnd
            loss3 = dst_loss(out_dst, dst) * msk
            loss = (loss1.sum() / msk.sum()) \
                 + 0.5 * (loss2.sum() / bnd.sum()) \
                 + 0.001 * (loss3.sum() / msk.sum())
            
            # Save the loss values
            running_loss_save += loss.item()
            running_loss_report += loss.item()

            # Run gradient descent
            loss.backward()
            optimizer.step()
            
            if ((it + 1) % save_interval == 0):
                # Print to the log
                s = status_string(it, start_time, 
                                  running_loss_save / save_interval)
                f.write(s + "\n")
                f.flush()
                os.fsync(f)
                
                # Save the current model
                save_path = f"models/model_{it + 1:06}.pth"
                torch.save(net.state_dict(), save_path)

                # Reset the running loss
                running_loss_save = 0
                
            elif ((it + 1) % report_interval == 0):
                # Print to the log
                s = status_string(it, start_time, 
                                  running_loss_report / report_interval)
                f.write(s + "\n")
                f.flush()
                os.fsync(f)
                
                # Reset the running loss
                running_loss_report = 0
            
            # Print the current iteration's training loss
            s = status_string(it, start_time, loss.item())
            print(s, end="\r")
    
    save_path = f"models/model_{max_iter:06}.pth"
    torch.save(net.state_dict(), save_path)
    s = status_string(it, start_time, loss.item())

    print(s)

    f.close()
            
    return net


def test(model_path=None, prep_func=None, cpu=False):
    with torch.no_grad():
        # Select the device to run on
        if cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load the model from a file
        net = UNetLite(scale=0.6, double=True)
        if model_path is not None:
            load_path = model_path
        else:
            load_path = "models/" + max(os.listdir("models"))
            
        net.load_state_dict(torch.load(load_path))
        net.to(device)
        net.eval()
        
        # Set the output folder
        out_path = "output/test/"
        
        # Create a dataloader for the test set
        test_set = CellDataset(preprocess=prep_func, normalize=True)
        test_loader = DataLoader(test_set, batch_size=1)
    
        curr = 0
        for i, batch in enumerate(test_loader):
            imgs = batch["img"]
            segs = batch["seg"]
            nrms = batch["nrm"]

            output = net(nrms.to(device))
    
            output = output.cpu()
    
            for j in range(imgs.shape[0]):
                curr += 1
                img = imgs.numpy()[j, 0, :, :].astype(np.uint8)

                out_seg = output.numpy()
                out_seg = (out_seg > 0)[j, 0, :, :].astype(np.uint8)

                out_dst = output.numpy()[j, 1, :, :]
                out_dst[out_dst > 255] = 255
                out_dst[out_dst < 0] = 0
                out_dst = out_dst.astype(np.uint8)
    
                cmp = np.concatenate((img, out_seg * 255, out_dst), axis=1)
    
                cv.imwrite(f"{out_path}/{curr}.png", cmp)
            
            
if __name__ == "__main__":
    prep_func = preprocess_phc

    train(prep_func=prep_func)
    test(prep_func=prep_func)
