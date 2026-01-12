from orientationMapping.LossFunctions import pointGroup_map_rotation_prediction
from orientationMapping.dataModules import cubic_proper_point_group_operations
import torch
import numpy as np
from tqdm import tqdm
import os
#import orientationMapping.dataModules as pPd

def save_checkpoint(model, optimizer, linear_warmup, cos_decay, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'linear_warmup_state_dict': linear_warmup.state_dict(),
        'cos_decay_state_dict': cos_decay.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    
def load_checkpoint(model, optimizer, linear_warmup, cos_decay, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    linear_warmup.load_state_dict(checkpoint['linear_warmup_state_dict'])
    cos_decay.load_state_dict(checkpoint['cos_decay_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, linear_warmup, cos_decay, start_epoch

def train_epoch(model, dataloader, optimizer, device, point_group_op_matrices, PAD = 0):
    model.train()
    losses,  count = [], 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (x, y)  in  pbar:
        optimizer.zero_grad()
        features = x.to(device)
        labels_r  = y.to(device)
        # print("features\n", features, "\n")
        # print("labels_r\n", labels_r, "\n")
        # print("features.shape", features.shape, "\n")
        # print("labels.shape", labels.shape)
        pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
        # print("pad_mask.shape", pad_mask.shape)
        # print("pad_mask\n", pad_mask ,"\n")
        
        pred = model(features, pad_mask)
        # print("pred", pred)
        loss = pointGroup_map_rotation_prediction(pred, labels_r, point_group_op_matrices)
        loss = loss.to(device)

        loss.backward()
        optimizer.step()
        

        losses.append(loss.item())
        # acc += (pred.argmax(1) == labels).sum().item()
        
        count += 1
        # report progress
        if idx>0 and idx%50 == 0:
            pbar.set_description(f'train loss={loss.item():.4f}')
    return np.mean(losses)

def train(model, train_loader, test_loader, epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch, device, file_path, PAD = 0, start_epoch = 0, save_interval = 5, best_valid_loss = 1000.0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    train_error = []
    valid_error = []
    for ep in range(start_epoch, epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, point_group_op_matrices, PAD)
        train_error.append(train_loss)
        print("")
        print(f'ep {ep}: tra_loss={train_loss:.7f}')

        del train_loss

        torch.cuda.empty_cache()

        val_loss = evaluate(model, test_loader, device, point_group_op_matrices, PAD)
        print("")
        print(f'ep {ep}: val_loss={val_loss:.4f}')

        valid_error.append(val_loss)

        # update scheduler
        if ep < num_warmup_epochs:
            linear_warmup.step()
            print("linear_warmup.get_last_lr()", linear_warmup.get_last_lr())
        else:
            cos_decay.step()
            print("cos_decay.get_last_lr()", cos_decay.get_last_lr())
            # if ep < cos_decay_epoch + num_warmup_epochs:
            #     cos_decay.step()
            #     print("cos_decay.get_last_lr()", cos_decay.get_last_lr())
            # else:
            #     for param_group in optimizer.param_groups:
            #         print("learning rate: ", param_group['lr'])
                    
        
        # save checkpoint
        if val_loss < best_valid_loss:
            checkpoint_path = os.path.join(file_path, "best_model.pth")
            save_checkpoint(model, optimizer, linear_warmup, cos_decay, ep, checkpoint_path)
            print("")
            print("ep", ep, " val_loss", val_loss, ", new best model saved")
            print("")
            
            best_valid_loss = val_loss

        the_most_recent_model_path = os.path.join(file_path, "last_updated_model.pth")
        save_checkpoint(model, optimizer, linear_warmup, cos_decay, ep, the_most_recent_model_path)
    return train_error, valid_error
        
def evaluate(model, dataloader, device, point_group_op_matrices, PAD):
    model.eval()
    losses, count = [], 0
    with torch.no_grad():
        for x, y in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            pred = model(features, pad_mask)
            # reshaped_pred = pPd.symmetric_orthogonalization(pred)

            loss = pointGroup_map_rotation_prediction(pred, labels_r, point_group_op_matrices)
            loss = loss.to(device)
            

            # loss = loss_fn(reshaped_pred, labels).to(device)
            losses.append(loss.item())
            # acc = (pred.argmax(1) == labels).sum().item()
            #sign_error += pPd.sign_consistency_loss(pred, labels)
            count += 1
    return np.mean(losses)
