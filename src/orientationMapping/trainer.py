from orientationMapping.LossFunctions import mirrorMap_crossEntropy_loss, composite_objective_criterion
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
#import orientationMapping.dataModules as pPd

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch


def train_epoch(model, dataloader, optimizer, device, PAD = 0):
    model.train()
    losses, mse_rot_error, geo_error, mirror_error, count = [], 0, 0, 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (x, y, z)  in  pbar:
        optimizer.zero_grad()
        features = x.to(device)
        labels_r  = y.to(device)
        labels_m = z.to(device)
        # print("features\n", features, "\n")
        # print("labels_r\n", labels_r, "\n")
        # print("features.shape", features.shape, "\n")
        # print("labels.shape", labels.shape)
        pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
        # print("pad_mask.shape", pad_mask.shape)
        # print("pad_mask\n", pad_mask ,"\n")
        
        pred = model(features, pad_mask)
        # print("pred", pred)
        loss, rotatLoss, mirrorLoss, predicted_rotation_matrix = composite_objective_criterion(pred, labels_r, labels_m)
        loss = loss.to(device)

        loss.backward()
        optimizer.step()
        
        geo_error += rotatLoss
        mirror_error += mirrorLoss

        losses.append(loss.item())
        # acc += (pred.argmax(1) == labels).sum().item()
        mse_rot_error += F.mse_loss(predicted_rotation_matrix, labels_r)
        
        count += 1
        # report progress
        if idx>0 and idx%50 == 0:
            pbar.set_description(f'train loss={loss.item():.4f}, rotLoss={geo_error/count:.7f}, mirLoss={mirror_error/count:.7f}, rotMSE={mse_rot_error/count:.7f}')
    return np.mean(losses), geo_error/count, mirror_error/count, mse_rot_error/count

def train(model, train_loader, test_loader, epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch, device, file_path, PAD = 0, start_epoch = 0, save_interval = 10, best_valid_loss = 1000.0):
    train_error = []
    valid_error = []
    for ep in range(start_epoch, epochs):
        train_loss, train_rotLoss, train_mirrLoss, train_mse_rot = train_epoch(model, train_loader, optimizer, device, PAD)
        train_error.append(train_loss)
        print("")
        print(f'ep {ep}: tra_loss={train_loss:.7f}, tra_geod={train_rotLoss:.7f}, tra_mirr={train_mirrLoss:.7f}, tra_rotMSE={train_mse_rot:.7f}')

        del train_loss, train_rotLoss, train_mirrLoss, train_mse_rot

        torch.cuda.empty_cache()

        val_loss, val_rotLoss, val_mirrLoss, val_mse_rot = evaluate(model, test_loader, device, PAD)
        print("")
        print(f'ep {ep}: val_loss={val_loss:.4f}, val_geod={val_rotLoss:.7f}, val_mirr={val_mirrLoss:.7f}, val_rotMSE={val_mse_rot:.7f}')

        valid_error.append(val_loss)

        # update scheduler
        if ep < num_warmup_epochs:
            linear_warmup.step()
            print("linear_warmup.get_last_lr()", linear_warmup.get_last_lr())
        elif ep >= num_warmup_epochs:
            if ep < cos_decay_epoch + num_warmup_epochs:
                cos_decay.step()
                print("cos_decay.get_last_lr()", cos_decay.get_last_lr())
            else:
                for param_group in optimizer.param_groups:
                    print("learning rate: ", param_group['lr'])
                    
        
        # save checkpoint
        if val_loss < best_valid_loss:
            checkpoint_path = os.path.join(file_path, "best_model.pth")
            save_checkpoint(model, optimizer, cos_decay, ep, checkpoint_path)
            print("")
            print("ep", ep, " val_loss", val_loss, ", new best model saved")
            print("")
            
            best_valid_loss = val_loss
        if ep % save_interval == 0:
            interval_checkpoint_path = os.path.join(file_path, f"model_epoch_{ep}_valEr_{val_loss:.7f}.pth")
            save_checkpoint(model, optimizer, cos_decay, ep, interval_checkpoint_path)
    return train_error, valid_error
        
def evaluate(model, dataloader, device, PAD):
    model.eval()
    losses, mse_rot_error, geo_error, mirror_error, count = [], 0, 0, 0, 0
    with torch.no_grad():
        for x, y, z in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            labels_m = z.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            pred = model(features, pad_mask)
            # reshaped_pred = pPd.symmetric_orthogonalization(pred)

            loss, rotatLoss, mirrorLoss, predicted_rotation_matrix = composite_objective_criterion(pred, labels_r, labels_m)
            loss = loss.to(device)
            
            geo_error += rotatLoss
            mirror_error += mirrorLoss

            # acc += (pred.argmax(1) == labels).sum().item()
            mse_rot_error += F.mse_loss(predicted_rotation_matrix, labels_r)

            # loss = loss_fn(reshaped_pred, labels).to(device)
            losses.append(loss.item())
            # acc = (pred.argmax(1) == labels).sum().item()
            #sign_error += pPd.sign_consistency_loss(pred, labels)
            count += 1
    return np.mean(losses), geo_error/count, mirror_error/count, mse_rot_error/count
