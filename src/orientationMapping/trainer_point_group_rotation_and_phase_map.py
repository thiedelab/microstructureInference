from orientationMapping.LossFunctions import pointGroup_map_rotation_and_phase_prediction
from orientationMapping.dataModules import cubic_proper_point_group_operations
import torch
import numpy as np
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
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Start from the next epoch
    start_epoch = checkpoint['epoch'] + 1

    # Resume scheduler with last_epoch as the last epoch in the checkpoint
    scheduler.last_epoch = checkpoint['epoch']

    return model, optimizer, scheduler, start_epoch

def train_epoch(model, dataloader, optimizer, device, point_group_op_matrices, PAD = 0):
    model.train()
    losses, geo_error, phase_prediction_error, count = [], 0, 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, (x, y, z)  in  pbar:
        optimizer.zero_grad()
        features = x.to(device)
        labels_r  = y.to(device)
        labels_phase  = z.to(device)
        # print("features\n", features, "\n")
        # print("labels_r\n", labels_r, "\n")
        # print("features.shape", features.shape, "\n")
        # print("labels.shape", labels.shape)
        pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
        # print("pad_mask.shape", pad_mask.shape)
        # print("pad_mask\n", pad_mask ,"\n")
        
        pred = model(features, pad_mask)
        # print("pred", pred)
        loss, rotation_prediction_loss, phase_prediction_loss = pointGroup_map_rotation_and_phase_prediction(pred, labels_r, labels_phase, point_group_op_matrices)
        loss = loss.to(device)

        loss.backward()
        optimizer.step()
        

        losses.append(loss.item())
        phase_prediction_error += phase_prediction_loss
        geo_error += rotation_prediction_loss
        
        count += 1
        # report progress
        if idx>0 and idx%50 == 0:
            pbar.set_description(f'train loss={loss.item():.4f}')
    return np.mean(losses), geo_error/count, phase_prediction_error/count

def train(model, train_loader, test_loader, epochs, optimizer, linear_warmup, cos_decay, num_warmup_epochs, cos_decay_epoch, device, file_path, PAD = 0, start_epoch = 0, save_interval = 10, best_valid_loss = 1000.0):
    point_group_op_matrices = cubic_proper_point_group_operations()
    point_group_op_matrices = point_group_op_matrices.to(device)
    train_error = []
    valid_error = []
    for ep in range(start_epoch, epochs):
        train_loss, train_geodesic, train_phaseLoss = train_epoch(model, train_loader, optimizer, device, point_group_op_matrices, PAD)
        train_error.append(train_loss)
        print("")
        print(f'ep {ep}: tra_loss={train_loss:.7f}, tra_geo_loss={train_geodesic:.7f},  tra_pha_loss={train_phaseLoss:.7f}')

        del train_loss
        del train_geodesic,
        del train_phaseLoss

        torch.cuda.empty_cache()

        val_loss, val_geodesic, val_phaseLoss = evaluate(model, test_loader, device, point_group_op_matrices, PAD)
        print("")
        print(f'ep {ep}: val_loss={val_loss:.4f}, val_geo_loss={val_geodesic:.7f},  val_pha_loss={val_phaseLoss:.7f}')

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
        
def evaluate(model, dataloader, device, point_group_op_matrices, PAD):
    model.eval()
    losses, geo_error, phase_prediction_error, count = [], 0, 0, 0
    with torch.no_grad():
        for x, y, z in dataloader:
            features = x.to(device)
            labels_r  = y.to(device)
            labels_phase  = z.to(device)
            pad_mask = (torch.sum(features, dim = 2) == PAD).view(features.size(0), 1, 1, features.size(1))
            pred = model(features, pad_mask)
            # reshaped_pred = pPd.symmetric_orthogonalization(pred)

            loss, rotation_prediction_loss, phase_prediction_loss = pointGroup_map_rotation_and_phase_prediction(pred, labels_r, labels_phase, point_group_op_matrices)
            loss = loss.to(device)
            

            losses.append(loss.item())
            phase_prediction_error += phase_prediction_loss
            geo_error += rotation_prediction_loss
            count += 1
    return np.mean(losses), geo_error/count, phase_prediction_error/count
