from helper_evaluate import compute_epoch_loss_autoencoder
from collections import OrderedDict
import time
import torch
import torch.nn.functional as F
import json


def train_vae_v1(num_epochs, model, optimizer, device, 
                 train_loader, loss_fn=None,
                 logging_interval=100, 
                 skip_epoch_stats=False,
                 reconstruction_term_weight=1,
                 save_model=None):
    print('starting training')
    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': [],
                'train_total_loss_per_epoch': [],       
                'total_reconstruction_loss_epoch': [],  
                'total_kl_loss_epoch': []  
                }
    
    

    if loss_fn is None:
        loss_fn = F.mse_loss

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss_epoch = 0.0      
        total_reconstruction_loss_epoch = 0.0
        total_kl_loss_epoch = 0.0  
        
        for batch_idx, features in enumerate(train_loader):
            features = features.to(device)

            # FORWARD AND BACK PROP
            encoded, z_mean, z_log_var, decoded = model(features)
            
            kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      axis=1)
            batchsize = kl_div.size(0)
            kl_div = kl_div.mean()
    
            pixelwise = loss_fn(decoded, features, reduction='none')
            pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)
            pixelwise = pixelwise.mean()
            
            loss = reconstruction_term_weight * pixelwise + kl_div
            
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
        
            total_loss_epoch += loss.item() 
            total_reconstruction_loss_epoch += reconstruction_term_weight * pixelwise.item() 
            total_kl_loss_epoch += kl_div.item() 
             
            # LOGGING
            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div.item())
            
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'
                      % (epoch+1, num_epochs, batch_idx,
                          len(train_loader), loss))
                

        if not skip_epoch_stats:
            model.eval()
            
            with torch.set_grad_enabled(False):  # save memory during inference
                train_loss = compute_epoch_loss_autoencoder(
                    model, train_loader, loss_fn, device)
                print('***Epoch: %03d/%03d | Loss: %.3f | Reconstruction Loss: %.3f | KL Divergence: %.3f' % (
                      epoch+1, num_epochs, total_loss_epoch / len(train_loader), 
                      total_reconstruction_loss_epoch / len(train_loader),
                      total_kl_loss_epoch / len(train_loader)))
                log_dict['train_combined_loss_per_epoch'].append(train_loss.item())
                
                # Store and print epoch-level losses
                log_dict['train_total_loss_per_epoch'].append(total_loss_epoch / len(train_loader))
                log_dict['total_reconstruction_loss_epoch'].append(total_reconstruction_loss_epoch / len(train_loader))
                log_dict['total_kl_loss_epoch'].append(total_kl_loss_epoch / len(train_loader))

        print('Time elapsed: %.2f min' % ((time.time() - start_time) / 60))
    
    print('Total Training Time: %.2f min' % ((time.time() - start_time) / 60))
    if save_model is not None:
        torch.save(model.state_dict(), save_model)
    
    return log_dict
