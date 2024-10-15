import torch 
from model import pde
import numpy as np



def train(nb_itt, train_loss, test_loss,
          points_pde, points_ic,
          points_bc, init_condition,
          rectangle, model, n_bc, 
          loss, optimizer, x_test_pde, x_test_ic,
          x_test_bc, y_test_ic, y_test_bc,
          resample_rate, display,
          poids, n_ic, n_pde, f): 
    nb_it_tot = nb_itt +len(train_loss)
    print(f'--------------------------\nStarting at epoch: {len(train_loss)}\n--------------------------',
          file=f)
    print(f'--------------------------\nStarting at epoch: {len(train_loss)}\n--------------------------')
    # Les datas qu'on utilise 
    ### Pour train
    x_input_pde = points_pde
    x_input_ic = points_ic
    x_input_bc = points_bc  
    y_ic = init_condition(points_ic[:,0:2])
    y_bc = torch.zeros(n_bc,1)
    
    for epoch in range(len(train_loss), nb_it_tot) : 
        model.train() # on dit qu'on va entrainer (on a le dropout)

        # loss du pde
        pred_pde = model(x_input_pde)
        loss_pde = torch.mean(pde(pred_pde, x_input_pde)**2) #(MSE)

        # loss des ic 
        pred_ic = model(x_input_ic)
        loss_ic = loss(pred_ic, y_ic) #(MSE)

        # loss des bc
        pred_bc = model(x_input_bc)
        loss_bc = loss(pred_bc, y_bc)

        # loss totale
        loss_totale = poids[0]*loss_pde + poids[1]*loss_bc + poids[2]*loss_ic
        train_loss.append(loss_totale.item())
        
        # Backpropagation
        loss_totale.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
        # Pour le test :
        model.eval()
        
        # loss du pde
        test_pde = model(x_test_pde)
        loss_test_pde = torch.mean(pde(test_pde, x_test_pde)**2) #(MSE)
        
        # loss des ic 
        test_ic = model(x_test_ic)
        loss_test_ic = loss(test_ic, y_test_ic) #(MSE)

        # loss des bc
        test_bc = model(x_test_bc)
        loss_test_bc = loss(test_bc, y_test_bc)

        # loss totale
        loss_test = poids[0]*loss_test_pde + poids[1]*loss_test_bc + poids[2]*loss_test_ic
        test_loss.append(loss_test.item())

        if ((epoch+1) % display == 0) or (epoch+1 == nb_it_tot):
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :",
                  file=f)
            print(f"---------------------\nEpoch {epoch+1}/{nb_it_tot} :")
            print(f"Train : loss: {np.mean(train_loss[-display:]):.3e}, pde: {loss_pde:.3e},"+
                  f"bc: {loss_bc:.3e}, ic: {loss_ic:.3e}",
                  file=f)
            print(f"Train : loss: {np.mean(train_loss[-display:]):.3e}, pde: {loss_pde:.3e},"+
                  f"bc: {loss_bc:.3e}, ic: {loss_ic:.3e}")
            print(f"Test  : loss: {np.mean(test_loss[-display:]):.3e}, pde: {loss_test_pde:.3e},"+
                  f"bc: {loss_test_bc:.3e}, ic: {loss_test_ic:.3e}",
                  file=f)
            print(f"Test  : loss: {np.mean(test_loss[-display:]):.3e}, pde: {loss_test_pde:.3e},"+
                  f"bc: {loss_test_bc:.3e}, ic: {loss_test_ic:.3e}")
            
        if  (epoch <= 4) :
            print(f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss[-1]:.3e}, pde:{loss_pde:.3e},"+
                  f"bc: {loss_bc:.3e}, ic:{loss_ic:.3e}",
                  file=f)
            print(f"Epoch: {epoch+1}/{nb_it_tot}, loss: {train_loss[-1]:.3e}, pde:{loss_pde:.3e},"+
                  f"bc: {loss_bc:.3e}, ic:{loss_ic:.3e}")
        
        if (epoch+1) % resample_rate == 0:
            # On ressample les points
            x_input_pde = rectangle.generate_random(n_pde)
            x_input_ic =  rectangle.generate_random(n_ic, init=True)
            x_input_bc = rectangle.generate_border(n_bc)  
            y_ic = init_condition(x_input_ic[:,0:2])
            y_bc = torch.zeros(n_bc,1)
            