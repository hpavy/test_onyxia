from deepxrte.geometry import Rectangle
import torch 
import torch.nn as nn 
import torch.optim as optim
from matplotlib.animation import FuncAnimation
from model import PINNs
from utils import read_csv, write_csv
from train import train
from pathlib import Path

############# LES VARIABLES ################

folder_result = 'test_piche' # le nom du dossier de résultat

##### Le modèle de résolution de l'équation de la chaleur
nb_itt=2000      # le nb d'epoch
resample_rate=500 # le taux de resampling
display=500       # le taux d'affichage
poids = [1,1,1]   # les poids pour la loss

t_min = 0
t_max = 0.3
x_max = 1
y_max = 1

    
n_bc = 152         # le nb de points sur le bord
n_ic = 152         # le nb de points initiaux
n_pde = 700        # le nb de points pour la pde




##### Le code ###############################
###############################################

# On regarde si le dossier existe 
dossier = Path(folder_result)
dossier.mkdir(parents=True, exist_ok=True)

rectangle = Rectangle(x_max = x_max, y_max = y_max,
                      t_min=t_min, t_max=t_max)    # le domaine de résolution

# les points initiaux du train 
points_bc = rectangle.generate_border(n_bc)
points_pde = rectangle.generate_random(n_pde)
points_ic = rectangle.generate_random(n_ic, init=True)

def init_condition(x):
    return (torch.sin(torch.pi*x[:,0:1])*torch.sin(torch.pi*x[:,1:2]))

### Pour test
x_test_pde = rectangle.generate_random(700)
x_test_ic = rectangle.generate_random(152, init=True)
x_test_bc =  rectangle.generate_border(152)
y_test_ic = init_condition(x_test_ic[:,0:2])
y_test_bc = torch.zeros(152,1)


# Initialiser le modèle
model = PINNs()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss = nn.MSELoss()

# On plot les print dans un fichier texte 
with open(folder_result+'/print.txt', 'a') as f:
    # On regarde si notre modèle n'existe pas déjà 
    if Path(folder_result+'/model_weights.pth').exists() :
        model.load_state_dict(torch.load(folder_result+'/model_weights.pth'))
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        train_loss = read_csv(folder_result+'/train_loss.csv')['0'].to_list()
        test_loss = read_csv(folder_result+'/test_loss.csv')['0'].to_list()
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
        
    else : 
        print('Nouveau modèle\n', file=f)
        print('Nouveau modèle\n')
        train_loss = []
        test_loss = []


    ######## On entraine le modèle 
    ###############################################
    train(nb_itt=nb_itt, train_loss=train_loss, test_loss=test_loss,
            points_pde=points_pde, points_ic=points_ic,
            points_bc=points_bc, init_condition=init_condition,
            rectangle=rectangle, model=model, n_bc=n_bc, 
            loss=loss, optimizer=optimizer, x_test_pde=x_test_pde,
            x_test_ic=x_test_ic, x_test_bc=x_test_bc, y_test_ic=y_test_ic,
            y_test_bc=y_test_bc, resample_rate=resample_rate,
            display=display, poids=poids, n_ic=n_ic, n_pde=n_pde, f=f)

    ####### On save le model et les losses
    torch.save(model.state_dict(), folder_result+'/model_weights.pth')
    write_csv(train_loss, folder_result, file_name='/train_loss.csv')
    write_csv(test_loss, folder_result, file_name='/test_loss.csv')