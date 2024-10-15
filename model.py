from deepxrte.gradients import gradient, derivee_seconde
import torch 
import torch.nn as nn 

alpha = 1.2

def pde(T, x_input):
    """Calcul la pde 

    Args:
        T (_type_): la température calculée avec NN
        x_input (_type_): l'input (x,y,t)
    """
    T_t = gradient(T, x_input, i=0, j=2, keep_gradient=True)
    T_xx = derivee_seconde(T, x_input, j=0)
    T_yy = derivee_seconde(T, x_input, j=1)
    return T_t - alpha * (T_xx + T_yy) # l'équation de la chaleur
    
    
## Le NN 

class PINNs(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)  # Couche d'entrée avec 2 neurones d'entrée et 16 neurones cachés
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fcf = nn.Linear(16, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fcf.weight)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fcf(x)
        return x # Retourner la sortie