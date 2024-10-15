import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Les fonctions pour plot
# pour un temps donné 

def plot_map_chaleur(X,T_reshape,i,vmin=None,vmax=None):
    plt.clf() 
    x_0 = X[0][:,:,i]
    y_0 = X[1][:,:,i]
    T_0 = T_reshape[:,:,i]
    plt.pcolormesh(x_0, y_0, T_0, cmap='coolwarm',shading='gouraud', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Valeur de u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Carte de chaleur à t={X[2][0,0,i]:.2f}')
    
def anim(title, X, T_reshape):    
    # Initialize figure
    fig = plt.figure()
    def animate(frame):
        plot_map_chaleur(X, T_reshape, frame, vmin=T_reshape.min(), vmax=T_reshape.max()) 
        # mettre None vmin et vmax pour que l'échelle bouge 
    ani = FuncAnimation(fig, animate, frames=np.arange(0, T_reshape.shape[2]), repeat=False)
    ani.save(title, writer='pillow', fps=6)
    
    
def plot_points_coloc(liste_points, i) :
    plt.clf()
    scatter = plt.scatter(liste_points[i][:,0], liste_points[i][:,1],
                          c = liste_points[i][:,2], cmap='viridis', marker = '.')
    plt.colorbar(scatter, label='Valeurs de t')
    plt.title(f"Nuage de points à l'ittération {i*100}")
    plt.xlabel("X")
    plt.ylabel("Y")
    
def anim_colocation(title, liste_points):    
    # Initialize figure
    fig = plt.figure()
    def animate(frame):
        print(frame)
        plot_points_coloc(liste_points=liste_points,i=frame) 
        # mettre None vmin et vmax pour que l'échelle bouge 
    ani = FuncAnimation(fig, animate, frames = len(liste_points), repeat=False) 
    #frames=np.arange(0, T_reshape.shape[2]), repeat=False)
    ani.save(title, writer='pillow',fps=2)