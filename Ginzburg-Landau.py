#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.integrate import odeint #importando do módulo para simular equações diferenciais o integrador odeint
import numpy as np #pacote básico da linguagem Python que permite trabalhar, vetores e matrizes de N dimensões (parece matlab)
import random 
import matplotlib.pyplot as plt #modulo para geração de gráficos
from math import exp


# In[2]:


# N Conjunto de equações acopladas
# vetor de N componentes, sendo i a posição no vetor e u(i,t) = x(i,t), u(i+N,t) = y(i,t) , ..., u(i+(Ne-1)*N,t)=z(i,t)



#________________________________________________Definindo parâmetros__________________________________________________________
#_______Parâmetros da integração______
N=501 #número de sítios
Ne=2 #qtd. de eqs. de 1º ordem em cada sítio, se mexer aqui também de incluir eq. nas condi. de contorno e acoplamento
gamma=10
vizinhotimos=2
tinicial=0
tfinal=500
step=0.02
Npontos= int(tfinal/step) #quantidade de tempos iterados para obter a solução, que é associado ao número de linhas da matriz solução
t = np.linspace(tinicial, tfinal, Npontos)
#_____________________________________
#_______Parâmetros das equações_______
w0 = 0
μ =  1
a = 1.5     # Simulação 2 a =  2 # Simulação 3 a =  1.0 # Simulação  4 a =  1.4  #  Simulação 5 a =  1.00 # Simulação 6 =  1.0 
b = -1    # Simulação 2 b = -2 # Simulação 3 b = -1.3 # Simulação  4 b = -0.6  #  Simulação 5 b = -1.35 # Simulação 6 = -1.4
D0 = a
D1 = a
#_____________________________________
#________________________________________________Condições Iniciais____________________________________________________________     
u0 = np.zeros(Ne*N) # define um vetor vazio com Ne*N componentes, para as c.i.
for i in range(0, Ne*N, 1) : 
#    u0[i]=random.random(0,9) # c.i. randômicas no intervalo (min,max)      
     u0[i] = 0.5 #todas as posições do vetor de 'Condições Iniciais' receberão atribuição (0.1)

u0[48] = 0.0001
u0[49] = 0 
u0[50] = 0

#imprimindo as condições iniciais
#print("Quantidade de Equações" ,len(u0))
#for iteste in range(0, Ne*N, 1) : 
#    print( "indice %d = %f"% (iteste, u0[iteste]) ) #depois do % se aparecer: d=>inteiro e f=float

#___________________Constante de Normalização e Otimização___________________________
#Otimização para casos de predominancia difusiva
if gamma>5:
    Nvizinhos=int(vizinhotimos)
else:
    Nvizinhos=int((N-1)/2)

#Constante de Normalização
norma=0
for j in range(1, Nvizinhos+1, 1) : 
    norma=norma+2/(exp(j*gamma))
#_____________________________________________________________________________________
#________________________________________________Definindo Equações____________________________________________________________
def fun(u, t):
    dudt = np.zeros(Ne*N) # define um vetor vazio, para as derivadas
    
    
#________________________Condições de contorno periódicas____________________________    
    contor0= np.zeros(3*N) # vetor para condições de contorno peródicas da 1° variável
    contor1= np.zeros(3*N) # vetor para condições de contorno peródicas da 2° variável
    for i0 in range(0,N,1):
        contor0[i0]=u[i0]   
        contor0[i0+N]=u[i0]
        contor0[i0+2*N]=u[i0]
        contor1[i0]=u[i0+N]   
        contor1[i0+N]=u[i0+N]
        contor1[i0+2*N]=u[i0+N]

#_____________________________Cálculo do acoplamento lei de potência________________________  
    acopl0= np.zeros(N) # vetor de acoplamento da 1° variável
    acopl1= np.zeros(N) # vetor de acoplamento da 2° variável
    for i1 in range(0,N,1):
        for i2 in range(1, Nvizinhos+1, 1) :
            acopl0[i1]=(contor0[N+i1-i2]+contor0[N+i1+i2])/(exp(i2*gamma))+acopl0[i1]
            acopl1[i1]=(contor1[N+i1-i2]+contor1[N+i1+i2])/(exp(i2*gamma))+acopl1[i1]        
        
#_____________________Entrando com as equações__________________________
    for i in range(0,N,1): 
        dudt[i] =    μ*u[i] - ( (  u[i])**2 + (u[i+N])**2 )*u[i]  + b*(( (u[i])**2 + (u[i+N])**2)*u[i+N])   +   (acopl0[i]/norma-u[i])   -  (D1)*(acopl1[i]/norma-u[i+N])
        dudt[i+N] =  μ*u[i+N]   - b*(( (u[i])**2 + (u[i+N])**2 )*u[i])  -   ( (u[i])**2 + (u[i+N])**2)*u[i+N]   + (D0)*(acopl0[i]/norma-u[i])   +  (acopl1[i]/norma-u[i+N])
    
#_______________________________________________________________________

    return dudt
#______________________________________________________________________________________________________________________________


sol = odeint(fun, u0, t, rtol = 1.0e-10, atol = 1.0e-10)


# In[3]:


plt.figure(1)
plt.title("Séries Temporais")
plt.plot(t, sol[:, 18], 'b', label='X1') 
plt.xlabel('t')
plt.axis([300,tfinal, -1, 1])
plt.grid()
plt.show()

#Perfil Espacial 
plt.figure(2)
plt.title("Perfil Espacial")
plt.plot(sol[Npontos-1,0:N])#plota os valores da 1º variável da ultima linha do vetor solução, que é o ultimo tempo de integração, assim imprime o perfil espacial da rede.
plt.xlim(0,501)
#plt.ylim(0.85,1.05)
plt.xlabel("k (sítios)")
plt.ylabel("X0_final")

plt.figure(3)
plt.title("Perfil Espacial")
plt.plot(sol[Npontos-1,N:2*N+1])#plota os valores da 2º variável da ultima linha do vetor solução, que é o ultimo tempo de integração, assim imprime o perfil espacial da rede.
plt.xlim(0,101)
#plt.ylim(0.85,1.05)
plt.xlabel("k (sítios)")
plt.ylabel("Y0_final")


# In[ ]:


#Imprimindo os Graficos 3D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
iterada = np.arange(20000, 25000,1)  #se quiser alterar o intervalo de pontos plotados pode ser feito por aqui
#iterada = np.arange(0, Npontos-1,100)
sitios = np.arange(0, N,1)
iterada, sitios = np.meshgrid(iterada, sitios)
X0=sol[iterada,sitios]
fig = plt.figure()
ax = Axes3D(fig)
#ax.plot_surface(iterada, sitios, X0, cmap='spring')
#ax.plot_wireframe(iterada, sitios, X0, rstride=10, cstride=10)
ax.plot_surface(iterada, sitios, X0, rstride=1, cstride=1, cmap=cm.brg, linewidth=0,antialiased=True)
ax.set_xlabel("t/step")
ax.set_ylabel("k (sítios)")
ax.set_zlabel("X0")
ax.set_xlim([20000.,Npontos])
ax.set_ylim([0,N])
#ax.set_zlim([0.85,1.05])
ax.view_init(elev=45, azim=30)
plt.show()


# In[18]:


# Exportando os resultados em arquivos .dat
#______________________________________________________________________________________________________
# 1° Maneira usando o savetxt do numpy do Python
#Escrevendo a matriz solução (para a 1º variável)
passoimp=100
np.savetxt('matriz_solucao1.dat', sol[::passoimp,0:N], fmt='%-.10e', delimiter='   ', newline='\n', header='', footer='', comments='# ', encoding=None)
#np.savetxt('matriz_solucao1.dat', sol[:,0:N], fmt='%-.10e', delimiter='   ', newline='\n', header='', footer='', comments='# ', encoding=None)

#Escrevendo as séries temporais para sítios especificados
np.savetxt('temporal.dat', np.column_stack([t,sol[:,0],sol[:,50]]), fmt='%-.10e', delimiter='   ', newline='\n', header='', footer='', comments='# ', encoding=None)


#_______________________________________________________________________________________________________
# 2° Maneira usando o write do Python
#serie_temporal = open('serie_temporal.dat', 'w')

# Imprimindo séries temporais
#for i in range(0,Npontos,1):
#    tempo=round(i*step,ndigits=3)
#    solu1=round(sol[i,1],ndigits=3)
#    solu2=round(sol[i,2],ndigits=3)
#    serie_temporal.write(str(tempo) +'    '+ str(solu1) +'    '+ str(solu2) +'\n')
#serie_temporal.close() 
#_______________________________________________________________________________________________________


# In[19]:


#Plotando gráficos do arquivo
#2D: Séries temporais
t, sol0, sol1 = np.loadtxt('temporal.dat', dtype='float', delimiter='   ', unpack=True)
plt.plot(t,sol0, 'b', label='XK')
plt.plot(t,sol1, 'r', label='XN')
plt.xlabel('tempo')
plt.ylabel('y')
plt.title('Séries temporais a partir do arquivo dat')
plt.legend()
plt.show()


#Exemplo de como plotar 3D a partir de dados externos
Data3D= np.loadtxt("matriz_solucao1.dat", delimiter='   ')
pts_imp=int(tfinal/(step*passoimp))
#print(pts_imp,Npontos)
#iterada = np.arange(0, Npontos-1,1000)
iterada = np.arange(0, pts_imp-1,1)
sitios = np.arange(0, N,1)
iterada, sitios = np.meshgrid(iterada, sitios)
X0=Data3D[iterada,sitios]
fig = plt.figure()
ax = Axes3D(fig)
#ax.plot_surface(iterada, sitios, X0, cmap='spring')
#ax.plot_wireframe(iterada, sitios, X0, rstride=10, cstride=10)
ax.plot_surface(iterada, sitios, X0, rstride=1, cstride=1, cmap=cm.brg, linewidth=0, antialiased=True)
ax.set_xlabel("t/(step*Qtd. Pontos)")
ax.set_ylabel("k (sítios)")
ax.set_zlabel("X0")
#ax.set_xlim([0.,Npontos])
ax.set_xlim([0.,pts_imp])
ax.set_ylim([0,N])
ax.set_zlim([0.85,1.05])
ax.view_init(elev=45, azim=30)
plt.show()


# In[ ]:




