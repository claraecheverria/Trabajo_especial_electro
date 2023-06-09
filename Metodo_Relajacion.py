import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, Eq, solve

def voltaje(N):

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = N/4

    #tolerancia para dibujar cfas
    tol = N/200

    # Establecer condiciones iniciales
    for i in range(N):
        for j in range(N):
            x = i - N // 2
            y = j - N // 2
            if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2:
                if x >= 0 and y >= 0:
                    grilla[i, j] = 1  # Primer cuadrante
                elif x <= 0 and y <= 0:
                    grilla[i, j] = -1  # Tercer cuadrante

    # Iteraciones para la relajación
    num_iteraciones = 250

    # Relajación del círculo
    for k in range(num_iteraciones):
        new_grilla = np.copy(grilla)
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                x = i - N // 2
                y = j - N // 2
                if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2 and ((x >= 0 and y >= 0) or (x <= 0 and y <= 0)):
                        new_grilla[i,j] = grilla[i,j]
                else:
                    new_grilla[i, j] = (grilla[(i + 1) % N, j] + grilla[(i - 1) % N, j] +
                                    grilla[i, (j + 1) % N] + grilla[i, (j - 1) % N]) / 4
                    
        grilla = np.copy(new_grilla)

    
    return grilla


def campo_electrico_x(voltaje, N):

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = N/4

    #tolerancia para dibujar cfas
    tol = N/200

    '''# Establecer condiciones iniciales
    for i in range(N):
        for j in range(N):
            x = i - N // 2
            y = j - N // 2
            if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2:
                if x >= 0 and y >= 0:
                    grilla[i, j] = 0  # Primer cuadrante
                elif x <= 0 and y <= 0:
                    grilla[i, j] = 0  # Tercer cuadrante
    '''
    # Iteraciones para la relajación
    num_iteraciones = 250

    # Relajación del círculo
    for k in range(num_iteraciones):
        new_grilla = np.copy(grilla)
        for i in range(0, N):
            for j in range(0, N):
                x = i - N // 2
                y = j - N // 2
                # if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2 and ((x >= 0 and y >= 0) or (x <= 0 and y <= 0)):
                #         new_grilla[i,j] = grilla[i,j]
                # else:
                new_grilla[i, j] = abs((voltaje[(i + 1) % N, j] - voltaje[(i - 1) % N, j])/(2/(1)))
                    
        grilla = np.copy(new_grilla)
        return grilla
    
def campo_electrico_y(voltaje, N):

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = N/4

    #tolerancia para dibujar cfas
    tol = N/200

    '''# Establecer condiciones iniciales
    for i in range(N):
        for j in range(N):
            x = i - N // 2
            y = j - N // 2
            if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2:
                if x >= 0 and y >= 0:
                    grilla[i, j] = 0  # Primer cuadrante
                elif x <= 0 and y <= 0:
                    grilla[i, j] = 0  # Tercer cuadrante
    '''
    # Iteraciones para la relajación
    num_iteraciones = 250

    # Relajación del círculo
    for k in range(num_iteraciones):
        new_grilla = np.copy(grilla)
        for i in range(0, N):
            for j in range(0, N):
                x = i - N // 2
                y = j - N // 2
                # if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2 and ((x >= 0 and y >= 0) or (x <= 0 and y <= 0)):
                #         new_grilla[i,j] = grilla[i,j]
                # else:
                new_grilla[i, j] = abs((voltaje[i, (j + 1) % N] - voltaje[i, (j - 1) % N])/(2/(1)))
                    
        grilla = np.copy(new_grilla)
        return grilla
    
def print_campo_elec(v, N):
    # Graficar la grilla
    fig, ax = plt.subplots()
    im = ax.imshow(v, cmap='hot', origin='lower')
    plt.colorbar(im)
    plt.title('Método de Relajación - Voltaje')
    ax.set_xlabel('Aumento de h según x')
    ax.set_ylabel('Aumento de h según y')
    plt.show()
    x = campo_electrico_x(v,N)
    y = campo_electrico_y(v,N)

    campo_elec = np.zeros((N, N))
    for i in range(0, N):
        for j in range(0, N):
            campo_elec[i,j]= np.sqrt((x[i,j])**2+(y[i,j])**2)  

    # Graficar la grilla
    fig, ax = plt.subplots()
    im = ax.imshow(campo_elec, cmap='hot', origin='lower')
    plt.colorbar(im)
    plt.title('Método de Relajación - Módulo Campo eléctrico ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return campo_elec

def lineas_campo_elec(e,N):


    # Crear una matriz de numpy con el módulo del campo eléctrico
    campo_electrico = e

    # Crear una cuadrícula de coordenadas para los puntos en la matriz
    x = np.arange(0, campo_electrico.shape[1], 1)
    y = np.arange(0, campo_electrico.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    # Calcular los componentes x e y del campo eléctrico a partir del módulo
    Ex = campo_electrico * np.cos(np.pi/4)  # Componente x del campo eléctrico
    Ey = campo_electrico * np.sin(np.pi/4)  # Componente y del campo eléctrico
    v= voltaje(N)
    Ex = campo_electrico_x(v,N)
    Ex = -1*Ex
    Ey = campo_electrico_y(v,N)
    Ey = -1*Ey #es porque al graficar las líneas la función por defecto va de - a +

    # Graficar las líneas de campo
    plt.figure()
    #plt.quiver(X, Y, Ex, Ey)
    plt.streamplot(X, Y, Ex, Ey, color='b')
    plt.xlabel('Coordenada x')
    plt.ylabel('Coordenada y')
    plt.title('Líneas de campo eléctrico')
    plt.show()

def deducir_recta_dos_puntos(x1, y1, x2, y2):
    # Calcula la pendiente utilizando la fórmula (y2 - y1) / (x2 - x1)
    m = (y2 - y1) / (x2 - x1)
    
    # Utiliza la forma punto-pendiente para deducir la ecuación de la recta
    b = y1 - m * x1
    
    # Retorna la ecuación de la recta en forma de diccionario
    return {"m": m, "b": b}

def calcular_angulo_recta(equacion):
    # Extrae la pendiente de la ecuación de la recta
    m = equacion["m"]
    
    # Calcula el ángulo en radianes utilizando la función arcotangente
    angulo_radianes = math.atan(m)
    
    # Convierte el ángulo de radianes a grados
    #angulo_grados = math.degrees(angulo_radianes)
    
    return angulo_radianes

def campo_elec_radial(Ex, Ey, N):
    pos_x = np.zeros((N, N))
    pos_y = np.zeros((N, N))
    
    Er = np.zeros((N, N))

    for i in range(0, N):
        for j in range(0, N):
            x = i - N // 2
            y = j - N // 2
            pos_x[i,j]= x
            pos_y[i,j]= y
            if(x ==0 and y==0):
                print("Posicion:")
                print(i,j)

    for i in range(0, N):
        for j in range(0, N):
            if(pos_x[i,j]== 0):
                Er[i,j] = (Ey[i,j])          #no puedo divir entre cero, si es cero tiene solo componente en y
            else:            
                recta = deducir_recta_dos_puntos(0,0,pos_x[i,j], pos_y[i,j])
                tita = calcular_angulo_recta(recta)
                Er[i,j] = (Ex[i,j]*np.cos(tita)) + (Ey[i,j]*np.sin(tita))
    #sigma = epsilon cero* Er
    return Er

def carga(sigma,N):
    #tolerancia para dibujar cfas
    tol = 0.5
    suma = 0
    contador = 0
    # Radio del círculo
    radio = N/4
    # Establecer condiciones iniciales
    for i in range(N):
        for j in range(N):
            x = i - N // 2
            y = j - N // 2
            if (x**2 + y**2) <= (radio+tol)**2 and (x**2 + y**2) >= (radio-tol)**2:
                if x >= 0 and y >= 0:
                    suma = suma +sigma[i,j]
                    contador = contador+1

    promedio = suma/contador
    print("Promedio:")
    print(promedio)
    carga = promedio*2*np.pi*radio/4
    return carga

def capacitancia(carga):
    cap = carga/2           #delta V = 2
    print("Capacitancia:")
    print(cap)
    return cap

N=250
v = voltaje(N)
Ex = campo_electrico_x(v,N)
Ey = campo_electrico_y(v,N)
e =print_campo_elec(v,N)
# lineas_campo_elec(e)
sigma = campo_elec_radial(Ex, Ey, N)
capacitancia(carga(sigma,N))

def richardson(n1,n2,n3,C1,C2,C3):
    # Define the variables
    A, B, C = symbols('A B C')

    # Define the equations
    equation1 = Eq(A + B/n1 + C/(n1**2), C1)
    equation2 = Eq(A + B/n2 + C/(n2**2), C2)
    equation3 = Eq(A + B/n3 + C/(n3**2), C3)

    # Solve the system of equations
    solution = solve((equation1, equation2, equation3), (A, B, C))

    # Print the solution
    print("Solution:")
    print("A =", solution[A])
    print("B =", solution[B])
    print("C =", solution[C])

richardson(100,80,60,4.8e-12,3.2e-12,2.7e-12)



