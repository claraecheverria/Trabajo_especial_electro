import numpy as np
import matplotlib.pyplot as plt
import math

def voltaje():

    # Tamaño de la grilla
    N = 100

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = 25

    #tolerancia para dibujar cfas
    tol = 0.5

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


def campo_electrico_x(voltaje):
    # Tamaño de la grilla
    N = 100

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = 25

    #tolerancia para dibujar cfas
    tol = 0.5

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
    
def campo_electrico_y(voltaje):
    # Tamaño de la grilla
    N = 100

    # Crear una matriz para representar la grilla
    grilla = np.zeros((N, N))

    # Radio del círculo
    radio = 25

    #tolerancia para dibujar cfas
    tol = 0.5

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
    

def print_campo_elec(v):
    # Graficar la grilla
    fig, ax = plt.subplots()
    im = ax.imshow(v, cmap='hot', origin='lower')
    plt.colorbar(im)
    plt.title('Método de Relajación - Voltaje')
    ax.set_xlabel('Aumento de h según x')
    ax.set_ylabel('Aumento de h según y')
    plt.show()
    x = campo_electrico_x(v)
    y = campo_electrico_y(v)

    campo_elec = np.zeros((100, 100))
    for i in range(0, 100):
        for j in range(0, 100):
            campo_elec[i,j]= np.sqrt((x[i,j])**2+(y[i,j])**2)  

    # Graficar la grilla
    fig, ax = plt.subplots()
    im = ax.imshow(campo_elec, cmap='hot', origin='lower')
    plt.colorbar(im)
    plt.title('Método de Relajación - Campo eléctrico ')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return campo_elec



def lineas_campo_elec(e):


    # Crear una matriz de numpy con el módulo del campo eléctrico
    campo_electrico = e

    # Crear una cuadrícula de coordenadas para los puntos en la matriz
    x = np.arange(0, campo_electrico.shape[1], 1)
    y = np.arange(0, campo_electrico.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    # Calcular los componentes x e y del campo eléctrico a partir del módulo
    Ex = campo_electrico * np.cos(np.pi/4)  # Componente x del campo eléctrico
    Ey = campo_electrico * np.sin(np.pi/4)  # Componente y del campo eléctrico
    v= voltaje()
    Ex = campo_electrico_x(v)
    Ex = -1*Ex
    Ey = campo_electrico_y(v)
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

def campo_elec_radial(Ex, Ey):
    pos_x = np.zeros((100, 100))
    pos_y = np.zeros((100, 100))
    
    Er = np.zeros((100, 100))

    for i in range(0, 100):
        for j in range(0, 100):
            x = i - 100 // 2
            y = j - 100 // 2
            pos_x[i,j]= x
            pos_y[i,j]= y
            if(x ==0 and y==0):
                print("Posicion:")
                print(i,j)

    for i in range(0, 100):
        for j in range(0, 100):
            if(pos_x[i,j]== 0):
                Er[i,j] = (Ey[i,j])          #no puedo divir entre cero, si es cero tiene solo componente en y
            else:            
                recta = deducir_recta_dos_puntos(0,0,pos_x[i,j], pos_y[i,j])
                tita = calcular_angulo_recta(recta)
                Er[i,j] = (Ex[i,j]*np.cos(tita)) + (Ey[i,j]*np.sin(tita))
    #sigma = epsilon cero* Er
    return Er

def carga(sigma):
    #tolerancia para dibujar cfas
    tol = 0.5
    suma = 0
    contador = 0
    # Radio del círculo
    radio = 25
    # Establecer condiciones iniciales
    for i in range(100):
        for j in range(100):
            x = i - 100 // 2
            y = j - 100 // 2
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

                     
v = voltaje()
Ex = campo_electrico_x(v)
Ey = campo_electrico_y(v)
e =print_campo_elec(v)
# lineas_campo_elec(e)
sigma = campo_elec_radial(Ex, Ey)
capacitancia(carga(sigma))

    



