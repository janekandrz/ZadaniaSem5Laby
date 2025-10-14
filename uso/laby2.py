import numpy as np  
import matplotlib.pyplot as plt 
import scipy.signal as sp

from scipy.integrate import odeint


def main():
    zad2_4()

def zad2():
    kp = 2
    T = 10

    # G(s) = kp/(T*s + 1)
    num = [kp]
    den = [T, 1]
    system = sp.TransferFunction(num, den)
    t, y = sp.step(system)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'b-', linewidth=2, label=f'Step Response (Kp={kp}, T={T})')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title('Step Response')
    plt.legend()
    plt.show()
    
    '''
    print(f"Transfer function: {system}")
    print(f"Final value: {y[-1]:.3f}")
    print(f"Time constant (T): {T}s")
    '''

def zad2_3():
    kp = 2
    T = 10

    A=-1/T
    B=kp/T
    C=1
    D=0

    G = sp.StateSpace(A,B,C,D)
    t, y = sp.step(G)

    plt.figure(figsize=(10, 6))
    plt.plot(t, y, 'b-', linewidth=2, label=f'Kp={kp}, T={T}')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title('Step response tf from state space')
    plt.legend()
    plt.show()

def zad2_4():
    t=np.arange(0,15,0.01)
    res = [0]
    for i in range(len(t)):
        res.append(odeint(model(res[i],t[i]),0,t))

    plt.figure()
    plt.plot(t,res)
    plt.show()

def u(t):
    return 1

def model(y,t):
    kp=2
    T=10
    return (kp*u(t)-y)/T
    

def zad2_5():
    t=np.arrange(0,15,0.01)

    res= odeint()

    pass



if __name__=="__main__":
    main()
