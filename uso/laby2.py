import numpy as np  
import matplotlib.pyplot as plt 
import scipy.signal as sp


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

    sys(10,t)

def customstep(t):
    return 1 if t>0 else 0

def sys(y,t):
    kp = 2
    T = 10

    doty=(kp*customstep(t)-y)/T
    return  doty



if __name__=="__main__":
    main()
