import numpy as np 
import numpy.linalg as npl


def main():
    zad1()


def zad1():
    a=pow(3,12)-5
    print(a)
    b=np.array([[2,0.5]])*np.array([[1,4],[-1,3]])*np.array([[-1,-3]]).transpose()
    print(b)
    c=npl.matrix_rank(np.array([[1,-2,0],[-2,4,0],[2,-1,7]]))
    print(c)
    d=np.invert(np.array([[1,2],[-1,0]]))*np.array([[-1,2]])
    print(d)

def zad2():
    arr=np.array([[1,1,-129,171,1620]])
    p = np.poly1d(arr)


if __name__ == "__main__":
    main()