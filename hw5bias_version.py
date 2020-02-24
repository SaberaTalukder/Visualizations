
import numpy as np
from numpy import linalg as LA

def grad_U(Ui, Yij, Vj, mu, ai, bj, reg, eta):
    """
    Takes as input Ui (the ith row of U), a training point Yij, the column
    vector Vj (jth column of V^T), mu (the avg of all observations in Y), 
    ai (bias term for user), bj (bias term for movie),
    reg (the regularization parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Ui multiplied by eta.
    """
    return eta*(reg*Ui + (Yij-mu-Ui@Vj-ai-bj) * -Vj)

def grad_V(Vj, Yij, Ui, mu, ai, bj, reg, eta):
    """
    Takes as input the column vector Vj (jth column of V^T), a training point Yij,
    Ui (the ith row of U), mu (the avg of all observations in Y), 
    ai (bias term for user), bj (bias term for movie),
    reg (the regularization parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to Vj multiplied by eta.
    """
    return eta*(reg*Vj + (Yij-mu-Ui@Vj-ai-bj) * -Ui)

def grad_a(ai, Yij, Ui, Vj, mu, bj, reg, eta):
    """
    Takes as input ai (bias term for user), a training point Yij,
    Ui (the ith row of U), the column vector Vj (jth column of V^T), mu
    (the avg of all observations in Y), bj (bias term for movie),
    reg (the regularization parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to ai multiplied by eta.
    """
    print('Ui')
    print(Ui)
    print('Vj')
    print(Vj)
    return eta*(reg*ai + (Yij-mu-Ui@Vj-ai-bj) * -1)

def grad_b(bj, Yij, Ui, Vj, mu, ai, reg, eta):
    """
    Takes as input bj (bias term for movie), a training point Yij,
    Ui (the ith row of U), the column vector Vj (jth column of V^T), mu
    (the avg of all observations in Y), ai (bias term for user),
    reg (the regularization parameter lambda), and eta (the learning rate).

    Returns the gradient of the regularized loss function with
    respect to bj multiplied by eta.
    """
    print('Ui')
    print(Ui)
    print('Vj')
    print(Vj)
    return eta*(reg*bj + (Yij-mu-Ui@Vj-ai-bj) * -1)

def get_err(U, V, Y, mu, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    print(mu.shape)
    print((U@V.T).shape)
    return reg*0.5*(np.linalg.norm(U)**2 + np.linalg.norm(V)**2 + np.linalg.norm(a)**2 + np.linalg.norm(b)**2) + 0.5*np.mean((Y[:,2] - (mu+U@V.T+a+b.T)[Y[:,0]-1,Y[:,1]-1])**2)

def train_model(M, N, K, eta, reg, Y, eps=0.0001, max_epochs=300):
    """
    Given a training data matrix Y containing rows (i, j, Y_ij)
    where Y_ij is user i's rating on movie j, learns an
    M x K matrix U and N x K matrix V such that rating Y_ij is approximated
    by (UV^T)_ij.

    Uses a learning rate of <eta> and regularization of <reg>. Stops after
    <max_epochs> epochs, or once the magnitude of the decrease in regularized
    MSE between epochs is smaller than a fraction <eps> of the decrease in
    MSE after the first epoch.

    Returns a tuple (U, V, err) consisting of U, V, and the unregularized MSE
    of the model.
    """
    U = np.random.rand(M,K) - .5
    V = np.random.rand(N,K) - .5
    a = np.random.rand(M,1) - .5
    b = np.random.rand(N,1) - .5
    mu = np.mean(Y)
    
    init_err = get_err(U, V, Y, mu, a, b, reg)
    print(init_err)
    
    for e in range(max_epochs):
        print(e)
        for idx in np.random.permutation(range(len(Y))):
            i = Y[idx, 0] - 1
            j = Y[idx, 1] - 1
            Y_ij = Y[idx, 2]
            U[i] = U[i] - grad_U(U[i], Y_ij, V[j], mu, a[i], b[j], reg, eta)
            V[j] = V[j] - grad_V(V[j], Y_ij, U[i], mu, a[i], b[j], reg, eta)
            a[i] = a[i] - grad_a(a[i], Y_ij, U[i], V[j], mu, b[j], reg, eta)
            b[j] = b[j] - grad_b(b[j], Y_ij, U[i], V[j], mu, a[i], reg, eta)
        
        if e == 0:
            first_err = get_err(U, V, Y, mu, a, b, reg)
            prev_err = first_err
        elif e > 0:
            
            err = get_err(U, V, Y, mu, a, b, reg)
            if (err-prev_err) / (first_err-init_err) <= eps:
                break
            print(err)
            prev_err = err
        
    err = get_err(U, V, Y, mu, a, b)
    return (U, V, err)

def apply_model(U,V):
    pass


