
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


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
    return eta*(reg*bj + (Yij-mu-Ui@Vj-ai-bj) * -1)

def get_err(U, V, Y, mu, a, b, reg=0.0):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    result = []
    for Y_row in Y:
        i = Y_row[0] - 1
        j = Y_row[1] - 1
        Y_ij = Y_row[2]
#         print("calculating error")
#         print(reg/2 * (LA.norm(U)**2 + LA.norm(V)**2 + LA.norm(a)**2 + LA.norm(b)**2) + 1/2 * ((Y_ij-mu) - (np.dot(U[i].T, V[j])+a[i]+b[j]))**2)
        result.append(reg/2 * (LA.norm(U)**2 + LA.norm(V)**2 + LA.norm(a)**2 + LA.norm(b)**2) + 1/2 * ((Y_ij-mu) - (np.dot(U[i].T, V[j])+a[i]+b[j]))**2)
    return np.mean(result)

#     return reg*0.5*(np.linalg.norm(U)**2 + np.linalg.norm(V)**2 + np.linalg.norm(a)**2 + np.linalg.norm(b)**2) + 0.5*np.mean((Y[:,2]-mu) - ((U@V.T)[Y[:,0]-1,Y[:,1]-1]+a[Y[:,0]-1]+b[Y[:,1]-1])**2)

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
    print(M, N, K)
    U = np.random.uniform(-0.5,0.5,(M,K))
    V = np.random.uniform(-0.5,0.5,(N,K))
    a = np.random.uniform(-0.5,0.5,(M,1))
    b = np.random.uniform(-0.5,0.5,(N,1))
    mu = np.mean(Y)
    
    prev_err = np.Inf
    err = get_err(U, V, Y, mu, a, b, reg)
    print(err)
    err_list = [err]
    baseline_improvement = -1
    for epoch in range(max_epochs):
        if epoch % 10 == 0:
            # Plot values of E_out across k for each value of lambda
            plt.plot(range(len(err_list)), err_list)
            plt.title('$Error$ vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.show()
        print(epoch)
        # Save the previous error
        prev_err = err
        # Iterate over Y in a random order
        shuffle = np.random.permutation(Y.shape[0])
        for ind in shuffle:
            i = Y[ind,0]-1
            j = Y[ind,1]-1
            Y_ij = Y[ind,2]
            # Update U and V
            U[i,:] -= grad_U(U[i,:],Y_ij,V[j,:],mu,a[i,:],b[j,:],reg,eta)
            V[j,:] -= grad_V(V[j,:],Y_ij,U[i,:],mu,a[i,:],b[j,:],reg,eta)
            a[i,:] -= grad_a(a[i,:],Y_ij,U[i,:],V[j,:],mu,b[j,:],reg,eta)
            b[j,:] -= grad_b(b[j,:],Y_ij,U[i,:],V[j,:],mu,a[i,:],reg,eta)
        # Get the current error
        err = get_err(U, V, Y, mu, a, b, reg)
        print(err)
        err_list.append(err)
        # Check if the error is less than the tolerance
        if baseline_improvement == -1 :
            baseline_improvement = prev_err-err
        if (prev_err-err)/baseline_improvement <= eps :
            break
    
    return (U, V, get_err(U, V, Y, mu, a, b), err_list)

def apply_model(U,V):
    pass


