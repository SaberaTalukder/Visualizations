import numpy as np
import pandas as pd
import surprise

def get_err(model, Y):
    """
    Takes as input a matrix Y of triples (i, j, Y_ij) where i is the index of a user,
    j is the index of a movie, and Y_ij is user i's rating of movie j and
    user/movie matrices U and V.

    Returns the mean regularized squared-error of predictions made by
    estimating Y_{ij} as the dot product of the ith row of U and the jth column of V^T.
    """
    results = np.array(list(map(lambda y : y.est, model.test(Y)))[0])
    return 0.5*np.mean((Y[:,2] - results)**2)
    #return reg*0.5*(np.linalg.norm(U)**2 + np.linalg.norm(V)**2) + 0.5*np.mean((Y[:,2] - results)**2)


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
    df = pd.DataFrame(Y)
    model = surprise.SVD()
    reader = surprise.Reader(rating_scale=(1, 5))
    data = surprise.Dataset.load_from_df(df[[0, 1, 2]], reader)

    trainset = data.build_full_trainset()
    model.fit(trainset)
    
    return (model, get_err(model, Y))


