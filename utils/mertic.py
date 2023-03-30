import numpy as np
import torch
def q_error(pred, real):
    """
     @brief Computes the Q error between two tensors. This is used to compute the probability that a prediction is better than a real prediction.
     @param pred The prediction of the model. Can be a Tensor or a Numpy array.
     @param real The real prediction of the model. Can be a Tensor or a Numpy array.
     @return The Q error between the two tensors. If both are 0 it is 1.
    """
    if isinstance(pred, torch.FloatTensor) or isinstance(pred, torch.IntTensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()
    pred = np.float32(pred)
    real = np.float32(real)
    if real == 0 and pred != 0:
        return pred
    if real != 0 and pred == 0:
        return real
    if real == 0 and pred == 0:
        return 1.0
    return max(pred / real, real / pred)


def batch_q_error(pred_list, real_list):
    """
     @brief Computes Q error for a batch of predictions. This is a wrapper for : func : ` q_error `
     @param pred_list List of predictions as returned by
     @param real_list List of real predictions as returned by
     @return A numpy array of q error values for each prediction
    """
    ret = np.zeros(len(pred_list))
    ID = 0
    # Return the error of the first prediction.

    for est, real in zip(pred_list, real_list):
        ret[ID] = q_error(est, real)
        ID = ID + 1
    return ret


def relative_error(pred, real):
    if isinstance(pred, torch.FloatTensor) or isinstance(pred, torch.IntTensor):
        pred = pred.cpu().detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().detach().numpy()

    pred, real = float(pred), float(real)
    pred = np.float32(pred)
    real = np.float32(real)
    if real == 0 and pred != 0:
        return 1
    if real != 0 and pred == 0:
        return 1
    if real == 0 and pred == 0:
        return 0
    return 100 * abs(pred - real) / real

def batch_relative_error(pred, real):
    batch, dim = pred.shape
    pred, real = pred.numpy(), real.numpy()
    err = np.empty([batch, dim])
    for bi, (bp, br) in enumerate(zip(pred, real)):
        for di, (dp, dr) in enumerate(zip(bp, br)):
            err[bi, di] = relative_error(dp, dr)
    return err


def groupby_relative_error(pred, real, eps=1e-9):
    assert pred.shape == real.shape
    pred, real = pred[:, 1:], real[:, 1:]
    err = np.abs(pred - real)
    rerr = err / (real + eps)
    return rerr



def sMAPE(pred, real):
    """ bounded relative error """
    return 2 * abs(pred - real) / (abs(pred) + abs(real))


def log_metric(m):
    s = ''
    for c in m.index:
        s += f"{c}:{m[c]:.3f}% \n"
    return s
