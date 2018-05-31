import numpy as np

def auc(score_func, y_true, y_pred):
    """
    Arguments:
        score_func: function (roc_auc_score or average_precision_score)
        y_true: [n_samples, n_classes]
        y_pred: [n_samples, n_classes]
    """
    return np.array([score_func(y_t, y_p) 
                     for y_t, y_p in zip(y_true, y_pred)]).mean()

def lda_vec(corp, lda):
    idx, value = zip(*lda.get_document_topics(corp, minimum_probability=0))
    idx, value = np.array(idx), np.array(value) 
    res_vec = np.zeros(len(idx))
    res_vec[idx] = value
    return res_vec