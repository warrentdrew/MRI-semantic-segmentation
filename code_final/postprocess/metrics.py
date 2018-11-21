import numpy as np

def confusion_matrix(patients, model, batch_size, max_crop_size=32, discrete=True, normalized=True):
    """Confusion matrix of patients with a trained model."""
    preds = []
    truths = []
    for patient in patients:
        prediction = patient.get_prediction(model, batch_size)
        # be comparable between all model-modifications, with max crops-size 
        prediction = prediction[:-max_crop_size,:-max_crop_size,:-max_crop_size,:]
        if discrete:
            prediction = np.argmax(prediction, axis=-1)
        preds.append(prediction)
        _ , truth = patient.get_slices(count=False)
        truth = truth[:-max_crop_size,:-max_crop_size,:-max_crop_size,:]
        truths.append(truth)
    matrix, accuracy = confma(preds, truths, discrete=discrete, normalized=normalized)
    # compute derivations from confusion matrix
    cls = np.shape(matrix)[1]
    true_positives = np.array([matrix[c,c] for c in range(cls)])
    s = np.sum(true_positives)
    true_negatives = np.array([s - true_positives[c] for c in range(cls)])
    false_positives = np.array([np.sum(matrix[c,:]) - matrix[c,c]  for c in range(cls)])
    false_negatives = np.array([np.sum(matrix[:,c]) - matrix[c,c]  for c in range(cls)])
    return matrix, accuracy, true_positives, true_negatives, false_positives, false_negatives

def confma(preds, truths, discrete, normalized):
    cls = truths[0].shape[-1]
    onehot_enc = lambda arr: np.eye(cls)[arr]
    preds = [pred.reshape((-1)) for pred in preds] # flatten prediction (dimensions are not relevant)
    if discrete:
        preds = [onehot_enc(pred) for pred in preds]
    truths = [truth.reshape((-1, cls)) for truth in truths] # flatten truth
    # sum confusion matrices of all patients
    z = np.sum(np.array([np.dot(truth.T, pred) for (pred, truth) in zip(preds, truths)]), axis=0)
    accuracy = np.trace(z)/np.sum(z)
    if normalized:
        # normalization by class (number of elements in each class)
        norm = np.sum(z, axis=0)
        z = np.divide(z, norm)
    return z, accuracy

# derivations from a confusion matrix

def sensitivity(true_positives, false_negatives):
    """Compute Sensitivity/Recall/Hit Rate/True Positive Rate (TPR)."""
    return np.divide(true_positives, true_positives + false_negatives)

def specificity(true_negatives, false_positives):
    """Compute Specificity/Selectivity/True Negative Rate (TNR).""" 
    return np.divide(true_negatives, true_negatives + false_positives)

def precision(true_positives, false_positives):
    """Compute Precision/Positive Predictive Value (PPV)."""
    return np.divide(true_positives, true_positives + false_positives)

def false_negative_rate(true_positives, false_negatives):
    """Compute False Negative Rate (FNR)/Miss Rate."""
    return np.divide(false_negatives, false_negatives + true_positives)

def false_positive_rate(true_negatives, false_positives):
    """Compute False Positive Rate (FPR)/Fall-out."""
    return np.divide(false_positives, false_positives + true_negatives)

def dice(true_positives, false_positives, false_negatives):
    """Compute Dice coefficient (DICE) as an overlap based metric."""
    return np.divide(2*true_positives, 2*true_positives + false_positives + false_negatives)

def jaccard(true_positives, false_positives, false_negatives):
    """Compute Jaccard index (JAC) as a similarity index of two sets."""
    return np.divide(true_positives, true_positives + false_positives + false_negatives)




