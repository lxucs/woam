def get_accuracy(preds, golds):
    assert preds.shape[0] == golds.shape[0] != 0  # Numpy
    return (preds == golds).sum().item() / preds.shape[0]


def get_f1(p, r):
    f1 = (2 * p * r / (p + r)) if (p + r) else 0
    return f1


def get_prf(num_tps, num_preds, num_golds):
    precision = (num_tps * 1.0 / num_preds * 100) if num_preds != 0 else 0
    recall = (num_tps * 1.0 / num_golds * 100) if num_golds != 0 else 0
    f1 = get_f1(precision, recall)
    return precision, recall, f1
