import torch
from sklearn import metrics

def mse(output, target):
    with torch.no_grad():
        return metrics.mean_squared_error(list(target.cpu().numpy()), list(output.cpu().numpy()))

def macro_f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return metrics.f1_score(list(target.cpu().numpy()), list(pred.cpu().numpy()), average='macro')

def weighted_f1(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        return metrics.f1_score(list(target.cpu().numpy()), list(pred.cpu().numpy()), average='weighted')

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
