from model.model import FModel
from data.data_loader import RandomDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
def main():
    lr = 0.01
    momentum = 0.9
    weight_decay  = 1e-4
    criterion = nn.CrossEntropyLoss().cuda()
    print_every = 10


    dataset  = RandomDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    model = FModel()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    for i, data in enumerate(dataloader):
        # measure data loading time
        patches, letters, expected_class = data
        output = model(patches, letters)
        loss = criterion(output, expected_class.flatten())
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, expected_class.flatten(), topk=(1, 5))
        if i%print_every == 0:
            print("acc1: {}, acc5: {}".format(acc1, acc5))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()

