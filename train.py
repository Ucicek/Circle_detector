import torch
from tqdm import tqdm
import math

from utils import mae_loss, get_optimizer, get_dataloader, get_network
from setting import Setting


setting = Setting()

device = torch.device('cuda')

train_loader = get_dataloader(setting, train=True)
valid_loader = get_dataloader(setting, train=False)
network = get_network(setting, device)

optimizer = get_optimizer(setting.opt_hyp, network)

nb = len(train_loader)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)  # 5, 0.5 best result so far
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
for epoch in range(0, setting.n_epochs):
    least_loss = math.inf
    train_loss = 0.0
    valid_loss = 0.0

    network.train()

    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, desc=f'training epoch {epoch}', total=nb)

    for i, item in pbar:
        image, targets = item

        image = image.to(device)
        targets = targets.to(device)
        print(targets)
        optimizer.zero_grad()
        outputs = network(image)
        print(outputs)
        loss = mae_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    network.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = mae_loss(outputs, labels)
            valid_loss += loss.item()
            # Calculate the average loss for this epoch
    scheduler.step()

    train_loss = train_loss / len(train_loader)
    valid_loss = valid_loss / len(valid_loader)

    if valid_loss < least_loss:
        least_loss = valid_loss
        model_path = f'model1_{epoch}.pth'
        torch.save(network.state_dict(), model_path)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss
    ))




