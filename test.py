import torch

from utils import get_dataloader,get_network, np_to_circleparams, batch_sum_iou
from setting import Setting

if __name__ == 'main':
    setting = Setting()
    setting.DATA_VAL = setting.PATHDATA + '/test'
    device = torch.device('cpu', 0)
    setting.batch_size = 1
    model = get_network(setting,device)
    model.eval()

    test_dataloader = get_dataloader(setting, train=False)

    iou_score = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            images, labels = data
            outputs = model(images)
            labels = np_to_circleparams(labels)
            outputs = np_to_circleparams(outputs)
            iou_score += batch_sum_iou(labels,outputs)

    print(iou_score/(len(test_dataloader)*setting.batch_size))