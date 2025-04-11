import torch
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import accuracy_score
from networks.ACIR import ACIR
from util.tools import inference_ACIR, compute_topK, CalcTopMap
from util.losses import L_WSC
from config import get_config
from util.dataloader import get_radio_data
from tqdm import tqdm
import os
from timm.scheduler.cosine_lr import CosineLRScheduler

# Allow sharing of CUDA tensors among processes.
torch.multiprocessing.set_sharing_strategy('file_system')



def Train_ACIR(config, bit):
    """
    Train and validate the ACIR model.

    Args:
        config (dict): Configuration dictionary.
        bit (int): Bit length for the hash codes.
    """
    device = torch.device("cuda:{}".format(config["gpus"][0])
                          if torch.cuda.is_available() else "cpu")

    # Load datasets and corresponding statistics.
    train_loader, valid_loader, test_loader, num_train, num_valid, num_test, class_weights = \
        get_radio_data(config, root_path=r'/media/ssd1/ny/retrieval/', numworkers=4, gray=False)

    config["num_train"] = num_train
    train_weights = class_weights['train']

    # Initialize the ACIR classification model.
    net = ACIR(bit=bit, input_size=224, num_cls=config["n_class"]).to(device)
    if len(config["gpus"]) != 1:
        net = torch.nn.DataParallel(net.to("cuda"), device_ids=config["gpus"], output_device=config["gpus"][0])
    # Optionally load a pretrained model:
    # net.load_state_dict(torch.load('/path/to/your/model.ptl'))

    optimizer = optim.AdamW(net.parameters(), lr=3e-4, betas=(0.9, 0.95))
    scheduler = CosineLRScheduler(
        optimizer, lr_min=5e-6, t_initial=config["epoch"], warmup_t=10,
        warmup_lr_init=1e-4, warmup_prefix=True
    )

    criterion = L_WSC(bit, train_weights, device)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    Best_mAP = 0
    save_mAP = 0.6
    checkpoint_folder = config["save_path"]
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    patience = 0  # Patience counter for early stopping

    for epoch in range(config["epoch"]):
        net.train()
        train_loss = 0

        for image, _, label, ind, _ in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            hashcode, pd_cls_logits = net(image)
            ce_loss = cross_entropy_loss(pd_cls_logits, torch.argmax(label, dim=-1))
            raw_labels = torch.argmax(label, dim=1)
            loss_fusion, _, _ = criterion(hashcode, raw_labels)
            loss = loss_fusion + 0.5 * ce_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        if (epoch + 1) % config["test_freq"] == 0:
            with torch.no_grad():
                tst_binary, tst_label, pd_tst_clslogits = inference_ACIR(
                    test_loader, net, device=device, act=None
                )
                valid_binary, valid_label, pd_valid_clslogits = inference_ACIR(
                    valid_loader, net, device=device, act=None
                )

            acc = accuracy_score(np.argmax(tst_label, axis=1), np.argmax(pd_tst_clslogits, axis=1))
            map_score = CalcTopMap(valid_binary, tst_binary, valid_label, tst_label, topk=100)
            Nmap, mar, _ = compute_topK(
                valid_binary, tst_binary, valid_label, tst_label,
                radius=range(bit // 4 + 1), number_class=config['n_class'],
                topK=[100, 500, 1000, len(valid_label)]
            )

            if map_score > Best_mAP:
                print("========================Updating the best model!=========================")
                Best_mAP = map_score
                if Best_mAP > save_mAP:
                    torch.save(net.state_dict(),
                               os.path.join(checkpoint_folder, f'best_model_{bit}b.ptl'))
                patience = 0
            else:
                patience += 1

            scheduler.step(epoch)
            print(f"{config['info']} epoch:{epoch + 1} trainLoss:{train_loss:.4f} bit:{bit} "
                  f"dataset:{config['dataset']} MAP:{round(map_score, 4)} myMAP:{round(Nmap, 4)}, "
                  f"acc:{round(acc, 4)}")

        if patience > 50:
            break

    best_model_path = os.path.join(checkpoint_folder, f'best_model_{bit}b.ptl')
    new_model_path = os.path.join(checkpoint_folder, f'best_model_{bit}_{round(Best_mAP, 4)}.ptl')
    os.rename(best_model_path, new_model_path)


if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        Train_ACIR(config, bit)
