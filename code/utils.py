import torch.optim as optim
from adamp import AdamP
from networks.Attention import Attention
from networks.SATRN import SATRN
from networks.SATRN_3 import SATRN_3
from networks.SATRN_4 import SATRN_4
from networks.SATRN_adamP import SATRN_adamP
from networks.SATRN_extension import SATRN_extension
from networks.SATRN_Final_all import SATRN_Final_all
from networks.ViT import ViT
from networks.EFFICIENT_SATRNv6 import EFFICIENT_SATRNv6


def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "Attention":
        model = Attention(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_3":
        model = SATRN_3(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_4":
        model = SATRN_4(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_adamP":
        model = SATRN_adamP(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_extension":
        model = SATRN_extension(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "SATRN_Final_all":
        model = SATRN_Final_all(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "EFFICIENT_SATRNv6":
        model = EFFICIENT_SATRNv6(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "ViT":
        model = ViT(FLAGS, train_dataset, model_checkpoint).to(device)
    else:
        raise NotImplementedError

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "AdamP":
        optimizer = AdamP(params, lr=lr, betas=(0.9, 0.999))
    else:
        raise NotImplementedError
    return optimizer
