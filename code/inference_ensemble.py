import torch
import os
from train import id_to_string
from metrics import word_error_rate, sentence_acc
from checkpoint import load_checkpoint
import torch.nn as nn
from torchvision import transforms
from dataset import LoadEvalDataset, collate_eval_batch, START, PAD
from flags import Flags
from utils import get_network, get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm


def main(parser):
    is_cuda = torch.cuda.is_available()
    checkpoint1 = load_checkpoint(parser.checkpoint1, cuda=False)
    # checkpoint2 = load_checkpoint(parser.checkpoint2, cuda=False)
    checkpoint3 = load_checkpoint(parser.checkpoint3, cuda=False)
    
    options1 = Flags(checkpoint1["configs"]).get()
    # options2 = Flags(checkpoint2["configs"]).get()
    options3 = Flags(checkpoint3["configs"]).get()


    torch.manual_seed(options1.seed)
    random.seed(options1.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Model1 Running {} on device {}\n".format(options1.network, device))
    # print("Model2 Running {} on device {}\n".format(options2.network, device))
    print("Model3 Running {} on device {}\n".format(options3.network, device))


    model_checkpoint1 = checkpoint1["model"]
    # model_checkpoint2 = checkpoint2["model"]
    model_checkpoint3 = checkpoint3["model"]

    if model_checkpoint1:
        print(
            "[+] Checkpoint\n",
            "Model1 Resuming from epoch : {}\n".format(checkpoint1["epoch"]),
        )

    # if model_checkpoint2:
    #     print(
    #         "[+] Checkpoint\n",
    #         "Model2 Resuming from epoch : {}\n".format(checkpoint2["epoch"]),
    #     )

    if model_checkpoint3:
        print(
            "[+] Checkpoint\n",
            "Model3 Resuming from epoch : {}\n".format(checkpoint3["epoch"]),
        )

    print(options1.input_size.height)
    print('model check')

    transformed = transforms.Compose(
        [
            transforms.Resize((options1.input_size.height, options1.input_size.width)),
            transforms.ToTensor(),
        ]
    )

    dummy_gt = "\sin " * parser.max_sequence  # set maximum inference sequence

    root = os.path.join(os.path.dirname(parser.file_path), "images")
    with open(parser.file_path, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
    test_data = [[os.path.join(root, x[0]), x[0], dummy_gt] for x in data]
    test_dataset = LoadEvalDataset(
        test_data, checkpoint1["token_to_id"], checkpoint1["id_to_token"], crop=False, transform=transformed,
        rgb=options1.data.rgb
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=parser.batch_size,
        shuffle=False,
        num_workers=options1.num_workers,
        collate_fn=collate_eval_batch,
    )

    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    model1 = get_network(
        "SATRN_3",
        options1,
        model_checkpoint1,
        'cpu',
        test_dataset,
    )

    # model2 = get_network(
    #     options2.network,
    #     options2,
    #     model_checkpoint2,
    #     'cpu',
    #     test_dataset
    # )

    model3 = get_network(
        "SATRN_4",
        options3,
        model_checkpoint3,
        'cpu',
        test_dataset
    )

    results = []
    # model_list = [model1, model2, model3]
    model_list = [model1, model3]
    
    # ver1
    for d in tqdm(test_data_loader):
        input = d["image"].to(device)
        print(input)
        expected = d["truth"]["encoded"].to(device)
        for i in range(2):
            if i == 1:
                m = nn.Upsample(size =[96, 384])         
                input = m(input)

            for idx1, model in tqdm(enumerate(model_list)):
                model.to('cuda')
                model.eval()
                output = model(input, expected, False, 0.0)
                decoded_values = output.transpose(1, 2)
                del model

                if idx1 == 0:
                    tensor = decoded_values
                else:
                    tensor += decoded_values
        _, sequence = torch.topk(tensor, 1, dim=1)
        sequence = sequence.squeeze(1)
        # hard voting

        sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
        # print(sequence_str)
        for path, predicted in zip(d["file_path"], sequence_str):
            results.append((path, predicted))

    # Ver2: Out of Memory
    # for idx1, model in enumerate(model_list):
    #     model.to(device)
    #     model.eval()
    #     for idx2, d in tqdm(enumerate(test_data_loader)):
    #         input = d["image"].to(device)
    #         expected = d["truth"]["encoded"].to(device)
    #         output = model(input, expected, False, 0.0)
    #         if idx2 == 0:
    #             decoded_values = output.transpose(1, 2) #[8, 245, 231]
    #         else:
    #             decoded_values_temp = output.transpose(1, 2)
    #             decoded_values= torch.cat((decoded_values, decoded_values_temp), dim=0)            
    #            # [32, 245, 231]
    #     del model
    #     if idx1 == 0:
    #         tensor = decoded_values
    #     else:
    #         tensor += decoded_values
    # _, sequence = torch.topk(tensor, 1, dim=1)
    # sequence = sequence.squeeze(1)
    # sequence_str = id_to_string(sequence, test_data_loader, do_eval=1)
    # for path, predicted in zip(d["file_path"], sequence_str):
    #     results.append((path, predicted))

    os.makedirs(parser.output_dir, exist_ok=True)
    with open(os.path.join(parser.output_dir, "output.csv"), "w") as w:
        for path, predicted in results:
            w.write(path + "\t" + predicted + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model1
    parser.add_argument(
        "--checkpoint1",
        dest="checkpoint1",
        default="./log/satrn_3/0049.pth",
        type=str,
        help="Path of checkpoint file",
    )

    parser.add_argument(
        "--checkpoint2",
        dest="checkpoint2",
        default="./log/satrn_ratio_dataset_ver2_2_augmentation/checkpoints/0050.pth",
        type=str,
        help="Path of checkpoint file",
    )


    parser.add_argument(
        "--checkpoint3",
        dest="checkpoint3",
        default="./log/satrn_4/0050.pth",
        type=str,
        help="Path of checkpoint file",
    )

    parser.add_argument(
        "--max_sequence",
        dest="max_sequence",
        default=230,
        type=int,
        help="maximun sequence when doing inference",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=4,
        type=int,
        help="batch size when doing inference",
    )

    eval_dir = os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/')
    file_path = os.path.join(eval_dir, 'eval_dataset/input.txt')
    parser.add_argument(
        "--file_path",
        dest="file_path",
        default=file_path,
        type=str,
        help="file path when doing inference",
    )

    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'submit')
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default=output_dir,
        type=str,
        help="output directory",
    )

    parser = parser.parse_args()
    main(parser)
