import torch
import os
import argparse
import time
import datetime
from torch.utils.data import DataLoader

from net.solver import MyNetTrainer
from dataset import get_training_set, get_test_set


# ===========================================================
# Training settings
# ===========================================================
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
parser = argparse.ArgumentParser(description='PyTorch Super Resolution')

# dataset settings
parser.add_argument('--dataSet', type=str, default='DIV2K', help='desicion of dataset')
parser.add_argument('--crop_size', type=int, default=256, help='crop size of the sample')

# hyper-parameters
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
parser.add_argument('--seed', type=int, default=42, help='random seed to use')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="only 2 * k")
args = parser.parse_args()


if __name__ == '__main__':

    # ===========================================================
    # prepare for store model
    # ===========================================================
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H:%M:%S")
    model_out_path = '/home/guozy/BISHE/MyNet/result/' + now
    if os.path.exists(model_out_path) == False:
            os.mkdir(model_out_path)
    
    
    # ===========================================================
    # To store settings information of model
    # ===========================================================
    argsDict = args.__dict__
    with open(model_out_path + '/information.txt', 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')


    # ===========================================================
    # Set train dataset & test dataset
    # ===========================================================
    print('\n===> Loading datasets')
    train_set = get_training_set(args.upscale_factor,args.crop_size, args.dataSet)
    test_set = get_test_set(args.upscale_factor, args.crop_size, args.dataSet)
    training_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=True, num_workers=4,)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)


    # ===========================================================
    # train model
    # ===========================================================
    time_start = time.time()
    model = MyNetTrainer(args, training_data_loader, testing_data_loader, model_out_path)
    model.run()


    # ===========================================================
    # check how much time was used to train model
    # ===========================================================
    time_end = time.time()
    print('\n===> time cost of traning:', time_end - time_start)  
    with open(model_out_path + '/information.txt', 'a') as f:
        f.write('\ntime:{}s\n\n'.format(time_end - time_start))
        f.close()


    # ===========================================================
    # save and eval the model
    # ===========================================================
    model.save(model_out_path)
