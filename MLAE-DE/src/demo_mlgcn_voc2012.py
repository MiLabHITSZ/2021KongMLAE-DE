import sys
sys.path.append('../')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch
import os
import numpy as np
import logging
from tqdm import tqdm
import torchvision.transforms as transforms
from ml_gcn_model.models import gcn_resnet101_attack
from ml_gcn_model.voc import Voc2012Classification
from ml_gcn_model.util import Warp
from ml_gcn_model.voc import write_object_labels_csv
from src.attack_model import AttackModel
import os
from PIL import Image
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='multi-label attack')
parser.add_argument('--data', default='../data/voc2012', type=str,
                    help='path to dataset (e.g. data/')
parser.add_argument('--image_size', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('--batch_size', default=32, type=int,
                    metavar='N', help='batch size (default: 32)')
parser.add_argument('--adv_batch_size', default=10, type=int,
                    metavar='N', help='batch size ml_cw, ml_rank1, ml_rank2 18, ml_lp, mlae_de, ml_deepfool is 10')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--adv_method', default='mlae_de', type=str, metavar='N',
                    help='attack method: ml_cw, ml_rank1, ml_rank2, ml_deepfool, mla_lp, mlae_de')
parser.add_argument('--target_type', default='hide_all', type=str, metavar='N',
                    help='target method: hide_all,hide_single')
parser.add_argument('--adv_file_path', default='../data/voc2012/files/VOC2012/classification_mlgcn_adv.csv', type=str, metavar='N',
                    help='all image names and their labels ready to attack')
parser.add_argument('--adv_save_x', default='../adv_save/mlgcn/voc2012/', type=str, metavar='N',
                    help='save adversiral examples')
parser.add_argument('--adv_begin_step', default=0, type=int, metavar='N',
                    help='which step to start attacking according to the batch size')

def new_folder(file_path):
    folder_path = os.path.dirname(file_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def init_log(log_file):
  new_folder(log_file)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)

  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  ch.setFormatter(formatter)

  logger.addHandler(ch)
  logger.addHandler(fh)

def get_target_label(y, target_type):
    '''
    :param y: numpy, y in {0, 1}
    :param A: list, label index that we want to reverse
    :param C: list, label index that we don't care
    :return:
    '''
    y = y.copy()
    # o to -1
    y[y == 0] = -1
    if  target_type == 'hide_single':
        for i, y_i in enumerate(y):
            pos_idx = np.argwhere(y_i == 1).flatten()
            pos_idx_c = np.random.choice(pos_idx)
            y[i, pos_idx_c] = -y[i, pos_idx_c]
    elif target_type == 'hide_all':
            y[y == 1] = -1
    return y

def gen_adv_file(model, target_type, adv_file_path):
    print("generating……")
    tqdm.monitor_interval = 0
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    test_dataset = Voc2012Classification(args.data, 'val', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    test_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    image_name_list = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
            image_name_list.extend(list(input[1]))
        output = np.asarray(output)
        y = np.asarray(y)
        image_name_list = np.asarray(image_name_list)

    # choose x which can be well classified and contains two or more label to prepare attack
    pred = (output >= 0.5) + 0
    y[y==-1] = 0
    true_idx = []
    count = 0
    for i in range(len(pred)):
        if (y[i] == pred[i]).all() and np.sum(y[i]) >= 1 and count < 1000:
            true_idx.append(i)
            count += 1
    adv_image_name_list = image_name_list[true_idx]
    adv_y = y[true_idx]
    y = y[true_idx]
    y_target = get_target_label(adv_y, target_type)
    y_target[y_target==0] = -1
    y[y==0] = -1

    print(len(adv_image_name_list))
    adv_labeled_data = {}
    for i in range(len(adv_image_name_list)):
        adv_labeled_data[adv_image_name_list[i]] = y[i]
    write_object_labels_csv(adv_file_path, adv_labeled_data)

    # save target y and ground-truth y to prepare attack
    # value is {-1,1}
    np.save('../adv_save/mlgcn/voc2012/y_target.npy', y_target)
    np.save('../adv_save/mlgcn/voc2012/y.npy', y)

def evaluate_model(model):
    tqdm.monitor_interval = 0
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    test_dataset = Voc2012Classification(args.data, 'val', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    test_dataset.transform = data_transforms
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    output = []
    y = []
    test_loader = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            x = input[0]
            if use_gpu:
                x = x.cuda()
            o = model(x).cpu().numpy()
            output.extend(o)
            y.extend(target.cpu().numpy())
        output = np.asarray(output)
        y = np.asarray(y)

    pred = (output >= 0.5) + 0
    y[y == -1] = 0

    from utils import evaluate_metrics
    metric = evaluate_metrics.evaluate(y, output, pred)
    print(metric)

def evaluate_adv_(state):
    model = state['model']
    y_target = state['y_target']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x:int(x[13:-4]))
    adv = []
    for f in adv_file_list:
        adv.extend(np.load(adv_folder_path+f))
    adv = np.asarray(adv)
    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    adv_dataset = Voc2012Classification(args.data, 'mlgcn_adv', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm_1 = []
    norm = []
    max_r = []
    mean_r = []
    rmsd = []
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x[0][0].cpu().numpy()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x / 2 + 0.5) * 255) - ((batch_test_x / 2 + 0.5) * 255)
            batch_norm = [np.linalg.norm(r.flatten()) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            norm_1.extend(np.sum(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
    adv_output = np.asarray(adv_output)
    adv_pred = adv_output.copy()
    adv_pred[adv_pred >= (0.5+0)] = 1
    adv_pred[adv_pred < (0.5+0)] = -1
    print(adv_pred.shape)
    print(y_target.shape)
    adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
    attack_fail_idx = np.argwhere(adv_pred_match_target==0).flatten().tolist()
    np.save('{}_attack_fail_idx.npy'.format(args.adv_method), attack_fail_idx)
    norm = np.asarray(norm)
    max_r = np.asarray(max_r)
    mean_r = np.asarray(mean_r)
    rmsd = np.asarray(rmsd)
    norm = np.delete(norm, attack_fail_idx, axis=0)
    max_r = np.delete(max_r, attack_fail_idx, axis=0)
    norm_1 = np.delete(norm_1, attack_fail_idx, axis=0)
    mean_r = np.delete(mean_r, attack_fail_idx, axis=0)
    rmsd = np.delete(rmsd, attack_fail_idx, axis=0)

    from utils import evaluate_metrics
    metrics = dict()
    y_target[y_target==-1] = 0
    metrics['ranking_loss'] = evaluate_metrics.label_ranking_loss(y_target, adv_output)
    metrics['average_precision'] = evaluate_metrics.label_ranking_average_precision_score(y_target, adv_output)
    metrics['auc'] = evaluate_metrics.roc_auc_score(y_target, adv_output)
    metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
    metrics['norm'] = np.mean(norm)
    metrics['norm_1'] = np.mean(norm_1)
    metrics['rmsd'] = np.mean(rmsd)
    metrics['max_r'] = np.mean(max_r)
    metrics['mean_r'] = np.mean(mean_r)
    print()
    print(metrics)

def evaluate_adv(state):
    model = state['model']
    y_target = state['y_target']

    adv_folder_path = os.path.join(args.adv_save_x, args.adv_method, 'tmp/')
    adv_file_list = os.listdir(adv_folder_path)
    adv_file_list.sort(key=lambda x:int(x[16:-4]))
    adv = []
    for f in adv_file_list:
        adv.extend(np.load(adv_folder_path+f))
    adv = np.asarray(adv)

    dl1 = torch.utils.data.DataLoader(adv,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.workers)

    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    adv_dataset = Voc2012Classification(args.data, 'mlgcn_adv', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms
    dl2 = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)
    dl2 = tqdm(dl2, desc='ADV')

    adv_output = []
    norm = []
    max_r = []
    mean_r = []
    rmsd = []
    import matplotlib
    import matplotlib.image as plot_img
    show_idx = 52
    batch_idx = 0
    with torch.no_grad():
        for batch_adv_x, batch_test_x in zip(dl1, dl2):
            if show_idx // args.batch_size > batch_idx:
                batch_idx = batch_idx + 1
                continue
            show_idx = show_idx % args.batch_size

            if use_gpu:
                batch_adv_x = batch_adv_x.cuda()
                batch_test_x[0][0] = batch_test_x[0][0].cuda()
            adv_output.extend(model(batch_adv_x).cpu().numpy())
            output = model(batch_test_x[0][0]).cpu().numpy()

            batch_adv_x = batch_adv_x.cpu().numpy()
            batch_test_x = batch_test_x[0][0].cpu().numpy()


            plot_img.imsave('original.pdf', batch_test_x[show_idx].transpose(1,2,0))
            plot_img.imsave('{}_adv.pdf'.format(args.adv_method), batch_adv_x[show_idx].transpose(1,2,0))
            distortion = batch_adv_x[show_idx] - batch_test_x[show_idx]
            from mpl_toolkits.mplot3d import Axes3D
            plot_lim = 0.03
            if args.adv_method == 'ml_deepfool':
               plot_lim = 1
            Z = distortion[0]
            size = Z.shape
            Y = np.arange(0, size[0], 1)
            X = np.arange(0, size[1], 1)

            X, Y = np.meshgrid(X, Y)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Reds')
            ax.set_zlim(-plot_lim, plot_lim)
            ax.set_rasterized(True)
            plt.savefig('{}_distortion1.pdf'.format(args.adv_method), transparent=True)
            Z = distortion[1]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Greens')
            ax.set_zlim(-plot_lim, plot_lim)
            ax.set_rasterized(True)
            plt.savefig('{}_distortion2.pdf'.format(args.adv_method), transparent=True)

            Z = distortion[2]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Blues')
            ax.set_zlim(-plot_lim, plot_lim)
            ax.set_rasterized(True)
            plt.savefig('{}_distortion3.pdf'.format(args.adv_method), transparent=True)

            sys.exit()

            batch_r = (batch_adv_x - batch_test_x)
            batch_r_255 = ((batch_adv_x / 2 + 0.5) * 255) - ((batch_test_x / 2 + 0.5) * 255)
            batch_norm = [np.linalg.norm(r) for r in batch_r]
            batch_rmsd = [np.sqrt(np.mean(np.square(r))) for r in batch_r_255]
            norm.extend(batch_norm)
            rmsd.extend(batch_rmsd)
            max_r.extend(np.max(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
            mean_r.extend(np.mean(np.abs(batch_adv_x - batch_test_x), axis=(1, 2, 3)))
    adv_output = np.asarray(adv_output)
    adv_pred = adv_output.copy()
    adv_pred[adv_pred >= (0.5+0)] = 1
    adv_pred[adv_pred < (0.5+0)] = -1
    adv_pred_match_target = np.all((adv_pred == y_target), axis=1) + 0
    attack_fail_idx = np.argwhere(adv_pred_match_target==0).flatten()

    norm = np.asarray(norm)
    max_r = np.asarray(max_r)
    mean_r = np.asarray(mean_r)
    norm = np.delete(norm, attack_fail_idx, axis=0)
    max_r = np.delete(max_r, attack_fail_idx, axis=0)
    mean_r = np.delete(mean_r, attack_fail_idx, axis=0)

    from utils import evaluate_metrics
    metrics = dict()

    metrics['ranking_loss'] = evaluate_metrics.label_ranking_loss(y_target, adv_output)
    metrics['average_precision'] = evaluate_metrics.label_ranking_average_precision_score(y_target, adv_output)
    metrics['auc'] = evaluate_metrics.roc_auc_score(y_target, adv_output)
    metrics['attack rate'] = np.sum(adv_pred_match_target) / len(adv_pred_match_target)
    metrics['norm'] = np.mean(norm)
    metrics['rmsd'] = np.mean(rmsd)
    metrics['max_r'] = np.mean(max_r)
    metrics['mean_r'] = np.mean(mean_r)
    print()
    print(metrics)

def main():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    init_log(os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.log'))

    # define dataset
    num_classes = 20

    # load torch model
    model = gcn_resnet101_attack(num_classes=num_classes,
                                 t=0.4,
                                 adj_file='../data/voc2012/voc_adj.pkl',
                                 word_vec_file='../data/voc2012/voc_glove_word2vec.pkl',
                                 save_model_path='../checkpoint/mlgcn/voc2012/voc_checkpoint.pth.tar')
    model.eval()
    if use_gpu:
        model = model.cuda()
    if not os.path.exists(args.adv_file_path):
        gen_adv_file(model, args.target_type, args.adv_file_path)

    # transfor image to torch tensor
    # the tensor size is [chnnel, height, width]
    # the tensor value in [0,1]
    data_transforms = transforms.Compose([
        Warp(args.image_size),
        transforms.ToTensor(),
    ])
    adv_dataset = Voc2012Classification(args.data, 'mlgcn_adv', inp_name='../data/voc2012/voc_glove_word2vec.pkl')
    adv_dataset.transform = data_transforms
    adv_loader = torch.utils.data.DataLoader(adv_dataset,
                                              batch_size=args.adv_batch_size,
                                              shuffle=False,
                                              num_workers=args.workers)

    # load target y and ground-truth y
    # value is {-1,1}
    y_target = np.load('../adv_save/mlgcn/voc2012/y_target.npy')
    y = np.load('../adv_save/mlgcn/voc2012/y.npy')

    state = {'model': model,
             'data_loader': adv_loader,
             'adv_method': args.adv_method,
             'target_type': args.target_type,
             'adv_batch_size': args.adv_batch_size,
             'y_target':y_target,
             'y': y,
             'adv_save_x': os.path.join(args.adv_save_x, args.adv_method, args.target_type + '.npy'),
             'adv_begin_step': args.adv_begin_step
             }

    # start attack
    attack_model = AttackModel(state)
    attack_model.attack()
    evaluate_adv(state)


if __name__ == '__main__':
    main()