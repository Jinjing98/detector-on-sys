import torch
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
from  detectors_eva.kp2d.evaluation.evaluate import evaluate_keypoint_net_syth_data
from  detectors_eva.kp2d.networks.keypoint_net import KeypointNet




def main():

    # dataset_dir = "/home/jinjing/Projects/data/out_imgs/"
    dataset_dir = '/home/jinjing/Projects/new_data/dominik_data/'


    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path",
                        default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    args = parser.parse_args()
    checkpoint = torch.load(args.pretrained_model)
    model_args = checkpoint['config']['model']['params']

    # Create and load disp net
    keypoint_net = KeypointNet(use_color=model_args['use_color'],
                               do_upsample=model_args['do_upsample'],
                               do_cross=model_args['do_cross'])
    keypoint_net.load_state_dict(checkpoint['state_dict'])
    keypoint_net = keypoint_net.cuda()
    keypoint_net.eval()
    print('Loaded KeypointNet from {}'.format(args.pretrained_model))
    print('KeypointNet params {}'.format(model_args))
    columns = ["top_k","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"]
    df = pd.DataFrame(columns= columns)
    # init data
    hp_dataset = syth_dataset(root_dir=dataset_dir, use_color=True)
    data_loader = DataLoader(hp_dataset,
                             batch_size=30,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=8,
                             worker_init_fn=None,
                             sampler=None)


    #
    top_ks = [50,100,200,400]
    for top_k in top_ks:
        print(colored(f'Evaluating for -- top_k {top_k}','green'))
        N1, N2, rep, loc, fail_cnt,avg_err, success_cnt= evaluate_keypoint_net_syth_data(
            data_loader=data_loader,
            keypoint_net=keypoint_net,
            top_k=top_k,# use confidence?
            use_color=True,
        )



        print('N1 {0:.3f}'.format(N1))
        print('N2 {0:.3f}'.format(N2))
        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('fail count {:.3f}'.format(fail_cnt))
        print('success count {:.3f}'.format(success_cnt))
        print('avg err {:.3f}'.format(avg_err))
        # print('MScore {:.3f}'.format(mscore))


        df_curr = pd.DataFrame([[top_k,N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
                  columns=columns)
        df = df.append(df_curr, ignore_index=True)

    with pd.ExcelWriter("/home/jinjing/Projects/detector_sysdata/results/"+'eval.xlsx',
                        mode='a') as writer: # mode = 'wa' /'w'
        df.to_excel(writer, sheet_name='kp2d')





if __name__ == '__main__':


    main()  # this function is for man made version images
