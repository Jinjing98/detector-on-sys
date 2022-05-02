import torch
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import argparse
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
from  detectors_eva.kp2d.evaluation.evaluate import evaluate_keypoint_net_syth_data
from  detectors_eva.kp2d.networks.keypoint_net import KeypointNet




def main2():

    # dataset_dir = "/home/jinjing/Projects/data/out_imgs/"
    dataset_dir = '/home/jinjing/Projects/new_data/dominik_data/'


    parser = argparse.ArgumentParser(
        description='Script for KeyPointNet testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pretrained_model", type=str, help="pretrained model path",
                        default="/home/jinjing/Projects/keypoints_comparision/git_src_code/kp2d/pretrained_models/v4.ckpt")
    # parser.add_argument("--input_dir", type=str, help="Folder containing input images",
    #                     default="/home/jinjing/Projects/data/out_imgs/")

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
    img_size = (0,0)  # doesn't matter; the size will be generated internally
    eval_params = [
                {  'top_k': 50, "res":img_size, 'conf_threshold':0},
                {  'top_k': 100, "res":img_size, 'conf_threshold':0}, #H W of cropped img
                {  'top_k': 200, "res":img_size, 'conf_threshold':0}, #H W of cropped img
                {  'top_k': 500, "res":img_size, 'conf_threshold':0},

                {  'top_k': 50, "res":img_size, 'conf_threshold':0.1}, #H W of cropped img
                {  'top_k': 100, "res":img_size, 'conf_threshold':0.1}, #H W of cropped img
                {  'top_k': 200, "res":img_size, 'conf_threshold':0.1},
                {  'top_k': 400, "res":img_size, 'conf_threshold':0.1}, #H W of cropped img

                {  'top_k': 50, "res":img_size, 'conf_threshold':0.2}, #H W of cropped img
                {  'top_k': 100, "res":img_size, 'conf_threshold':0.2}, #H W of cropped img
                {  'top_k': 200, "res":img_size, 'conf_threshold':0.2},
                {  'top_k': 400, "res":img_size, 'conf_threshold':0.2}, #H W of cropped img

                {  'top_k': 50, "res":img_size, 'conf_threshold':0.4}, #H W of cropped img
                {  'top_k': 100, "res":img_size, 'conf_threshold':0.4}, #H W of cropped img
                {  'top_k': 200, "res":img_size, 'conf_threshold':0.4},
                {  'top_k': 400, "res":img_size, 'conf_threshold':0.4}, #H W of cropped img
    ]


    df = pd.DataFrame(columns=["top_k","conf_trd","repeat","loc_error","C_d1","C_d3","C_d5","Match_Score"])
    hp_dataset = syth_dataset(root_dir=dataset_dir, use_color=True)
    data_loader = DataLoader(hp_dataset,
                             batch_size=30,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=8,
                             worker_init_fn=None,
                             sampler=None)

    for params in eval_params:


        print(colored('Evaluating for -- top_k {} conf_trd {}'.format( params['top_k'],params['conf_threshold']),'green'))
        N1, N2, rep, loc, fail_cnt,avg_err, success_cnt= evaluate_keypoint_net_syth_data(
            data_loader=data_loader,
            keypoint_net=keypoint_net,
            output_shape=params['res'],
            top_k=params['top_k'],
            use_color=True,
            conf_threshold=params['conf_threshold'],
        )
        print('N1 {0:.3f}'.format(N1))
        print('N2 {0:.3f}'.format(N2))
        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('fail count {:.3f}'.format(fail_cnt))
        print('success count {:.3f}'.format(success_cnt))
        print('avg err {:.3f}'.format(avg_err))
        # print('MScore {:.3f}'.format(mscore))


        df_curr = pd.DataFrame([[params['top_k'],params['conf_threshold'],N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
                  columns=["top_k","conf_trd","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"])
        df = df.append(df_curr, ignore_index=True)
        # print("debug:",df.shape)

    with pd.ExcelWriter("/home/jinjing/Projects/data"+'/12.xlsx',
                        mode='a') as writer:
        df.to_excel(writer, sheet_name='kp2d')





if __name__ == '__main__':


    main2()  # this function is for man made version images
