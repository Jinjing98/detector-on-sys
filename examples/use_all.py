import torch
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
from  detectors_eva.tradi_orb.evaluation.evaluate import evaluate_keypoint_net_syth_data_all




def main():

    dataset_dir = '/home/jinjing/Projects/new_data/dominik_data/'


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
    columns = ["top_k","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"]
    df = pd.DataFrame(columns= columns)

    method = 'agast_sift'
    agast_type = '5_8'
    trd = 5 #[5,10,15,20,25]
    param_tuple = (agast_type,trd)

    # method = 'orb'
    # num4features = 200
    # fast_trd = 5  #similar to topk; num of features is num4features if trd is 0
    # param_tuple = (num4features,fast_trd)


    # method = 'AKAZE'
    # trd = 1e-4 #  [1e-4,5e-4,25e-4,125e-4]   # affaect the num of detected poitns
    # diff_type = 0 # 0,1,2,3
    # param_tuple = (trd,diff_type)

    for top_k in top_ks:
        print(colored(f'Evaluating for {method} -- params {param_tuple}','green'))
        N1, N2, rep, loc, fail_cnt,avg_err, success_cnt= evaluate_keypoint_net_syth_data_all(
            method = method,
            param = param_tuple,
            data_loader=data_loader,
            # top_k=top_k,# use confidence?
            use_color=True,
            vis_flag=False
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
        df.to_excel(writer, sheet_name=method)





if __name__ == '__main__':


    main()  # this function is for man made version images
