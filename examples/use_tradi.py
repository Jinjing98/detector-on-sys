from termcolor import colored
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
from  detectors_eva.tradi_orb.evaluation.evaluate import evaluate_keypoint_net_syth_data_all
from detectors_eva.utils.args_init import init_args
from detectors_eva.utils.util_result import write_excel


def compute_metrics(method,data_loader,args):

    if method == 'orb':
        set1 = args.num4features_set
        set2 = args.fast_trd_set
    elif method == 'AKAZE':
        set1 = args.trd_set
        set2 = args.diff_type_set
    elif method == 'agast_sift':
        set1 = args.agast_type_set
        set2 = args.agast_trd_set

    df = pd.DataFrame(columns= args.cols_name)
    for param1 in set1:
        for param2 in set2:
            param_tuple = (param1,param2)
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

            df_curr = pd.DataFrame([[param1,param2,None,N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
                      columns=args.cols_name)
            df = df.append(df_curr, ignore_index=True)




    write_excel(args.result_path,method,df)
    # with pd.ExcelWriter(args.result_path,
    #                     mode='a') as writer: # mode = 'wa' /'w'
    #     df.to_excel(writer, sheet_name=method)

def main():
    args, unknown = init_args().parse_known_args()
    dataset_dir = args.dataset_prefix

    # init data
    hp_dataset = syth_dataset(root_dir=dataset_dir, use_color=True)
    data_loader = DataLoader(hp_dataset,
                             batch_size=args.batch_size,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=8,
                             worker_init_fn=None,
                             sampler=None)

    for method in args.methods_set:
        compute_metrics(method, data_loader, args)

if __name__ == '__main__':


    main()  # this function is for man made version images
