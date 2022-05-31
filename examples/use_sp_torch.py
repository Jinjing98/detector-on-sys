from torch.utils.data import DataLoader, Dataset
import pandas as pd
from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
from detectors_eva.utils.args_init import init_args
from detectors_eva.utils.util_result import write_excel
from termcolor import colored
from detectors_eva.SuperPoint_torch.evaluations_sp_torch.evaluate import evaluate_keypoint_net_syth_data_sp
from detectors_eva.SuperPoint_torch.evaluations_sp_torch.superpoint_net import get_sp_model
args, unknown = init_args().parse_known_args()
dataset_dir = args.dataset_prefix
sp_model = get_sp_model(args)

def main():


    # init data
    hp_dataset = syth_dataset(root_dir=dataset_dir, use_color=True)
    data_loader = DataLoader(hp_dataset,
                             batch_size=args.batch_size,
                             pin_memory=False,
                             shuffle=False,
                             num_workers=8,
                             worker_init_fn=None,
                             sampler=None)

    top_ks = args.topk_SP
    columns = args.cols_name
    df = pd.DataFrame(columns= columns)
    for top_k in top_ks:
        print(colored(f'Evaluating for -- top_k {top_k}','green'))
        N1, N2, rep, loc, fail_cnt,avg_err, success_cnt = evaluate_keypoint_net_syth_data_sp(
            data_loader=data_loader,
            sp_model=sp_model,
            top_k=top_k,# use confidence?
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

        df_curr = pd.DataFrame([[None,None,top_k,N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
                  columns=columns)
        df = df.append(df_curr, ignore_index=True)
    write_excel(args.result_path,'superpoint_torch',df)


if __name__ == '__main__':
    main()  # this function is for man made version images
