from pathlib import Path
import pandas as pd
from termcolor import colored
from detectors_eva.SuperPoint.superpoint.datasets.sys_surgery import syth_dataset_tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from detectors_eva.SuperPoint.superpoint.evaluations.evaluate import evaluate_keypoint_net_syth_data_sp
tf.config.set_visible_devices([], 'GPU')
from detectors_eva.utils.args_init import init_args

#below line of code will display the log for debug to see if it is real GPU
# To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

# below code may be needed if we want the code to run on GPU correctly.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            #below is an important line of code for enabling gpu
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs are: ",gpus, "  ",len(logical_gpus), "Logical GPUs are :",logical_gpus)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def main():
    args, unknown = init_args().parse_known_args()
    weights_dir = args.sp_model_path
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],#r"E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models\sp_v6.tar")
                                   str(weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name(
            'superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name(
            'superpoint/descriptors:0')

        top_ks = args.topk_SP
        columns = args.cols_name
        df = pd.DataFrame(columns= columns)

        for top_k in top_ks:
            print(colored(f'Evaluating for -- top_k {top_k}','green'))
            # inti dataset
            dataset_dir = '/home/jinjing/Projects/new_data/dominik_data/'
            # need to reinit every epoch; since the last generator arrived its end in last loop
            hp_dataset = syth_dataset_tf(root_dir=dataset_dir, use_color=True)
            generators = hp_dataset.get_training_set()
            data_size = hp_dataset.data_size

            N1, N2, rep, loc, fail_cnt,avg_err, success_cnt = evaluate_keypoint_net_syth_data_sp(sess,top_k,input_img_tensor,
                                                     output_prob_nms_tensor,output_desc_tensors,generators,data_size,vis_flag= False)

            print('N1 {0:.3f}'.format(N1))
            print('N2 {0:.3f}'.format(N2))
            print('Repeatability {0:.3f}'.format(rep))
            print('Localization Error {0:.3f}'.format(loc))
            print('fail count {:.3f}'.format(fail_cnt))
            print('success count {:.3f}'.format(success_cnt))
            print('avg err {:.3f}'.format(avg_err))

            df_curr = pd.DataFrame([[top_k,N1,N2,rep,loc,fail_cnt,success_cnt,avg_err]],
                      columns=columns)
            df = df.append(df_curr, ignore_index=True)

        with pd.ExcelWriter(args.result_path,
                            mode='a') as writer: # mode = 'wa' /'w'
            df.to_excel(writer, sheet_name='superpoint')

if __name__ == '__main__':
    main()

