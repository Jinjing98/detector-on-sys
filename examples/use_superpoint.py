# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/
from pathlib import Path
import sys
import pandas as pd
from termcolor import colored
from detectors_eva.SuperPoint.superpoint.datasets.sys_surgery import syth_dataset_tf
# from detectors_eva.kp2d.datasets.patches_dataset import syth_dataset
# from  detectors_eva.kp2d.evaluation.evaluate import evaluate_keypoint_net_syth_data
# from  detectors_eva.kp2d.networks.keypoint_net import KeypointNet
from detectors_eva.SuperPoint.superpoint.evaluations.evaluate import evaluate_keypoint_net_syth_data_sp


import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from detectors_eva.SuperPoint.superpoint.evaluations.evaluate import evaluate_keypoint_net_syth_data_sp
tf.config.set_visible_devices([], 'GPU')



#
#
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



#
# def preprocess_image(img_file):
#     img = cv2.imread(img_file, cv2.IMREAD_COLOR)
#
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = np.expand_dims(img, 2)
#     img = img.astype(np.float32)
#     img_preprocessed = img / 255.
#
#     return img_preprocessed
#


# def main():
#
#
#     warp_img_dir = "/home/jinjing/Projects/data/out_imgs/"
#     final_ori_img_dir = "/home/jinjing/Projects/data/final_ori_imgs/"
#
#
#     weights_dir = Path('/home/jinjing/Projects/detector_sysdata/detectors_eva/SuperPoint/pretrained_models',sp_v6.tgz)
#
#     # load model
#     graph = tf.Graph()
#     with tf.Session(graph=graph) as sess:
#         tf.saved_model.loader.load(sess,
#                                    [tf.saved_model.tag_constants.SERVING],#r"E:\Google Drive\\files.sem3\NCT\Reuben_lab\keypoint_detector_descriptor_evaluator-main\models\SuperPoint\pretrained_models\sp_v6.tar")
#                                    str(weights_dir))
#
#         input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
#         output_prob_nms_tensor = graph.get_tensor_by_name(  # nms is chosen here?
#             'superpoint/prob_nms:0')
#         output_desc_tensors = graph.get_tensor_by_name(
#             'superpoint/descriptors:0')
#
#         conf_trd_sets = [0,0.1,0.2,0.4]
#         top_k_points = [50,100,200,300,500]
#
#         columns = ["top_k","N1","N2","repeat","loc_error","fail_cnt","success_cnt","avg_err"]
#         df = pd.DataFrame(columns= columns)
#
#         for confidence in conf_trd_sets:
#             for k in top_k_points:
#
#
#
#                 sys.exit()
#
#                 rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_SP(final_ori_img_dir,warp_img_dir,sess,k,input_img_tensor,
#                                          output_prob_nms_tensor,output_desc_tensors,confidence )
#
#                 print(colored('Evaluating for super point: confidence {} k_points {}'.format(confidence,k),'green'))
#                 print('Repeatability {0:.3f}'.format(rep))
#                 print('Localization Error {0:.3f}'.format(loc))
#                 print('Correctness d1 {:.3f}'.format(c1))
#                 print('Correctness d3 {:.3f}'.format(c3))
#                 print('Correctness d5 {:.3f}'.format(c5))
#                 print('MScore {:.3f}'.format(mscore))
#
#
#                 df_curr = pd.DataFrame([[k,confidence,rep,loc,c1,c3,c5,mscore]],
#                           columns=["top_k","conf_trd","repeat","loc_error","C_d1","C_d3","C_d5","Match_Score"])
#                 df = df.append(df_curr, ignore_index=True)
#
#         with pd.ExcelWriter("/home/jinjing/Projects/data_old/new_data/output"+'/arti_results.xlsx',
#                             mode='a') as writer:
#             df.to_excel(writer, sheet_name='superpoint')
def main2():



    warp_img_dir = "/home/jinjing/Projects/new_data/dataset_cholec/"#idx_video_frame
    final_ori_img_dir =warp_img_dir
    EXPER_PATH="/home/jinjing/Projects/keypoints_comparision/pretrained_models"



    weights_name = "sp_v6"#args.weights_name""

    weights_root_dir = Path(EXPER_PATH, 'saved_models')
    weights_root_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(weights_root_dir, weights_name)

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



        # inti dataset

        dataset_dir = '/home/jinjing/Projects/new_data/dominik_data/'
        hp_dataset = syth_dataset_tf(root_dir=dataset_dir, use_color=True)
        generators = hp_dataset.get_training_set()
        # print(generators.__next__()['src_norm'])
        # print('nnnnnnnnnnn')


        # top_k_points = [50,100,200,400]
        # df = pd.DataFrame(columns=["top_k","conf_trd","repeat","loc_error","C_d1","C_d3","C_d5","Match_Score"])
        top_ks = [50,100,200,400]
        for k in top_ks:
            print(colored(f'Evaluating for -- top_k {k}','green'))

            rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_syth_data_sp(warp_img_dir,sess,k,input_img_tensor,
                                                     output_prob_nms_tensor,output_desc_tensors,generators)
            sys.exit()


            # N1, N2, rep, loc, fail_cnt,avg_err, success_cnt= evaluate_keypoint_net_syth_data(
            #     data_loader=data_loader,
            #     keypoint_net=keypoint_net,
            #     top_k=top_k,# use confidence?
            #     use_color=True,
            # )


        #
        #
        #
        #
        #
        #         rep, loc, c1, c3, c5, mscore = evaluate_keypoint_net_SP2(final_ori_img_dir,warp_img_dir,sess,k,input_img_tensor,
        #                                  output_prob_nms_tensor,output_desc_tensors)
        #         print(colored('Evaluating for super point: confidence {} k_points {}'.format(confidence,k),'green'))
        #         print('Repeatability {0:.3f}'.format(rep))
        #         print('Localization Error {0:.3f}'.format(loc))
        #         print('Correctness d1 {:.3f}'.format(c1))
        #         print('Correctness d3 {:.3f}'.format(c3))
        #         print('Correctness d5 {:.3f}'.format(c5))
        #         print('MScore {:.3f}'.format(mscore))
        #
        #
        #
        #         df_curr = pd.DataFrame([[k,confidence,rep,loc,c1,c3,c5,mscore]],
        #                   columns=["top_k","conf_trd","repeat","loc_error","C_d1","C_d3","C_d5","Match_Score"])
        #         df = df.append(df_curr, ignore_index=True)
        #
        # with pd.ExcelWriter("/home/jinjing/Projects/data_old/new_data/output"+'/temp_results.xlsx',
        #                     mode='a') as writer:
        #     df.to_excel(writer, sheet_name='superpoint')
        #




if __name__ == '__main__':
    # main()
    main2()

