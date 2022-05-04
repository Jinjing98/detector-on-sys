1. first run generate_ofnpy_ovimg for preparation before evluation.
note: some times the evalute has error is due to the groud truth data is damaged, so just regenerate to see if it fix the bug.
2. .idea dir comes with  cmd  'python develp setup.py'; since setup.py is under root; .idea is also generated there.
3. agast sift only can run with opencv_env(since it requir a spercfic opencv): 
this env installed with torch(we use torch dataset to do batching):
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
flaw: can not do vis here.
4. kp2d can run in flownet and opencv_env
5. superpoint can run in test_tf
