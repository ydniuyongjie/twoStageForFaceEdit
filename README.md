Our work is divided into four steps:the first step is model training;the second step is latent space traversal;the third step is attribute traversal; the fourth step is editing direction semantic identification and quantitative evaluation of disentanglement.
## 1. Train model for five methods
### Train single discovery dirextion(SDD) 
please run "train_Latent.py" file.Run the command in the "train_latent" section of the launch.json file.
### Train single Hessian Penalty(SHP) 
please run "hessian_penalty_stylegan.py" file.Run the command in the "hessian_penalty" section of the launch.json file.
## Train Hybrid Discovery and Penalty (HDP) 
please run "train_Mix_Latent.py" file.Run the command in the "train_Mix" section of the launch.json file.
### Train Discovery To OroJaR(DTJ) 
please run "line_2_orojarpenalty.py" file.Run the command in the "line_2_orojar" section of the launch.json file.
### Train Discovery to Hessian Penalty(DTH) 
please run "line_2_hessianpenalty.py" file.Run the command in the "line_2_hessian" section of the launch.json file.
## 2.Traversing the latent space
please run "traverse_line_latent.py",The relevant configuration parameters are in section "traverse_line_latent" of file 
launch.json.
## 3.Traversing the attribute space
please run "traverse_attribute_space.py",The relevant configuration parameters are in section "tra_attr" of file 
launch.json.
## 4.Editing direction semantic identification and quantitative evaluation of disentanglement.
please run "rank_interpretable_paths.py",The relevant configuration parameters are in section "Inter_path" of file 
launch.json.
## 5.Result  and GIF
- Method DTH edits result of au_12_Lip_Corner_Puller property in direction 86
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/gif/au_12_Lip_Corner_Puller_162_SDD.gif)
- Method SHP edits result of au_12_Lip_Corner_Puller property in direction 98
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/gif/au_12_Lip_Corner_Puller_98_SHP.gif)
- Method HDP edits result of au_12_Lip_Corner_Puller property in direction 170
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/gif/au_12_Lip_Corner_Puller_170_HDP.gif)
- Method DTJ edits result of au_12_Lip_Corner_Puller property in direction 6
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/gif/au_12_Lip_Corner_Puller_6_DTJ.gif)
- Method DTH edits result of au_12_Lip_Corner_Puller property in direction 86
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/gif/au_12_Lip_Corner_Puller_86_DTH.gif)
- Comparison of SDD and HDP accuracy
![](https://github.com/ydniuyongjie/twoStageForFaceEdit/blob/main/figs/accuracy.png)
## Related PDF Papers
The relevant papers have not been formally published and are not readily available to the public.