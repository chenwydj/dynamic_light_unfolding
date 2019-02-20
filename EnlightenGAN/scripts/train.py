import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--test", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

'''
single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1
--fineSize 640 \
--patchSize 64 \
--n_layers_D 5 # 5 for 640, 4 for 320, 3 for 160, 2 for 80\
--n_layers_patchD 4 \
'''

'''
single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1_360px_align

		--which_model_netG sid_unet_resize \
'''

'''
--name bdd.images_day100.105.night0.50_G.unet4.resblk6_latent.gamma_D.layer3.2_vgg1_180px_align
--name bdd.finetune_seg5.aux.noDloss.lr1e4_day100.105.night0.75_G.unet4.resblk6_latent.gamma_D.layer3.2_vgg0.5_180px_align \
--name bdd.images.DA_day100.105.night0.75_G.unet4.resblk6_latent.gamma_D.layer3.2_vgg0.5_90px_align \
--continue_train: copy last .pth files to new name_folder
--which_epoch 200
'''

if opt.train:
	os.system("python EnlightenGAN/train.py \
		--dataroot /ssd1/chenwy/bdd100k/light_enhance_AB/seg_85/ \
		--no_dropout \
		--name bdd.seg10_day110.125.night55.70_G.segargmax.conf11.edge.unet4.resblk6_latent.gamma_D.boundary.layer3.2_vgg0.3_180px_align \
		--model single \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_res_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 7 \
        --n_layers_D 3 \
        --n_layers_patchD 2 \
		--fineSize 180 \
        --patchSize 60 \
		--resize_or_crop='no' \
		--skip 1 \
		--batchSize 15 \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --self_attention \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0,1,2 \
		--save_epoch_freq 20\
		--display_port=" + opt.port)

elif opt.test:
	for i in range(20):
	        os.system("python test.py \
	        	--dataroot /ssd1/chenwy/light \
	        	--name single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1 \
	        	--model single \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode pair \
	        	--which_model_netG sid_unet_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
	        	--instance_norm 0 \
                --times_residual \
	        	--which_epoch " + str(i*5+100))

# \
# --which_model_netD no_norm_4 \
# --patchD \
# --patch_vgg \
# --patchD_3 5 \
# --n_layers_D 5 \
# --n_layers_patchD 4 \
# --fineSize 640 \
# --patchSize 64 \
# \
elif opt.predict:
	os.system("python EnlightenGAN/predict.py \
		--dataroot /ssd1/chenwy/bdd100k/images/100k/val \
		--name single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1 \
		--model single \
		--which_direction AtoB \
		--no_dropout \
		--dataset_mode unaligned \
		--which_model_netG sid_unet_res_resize \
		\
		--skip 1 \
		--use_norm 1 \
		--use_wgan 0 \
        --self_attention \
        --times_residual \
		--instance_norm 0 --resize_or_crop='no'\
		--which_epoch latest")
	# for i in range(3):
	#         os.system("python predict.py \
	#         	--dataroot /ssd1/chenwy/light \
	#         	--name single_unet_conv_add_vary_attention_Tresidual_bs32_BN_nonormDlayer5_4_final_ragan_lsgan_32D_PV_5_vgg_relu5_1 \
	#         	--model single \
	#         	--which_direction AtoB \
	#         	--no_dropout \
	#         	--dataset_mode unaligned \
	#         	--which_model_netG sid_unet_resize \
	#         	--skip 1 \
	#         	--use_norm 1 \
	#         	--use_wgan 0 \
        #         --self_attention \
        #         --times_residual \
	#         	--instance_norm 0 --resize_or_crop='no'\
	#         	--which_epoch " + str(200 - i*5))
