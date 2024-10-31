python infer.py \
    --sdxl_path /share/huangrenyuan/model_zoo/diffusion/stable-diffusion-xl-base-1.0 \
    --vision_encoder_path /share/huangrenyuan/model_zoo/vis_backbone/dinov2_large \
    --instantir_path /share2/huangrenyuan/model_zoo/models/models \
    --num_inference_steps 30 \
    --batch_size 1 \
    --out_path /share/huangrenyuan/haofan_test_out \
    --test_path /share/huangrenyuan/haofan_test \
    --post_fix test_7.0
