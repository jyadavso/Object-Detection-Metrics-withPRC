echo "Result  on d14_small_newanc2_i04_384_upscale"
python3 pascalvoc.py -gt sourabh_data/comparision/customer/groundtruths -det sourabh_data/comparision/customer/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
python3 pascalvoc.py -gt sourabh_data/comparision/generic_big/groundtruths -det sourabh_data/comparision/generic_big/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
python3 pascalvoc.py -gt sourabh_data/comparision/customer_mar20/groundtruths -det sourabh_data/comparision/customer_mar20/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
python3 pascalvoc.py -gt sourabh_data/comparision/wide_customer/groundtruths -det sourabh_data/comparision/wide_customer/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
python3 pascalvoc.py -gt sourabh_data/comparision/smartvue/groundtruths -det sourabh_data/comparision/smartvue/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
python3 pascalvoc.py -gt sourabh_data/comparision/voc_person/groundtruths -det sourabh_data/comparision/voc_person/efficientdet_d1_pytorch/d14_small_newanc2_i04_384_upscale -np
