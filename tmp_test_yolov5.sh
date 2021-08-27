echo "Result  on yolov5/yolov5m_ld_torch_14"
python3 pascalvoc.py -gt sourabh_data/comparision/customer/groundtruths -det sourabh_data/comparision/customer/yolov5/yolov5m_ld_torch_14 -np
python3 pascalvoc.py -gt sourabh_data/comparision/generic_big/groundtruths -det sourabh_data/comparision/generic_big/yolov5/yolov5m_ld_torch_14 -np
python3 pascalvoc.py -gt sourabh_data/comparision/customer_mar20/groundtruths -det sourabh_data/comparision/customer_mar20/yolov5/yolov5m_ld_torch_14 -np
python3 pascalvoc.py -gt sourabh_data/comparision/wide_customer/groundtruths -det sourabh_data/comparision/wide_customer/yolov5/yolov5m_ld_torch_14 -np
python3 pascalvoc.py -gt sourabh_data/comparision/smartvue/groundtruths -det sourabh_data/comparision/smartvue/yolov5/yolov5m_ld_torch_14 -np
python3 pascalvoc.py -gt sourabh_data/comparision/voc_person/groundtruths -det sourabh_data/comparision/voc_person/yolov5/yolov5m_ld_torch_14 -np
