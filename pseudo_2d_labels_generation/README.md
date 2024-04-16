```shell
├── s1_create_json.py                  -- create jsons files
├── s2_det2_bbox_inference.py          -- run the Keypoint-RCNN using detectron2
├── s3_create_pseudo_bboxes.py         -- create the pseudo 2d bounding boxes 
├── s4_hrnet_kpt2d_inference.sh        -- run the HRNet model for 2d pose estimation
├── s5_create_pseudo_kpt2d.py          -- create pseudo 2d poses
├── s6_vis_pseudo_kpt2d.py             -- visualize pseudo 2d poses
├── s7_create_pseudo_kpt2d_dbpickle.py -- create the pickle files 
└── s8_vis_compare_pseudo_kpt2d.py     -- compare pseudo and gt 2d poses
```