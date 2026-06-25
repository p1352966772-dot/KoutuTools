import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """input_dir: "input"
output_dir: "output"

detect:
  # Multi-scale UI detection
  white_threshold: 240

  # L1: Coarse layout (row detection)
  l1_row_kernel: 35
  row_gap_min: 15

  # L2: Fine element detection
  l2_element_kernel: 10
  col_gap_min: 10

  # Merge strategy
  merge_h_gap: 30
  merge_v_tol: 15
  merge_iou: 0.25
  merge_aspect_ratio_tol: 0.20

  # Noise filtering
  min_area_ratio: 0.0015
  min_element_size: 20
  max_aspect_ratio: 10.0
  min_aspect_ratio: 0.1

ocr_text_cleanup:
  enabled: true
  min_confidence: 0.55
  padding: 4
  fill_entire_box: true
  low_saturation_only: true
  saturation_max: 100
  dark_threshold: 190

label_text_cleanup:
  enabled: false
  full_page: true
  region_left_width: 170
  region_top_height: 45
  dark_threshold: 120
  saturation_max: 80
  max_component_area: 260
  max_component_width: 36
  max_component_height: 26
  protect_saturation_min: 90
  protect_color_radius: 8
  dilate: 1

preview:
  draw_box: true
  draw_index: true
  draw_groups: true
  draw_rows: true
  box_thickness: 2

photoshop:
  enabled: true
  visible: true
  add_original_layer: false
  add_box_layers: false
  box_color: [255, 0, 0]
  box_thickness: 2
  psd_name_suffix: "_auto"
"""

dst = os.path.expandvars(r"%USERPROFILE%\Desktop\KoutuTools\auto_psd_cutout\config.yaml")
with open(dst, "w", encoding="utf-8") as f:
    f.write(content)
print("config.yaml written OK")
