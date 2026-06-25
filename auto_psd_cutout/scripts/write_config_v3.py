import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """input_dir: "input"
output_dir: "output"

detect:
  white_threshold: 240

  # Row detection
  l1_row_kernel: 35
  row_gap_min: 15

  # Text-aware merging
  text_max_height_pct: 0.08
  text_max_area_pct: 0.005
  text_y_alignment_tol: 10
  text_merge_gap: 30

  # Block merging
  merge_h_gap: 25
  merge_y_overlap_threshold: 0.70
  merge_iou: 0.20

  # Character suppression
  min_char_size: 18
  min_element_size: 20
  max_aspect_ratio: 8.0
  min_aspect_ratio: 0.1
  min_area_ratio: 0.0015

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
