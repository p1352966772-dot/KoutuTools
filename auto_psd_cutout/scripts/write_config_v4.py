import sys, os
sys.stdout.reconfigure(encoding="utf-8")

content = """input_dir: "input"
output_dir: "output"

detect:
  # Multi-mask fusion
  white_threshold: 225
  adaptive_block: 31
  adaptive_c: 2
  canny_low: 50
  canny_high: 150

  # Multi-scale morphology
  sm_kernel: 5
  md_kernel: 13
  lg_kernel: 25

  # Row detection
  l1_row_kernel: 30
  row_gap_min: 12

  # Text-aware merging
  text_max_height_pct: 0.08
  text_max_area_pct: 0.005
  text_y_alignment_tol: 10
  text_merge_gap: 30

  # Aggressive merge
  merge_h_gap: 20
  merge_y_overlap_threshold: 0.60
  merge_iou: 0.30

  # Character suppression
  min_char_size: 14

  # Small object protection (low threshold)
  min_element_size: 10
  min_area_ratio: 0.0005
  max_aspect_ratio: 12.0
  min_aspect_ratio: 0.05

  # Two-stage detection
  two_stage_min_size: 150

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
