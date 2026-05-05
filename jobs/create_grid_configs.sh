#!/bin/bash
# Generate 48 grid-search YAML configs from the base config.
# Grid: pairing_threshold (4) × mask_margin (3) × abs_diff_boxes (2) × physical_padding_ratio (2) = 48

set -e

BASE_CONFIG="configs/run_all.yaml"
GRID_DIR="configs/grid_search"

pairing_thresholds=(0.2 0.3 0.4 0.5)
mask_margins=(0.0 0.1 0.2)
abs_diff_settings=(0 1)
physical_padding_ratios=(0.5 1.0)

mkdir -p "$GRID_DIR"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: Base config not found at $BASE_CONFIG"
  exit 1
fi

BOXES_PROMPT='You are a Holistic Image Consistency Inspector.
Your goal is to detect VISUAL INCONSISTENCIES or HALLUCINATIONS that are **clearly visible to the human eye**.

INPUTS:
- Image 1: Reference Image (Ground Truth).
- Image 2: Generated Output.

**IMPORTANT NOTE ON MASKS:**
Both images contain foreground objects masked out in SOLID BLACK.
**You must completely IGNORE these black masked areas.** Do not compare them. Focus ONLY on the visible content surrounding the masks.

YOUR MISSION:
**CRITICAL VISUAL GUIDE: Red bounding boxes have been drawn to highlight areas of potential change. Scrutinize the areas inside these red rectangles first, then check the rest of the image.**

Compare Image 2 against Image 1.
Compare the visible background textures, patterns, and objects in Image 2 against Image 1.
Ignore lighting/shadow differences caused by object insertion - DO NOT COMPARE SHADOWS.
**Focus only on meaningful visual changes.** Do not flag imperceptible pixel noise or compression artifacts.

DEFINITIONS OF HALLUCINATIONS (Look for these visible errors):
- Background Mutation: The scene is recognizable, but details changed incorrectly, structural elements vanished (e.g., wall color changed, carpet pattern altered, furniture shape morphed).
- Context Swap: The environment is completely different (e.g., bedroom became a living room).

OUTPUT RULES:
- If you find a visible error, write a concise description (1-2 sentences) explaining exactly what changed (e.g., "Background Mutation: The wooden floor turned into tiles.").
- **DO NOT mention the red bounding boxes.** They are technical guides for your eyes only; do not include them in your text description.
- Do not add any intro or outro text.'

echo "Generating grid search configurations..."
echo "Base config: $BASE_CONFIG"
echo "Output directory: $GRID_DIR"
echo

counter=0
for threshold in "${pairing_thresholds[@]}"; do
  for margin in "${mask_margins[@]}"; do
    for abs_diff in "${abs_diff_settings[@]}"; do
      for padding in "${physical_padding_ratios[@]}"; do
        counter=$((counter + 1))

        abs_diff_enabled=$([ "$abs_diff" -eq 1 ] && echo "True" || echo "False")
        abs_diff_label=$([ "$abs_diff" -eq 1 ] && echo "boxes" || echo "noboxes")

        config_name="grid_${counter}_th${threshold}_mg${margin}_${abs_diff_label}_pad${padding}.yaml"
        output_dir="outputs/grid_search/run_${counter}_th${threshold}_mg${margin}_${abs_diff_label}_pad${padding}"
        config_file="$GRID_DIR/$config_name"

        python3 << PYTHON_EOF
import yaml

with open("$BASE_CONFIG") as f:
    config = yaml.safe_load(f)

config['output_dir'] = '$output_dir'
config['segmentation']['pairing_threshold'] = $threshold
config['background']['mask_margin_percent'] = $margin
config['background']['absolute_difference_config']['enabled'] = $abs_diff_enabled
config['physical']['padding_ratio'] = $padding

if $abs_diff_enabled:
    config['background']['system_prompt'] = """$BOXES_PROMPT"""

with open('$config_file', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
PYTHON_EOF

        echo "[$counter/48] Created: $config_name"
      done
    done
  done
done

echo
echo "Generated $counter configuration files in $GRID_DIR"
echo "Run with: sbatch jobs/run_grid_search_parallel.sbatch"
