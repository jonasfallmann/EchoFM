#!/bin/bash

# EchoFM Attentive Probe Evaluation - Quick Start Script
# This script shows common usage patterns for the eval_attentive_probe.py script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}EchoFM Attentive Probe Evaluation - Quick Start${NC}"
echo "=================================================="
echo ""

# Configuration
CONFIG_FILE="${1:-config/echo_probe_config.yaml}"
NUM_GPUS="${2:-1}"

# Validate configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    echo ""
    echo "Usage: $0 [config_file] [num_gpus]"
    echo ""
    echo "Examples:"
    echo "  $0                                           # Single GPU, default config"
    echo "  $0 config/echo_probe_config.yaml 1          # Single GPU, custom config"
    echo "  $0 config/echo_probe_config.yaml 4          # 4 GPUs with distributed training"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Config file: $CONFIG_FILE"
echo "  Number of GPUs: $NUM_GPUS"
echo ""

# Function to run evaluation
run_evaluation() {
    local config=$1
    local num_gpus=$2

    if [ "$num_gpus" -gt 1 ]; then
        echo -e "${YELLOW}Starting distributed training with $num_gpus GPUs...${NC}"
        torchrun --nproc_per_node=$num_gpus eval_attentive_probe.py --config "$config"
    else
        echo -e "${YELLOW}Starting single-GPU training...${NC}"
        python eval_attentive_probe.py --config "$config"
    fi
}

# Function to run validation only
run_validation_only() {
    local config=$1

    echo -e "${YELLOW}Running validation only...${NC}"
    python eval_attentive_probe.py --config "$config" --val_only
}

# Function to show help
show_help() {
    cat << EOF
EchoFM Attentive Probe Evaluation Script Usage:

USAGE:
  python eval_attentive_probe.py --config CONFIG_FILE [OPTIONS]

OPTIONS:
  --config CONFIG_FILE    Path to config YAML file (required)
  --val_only             Run validation only (skip training)

EXAMPLES:

1. Single GPU training with default config:
   python eval_attentive_probe.py --config config/echo_probe_config.yaml

2. Multi-GPU distributed training (4 GPUs):
   torchrun --nproc_per_node=4 eval_attentive_probe.py --config config/echo_probe_config.yaml

3. Validation only:
   python eval_attentive_probe.py --config config/echo_probe_config.yaml --val_only

OUTPUT:
  Checkpoints and logs are saved to the folder specified in the config file.

  - latest.pt: Latest checkpoint (updated every epoch)
  - best.pt: Best checkpoint (based on validation accuracy)
  - epoch_XXX.pt: Checkpoint for specific epoch
  - log_r0.csv: Training logs with epoch, train_acc, val_acc

EOF
}

# Interactive menu
PS3="Select an option: "
options=(
    "Run full training"
    "Run validation only"
    "Show help"
    "Exit"
)

echo -e "${YELLOW}Select what you want to do:${NC}"
select opt in "${options[@]}"
do
    case $opt in
        "Run full training")
            run_evaluation "$CONFIG_FILE" "$NUM_GPUS"
            break
            ;;
        "Run validation only")
            run_validation_only "$CONFIG_FILE"
            break
            ;;
        "Show help")
            show_help
            break
            ;;
        "Exit")
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid option. Please try again."
            ;;
    esac
done

