# Parallel Training Guide

## Overview

The Snake RL project now supports parallel training using multiple worker processes. This significantly speeds up training by collecting experiences from multiple game instances simultaneously.

## Usage

### Basic Parallel Training

Run parallel training with default settings (uses CPU count - 1 workers):

```bash
python agent.py --parallel
```

### Specify Number of Workers

Run with a specific number of workers:

```bash
python agent.py --parallel --workers 4
```

### Train on Specific Level

Combine parallel training with a specific level:

```bash
python agent.py --parallel --workers 4 maze
```

### Traditional Sequential Training

The original training mode is still available:

```bash
python agent.py                    # Default sequential training
python agent.py train              # Sequential training
python agent.py train maze         # Sequential training on specific level
```

## How It Works

### Architecture

- **Main Process**: Trains the neural network model and displays the game UI
- **Worker Processes**: Run headless game instances and collect experiences
- **Shared Experience Queue**: Workers send experiences to main process
- **Model Synchronization**: Main process broadcasts updated weights to workers every 10 training steps

### Benefits

1. **Faster Training**: Multiple games run simultaneously
2. **Better Exploration**: Different workers explore different strategies
3. **Efficient CPU Usage**: Utilizes multiple CPU cores
4. **Same Model Quality**: Converges to similar or better performance

### Performance Tips

- **Optimal Workers**: Use `cpu_count() - 1` to leave one core for system tasks
- **Memory**: Each worker needs ~100MB, monitor total memory usage
- **Speed**: Expect 2-4x speedup with 4 workers depending on CPU

## Examples

```bash
# Quick test with 2 workers
python agent.py --parallel --workers 2

# Maximum performance (8 workers)
python agent.py --parallel --workers 8

# Train on obstacles level with 4 workers
python agent.py --parallel --workers 4 obstacles
```

## Testing

Test the trained model (uses best saved model):

```bash
python agent.py test              # Test on random level
python agent.py test maze         # Test on specific level
python agent.py test_level        # Test on all levels
```
