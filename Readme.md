# Deep Q-Learning for Atari Kaboom 🎮🤖

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.15.0-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-ALE-green.svg)](https://gymnasium.farama.org/)

A comprehensive implementation of Deep Q-Network (DQN) for mastering Atari Kaboom, demonstrating reinforcement learning principles through systematic experimentation and analysis.

## 🎯 Project Overview

This project implements a Deep Q-Learning agent that learns to play Kaboom, a classic Atari 2600 game requiring precise timing and strategic positioning to catch falling bombs. Through trial and error, our agent achieves **214% improvement** over random baseline performance.

### Key Achievements
- 🏆 **Mean Score**: 10.85 points (vs 3.45 random baseline)
- 🚀 **Best Performance**: 32 points in a single episode
- 📈 **Learning Efficiency**: Clear breakthrough at episode 10-11
- 🔬 **Systematic Analysis**: Comprehensive hyperparameter optimization study

## 🎮 Game Environment

**Kaboom** challenges players to position water buckets to catch bombs dropped by the "Mad Bomber." The game progressively increases in difficulty with faster bomb drops and more complex patterns, making it an ideal testbed for reinforcement learning algorithms.

- **State Space**: 210×160×3 RGB pixel observations (100,800 dimensions)
- **Action Space**: 4 discrete actions (NOOP, FIRE, RIGHT, LEFT)
- **Reward Structure**: Sparse rewards (+1 per bomb caught)
- **Challenge**: Fast-paced sequential decision making with visual input

## 🧠 Deep Q-Network Architecture

Our DQN implementation features a custom CNN architecture optimized for Kaboom's visual environment:

```python
Model Architecture:
├── Conv2D(32, 8×8, stride=4) → ReLU
├── Conv2D(64, 4×4, stride=2) → ReLU  
├── Conv2D(64, 3×3, stride=1) → ReLU
├── Flatten()
├── Dense(512) → ReLU
└── Dense(4, linear) → Q-values
```

**Key Features:**
- **11.6M parameters** for complex pattern recognition
- **Experience replay** buffer (5,000 experiences)
- **Target network** updates every 250 steps
- **Epsilon-greedy exploration** with decay

## 🔬 Experimental Results

### Performance Progression
![Learning Curve](results/training_plots.png)

Our systematic experiments revealed key insights:

| Configuration | Mean Score | Performance vs Baseline |
|---------------|------------|------------------------|
| **Trained DQN** | 10.85 | +214% |
| Low Start ε | 9.45 | +172% |
| Default Config | 8.95 | +159% |
| High Learning Rate | 6.80 | +97% |
| **Random Baseline** | 3.45 | 0% |

### Hyperparameter Analysis

**Learning Rate Impact:**
- α = 0.00025: **Optimal performance** (stable learning)
- α = 0.001: **-24% performance** (training instability)

**Discount Factor Impact:**
- γ = 0.99: **Optimal performance** (long-term planning)
- γ = 0.8: **-56% performance** (short-sighted behavior)

**Exploration Strategy:**
- ε-greedy: **8.95 mean score**
- Boltzmann: **8.30 mean score** (-7.3%)

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
tensorflow >= 2.15.0
gymnasium[atari,accept-rom-license]
numpy
matplotlib
```

### Installation
```bash
# Clone repository
git clone https://github.com/vedantmane12/Atari-Kaboom-Deep-QLearning.git
cd dqn-kaboom

# Install dependencies
pip install -r requirements.txt

# Install Atari ROMs
pip install "gymnasium[atari,accept-rom-license]"
```

### Basic Usage
```python
from dqn_agent import DQNAgent
import gymnasium as gym

# Create environment
env = gym.make('ALE/Kaboom-v5')

# Initialize agent
agent = DQNAgent(
    state_shape=(210, 160, 3),
    n_actions=4,
    learning_rate=0.00025,
    gamma=0.99
)

# Train agent
for episode in range(100):
    state, _ = env.reset()
    state = state / 255.0  # Normalize
    
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience and train
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) > 32:
            agent.replay(32)
        
        state = next_state
```


## 🔬 Research Contributions

This project provides several novel contributions to reinforcement learning research:

### 1. Systematic Hyperparameter Analysis
- **Comparative study** of learning rates and discount factors
- **Quantitative analysis** of parameter sensitivity in sparse reward environments
- **Evidence-based recommendations** for DQN hyperparameter selection

### 2. Exploration Strategy Evaluation
- **Implementation and comparison** of ε-greedy vs Boltzmann exploration
- **Novel "Low Start" strategy** showing superior performance
- **Practical insights** for exploration in simple action spaces

### 3. Comprehensive Performance Metrics
- **Statistical rigor** in performance evaluation
- **Learning progression analysis** with breakthrough identification
- **Reproducible experimental methodology**

## 📊 Technical Specifications

### Model Performance
- **Training Time**: ~28 minutes on CPU
- **Memory Usage**: 5,000 experience buffer
- **Convergence**: Stable improvement after episode 10
- **Peak Performance**: 32 points (episode 19)

### Hardware Requirements
- **Minimum**: 4GB RAM, dual-core CPU
- **Recommended**: 8GB RAM, quad-core CPU
- **GPU**: Not required (CPU implementation)
- **Storage**: ~100MB for model and results

## 🤝 Contributing

We welcome contributions to improve the project! Please see our [Code of Conduct](CODE_OF_CONDUCT.md) for community guidelines.

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- 🎯 **Algorithm improvements** (Double DQN, Dueling DQN, Rainbow)
- 📊 **Additional analysis** (attention visualization, ablation studies)
- 🎮 **New environments** (extend to other Atari games)
- 📖 **Documentation** (tutorials, code comments, examples)
- 🔧 **Performance optimization** (GPU acceleration, distributed training)

## 📚 Educational Use

This project is designed for educational purposes and demonstrates:

### Core Concepts
- **Deep Reinforcement Learning** principles and implementation
- **Experience Replay** and target networks for stable training
- **Exploration vs Exploitation** trade-offs in sequential decision making
- **Hyperparameter Optimization** through systematic experimentation

### Learning Objectives
- Understand DQN algorithm implementation details
- Learn systematic experimental methodology in AI research
- Practice performance analysis and statistical evaluation
- Explore the relationship between architecture and performance

## 📖 Documentation

### Academic Resources
- 📄 **[Technical Report](docs/implementation_report.pdf)**: Comprehensive analysis and results
- 🎥 **[Video Demo](docs/video_demonstration.mp4)**: Live agent performance and explanation
- 📚 **[API Docs](docs/api_documentation.md)**: Detailed code documentation

### Key References
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
- Watkins, C. J. C. H., & Dayan, P. (1992). "Q-learning." *Machine Learning*
- Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction"

## 🐛 Issues and Support

### Known Issues
- High variance in performance (±8.68) suggests longer training could improve stability
- CPU-only implementation limits training speed for larger experiments

### Getting Help
- 📋 **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/yourusername/dqn-kaboom/issues)
- 💬 **Discussions**: Join community discussions for questions and ideas
- 📧 **Contact**: Reach out to maintainers for academic collaboration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation
If you use this code in academic work, please cite:
```bibtex
@misc{Atari-Kaboom-Deep-QLearning-2025,
  title={Deep Q-Learning for Atari Kaboom: Comprehensive Implementation Study},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/dqn-kaboom}
}
```

## 🙏 Acknowledgments

- **OpenAI Gymnasium** team for the excellent RL environment interface
- **TensorFlow** developers for the robust deep learning framework
- **Atari** for the classic game that inspired decades of AI research
- **Academic community** for foundational research in deep reinforcement learning

## 🔮 Future Work

Potential extensions and improvements:

### Algorithm Enhancements
- [ ] **Double DQN** to reduce overestimation bias
- [ ] **Dueling DQN** for better value estimation
- [ ] **Prioritized Experience Replay** for more efficient learning
- [ ] **Rainbow DQN** combining multiple improvements

### Analysis Extensions
- [ ] **Attention visualization** to understand what the agent "sees"
- [ ] **Transfer learning** to other Atari games
- [ ] **Human-AI comparison** studies
- [ ] **Curriculum learning** for improved training efficiency

### Technical Improvements
- [ ] **GPU acceleration** for faster training
- [ ] **Distributed training** for parallel experimentation
- [ ] **Hyperparameter optimization** using Bayesian methods
- [ ] **Model compression** for deployment efficiency

---

**⭐ Star this repository** if you find it helpful for learning deep reinforcement learning!

**📞 Questions?** Feel free to open an issue or start a discussion. We're here to help you learn!

---
*Built with ❤️ for the AI research and education community*