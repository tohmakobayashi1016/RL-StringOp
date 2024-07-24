# RL-StringOp
##  Investigation of string-based topology finding framework for structured surface patterns using computer algorithms

This research aims to address novel implementation of learning algorithms and generative design to string-based topology exploration methods. It seeks to achieve flexibility in the generation of high-performing diverse structural patterns of shell-like structures that effectively align multiple engineering objectives including mechanical and material efficiency and buildability. Through an investigation into the tailored application of reinforcement learning and genetic algorithms for connectivity design within structural patterns, both qualitative and quantitative metrics will be defined to demonstrate the strength and generality of this approach. Ultimately, the research aims to facilitate general creative exploration in structural design during the conceptual stages, emphasizing the importance of closer collaboration between form-designers and form-analyzers to harness the potential of emerging computation techniques.
## Running the RL Model
Python 3.9.19 & Anaconda for virtual environment 

## Windows Setup
1. Install RL-StringOp repository as .zip and extract the contents at a convenient location
2. Open Anaconda Prompt and move into the RL-StringOp directory (i.e. Tohma's directory):
```cd C:\Users\footb\Desktop\Thesis\String-RL\RL-StringOp```
Tailor this to your own set-up
4. Create the virtual environment
```conda create --name stringRL python=3.9```
5. Activate virtual environment 
```conda activate stringRL```
6. Install dependencies using the given requirements.txt
```pip install -r requirements.txt```
7. Replace addition2.py in the grammar directory of compas_quad within the virtual environment with the addition2.py placed in \Change. The directory can be found by navigating through: ```C:\Users\footb\anaconda3\envs\stringRL\Lib\site-packages\compas_quad\grammar\addition2.py```
8. Run Q-Learning_attempt04.py or the latest attempt
