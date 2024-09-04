# Solving ARC by pretraining on tetris trajectories
- generate a lot of tetris gameplay trajectories (multicolor)
- use these to pretrain a 2D self-attention transformer to predict next grid-state
- finetune on ARC tasks to predict output-grid given input-output training examples
