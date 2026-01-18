import torch.nn as nn
import torch.nn.functional as F

class AdversaryAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(AdversaryAgent, self).__init__()
        self.args = args
        
        # 简单的 3 层 MLP
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # 输出维度：n_agents * n_actions (为每个智能体输出所有动作的 Q 值)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_agents * args.n_actions)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        # Reshape 为 [batch, n_agents, n_actions]
        return q_values.view(-1, self.args.n_agents, self.args.n_actions)