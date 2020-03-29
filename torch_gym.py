import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import random

env = gym.make('CartPole-v1')
goal_steps = 500
score_requirements = 50
initial_games = 1000

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        env.reset()
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirements:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1: #data 1 is the action which is 0 or 1
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]

                training_data.append([data[0], output])

        scores.append(score)

    return training_data

training_data = initial_population()

class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

model = PolicyNN()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 4)
Y = torch.Tensor([i[1] for i in training_data])
BATCH_SIZE = 300

for epoch in range(3):
    for data in range(len(X)):
        batch_X = X[data: data+BATCH_SIZE].view(-1, 4)
        batch_Y = Y[data: data+BATCH_SIZE]

        model.zero_grad()
        output = model(batch_X)
        loss = loss_function(output, batch_Y)
        loss.backward()
        optimizer.step()

scores = []

def play_games():
    for episode in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for frame in range(goal_steps):
            env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                action = torch.argmax(model(torch.Tensor(prev_obs).view(-1, 4))[0]).item()

            observation, reward, done, info = env.step(action)
            prev_obs = observation
            score += reward

            if done:
                break

        scores.append(score)

    print("Average Score", sum(scores)/len(scores))

play_games()
