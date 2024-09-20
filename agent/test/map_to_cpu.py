import pickle


with open('/home/wenhao/RL/rl/agent/training/checkpoints/turtle_igibson2.2.2_pickle/ckpt.0.pkl', 'rb') as f:
    agent = pickle.load(f)

# 将agent移动到cpu上，并保存

agent['agent'].actor_critic = agent['agent'].actor_critic.to('cpu')
agent['agent'] = agent['agent'].to('cpu')


with open('./ckpt.0.pkl', 'wb') as f:
    pickle.dump(agent, f)
