import json
import matplotlib.pyplot as plt

# Function to read and parse log data
def read_and_parse_log(file_path):
    with open(file_path, 'r') as f:
        log_data = f.read().split('EXP::')
    experiments = [json.loads(exp) for exp in log_data if exp]
    reward_means = [exp['reward-mean'] for exp in experiments if 'reward-mean' in exp]
    episodes = [exp['episodes'] for exp in experiments if 'episodes' in exp]
    return episodes, reward_means

# Read and parse data from three log files
episodes1, reward_means1 = read_and_parse_log(r'E:\USC Docs\CSCI567 ML\Rogue_agents_sc\rogue-gym-agents-cog19\agents\ppo_naturecnn-231126-220931-cb5567a1\log.txt')
episodes2, reward_means2 = read_and_parse_log(r'E:\USC Docs\CSCI567 ML\Rogue_agents_sc\rogue-gym-agents-cog19\agents\ppo_impalacnn-231127-200630-cb5567a1\log.txt')
episodes3, reward_means3 = read_and_parse_log(r'E:\USC Docs\CSCI567 ML\Rogue_agents_sc\rogue-gym-agents-cog19\agents\vae_ppo-231128-104753-cb5567a1\log.txt')

# Slice the lists to only include the first 100 elements
episodes1, reward_means1 = episodes1[:100], reward_means1[:100]
episodes2, reward_means2 = episodes2[:100], reward_means2[:100]
episodes3, reward_means3 = episodes3[:100], reward_means3[:100]

# Create the plot
plt.plot(episodes1, reward_means1, label='PPO_NatureCNN')
plt.plot(episodes2, reward_means2, label='PPO_ImpalaCNN (PPO Large)')
plt.plot(episodes3, reward_means3, label='VAE_PPO')
plt.xlabel('Episodes')
plt.ylabel('Reward Mean')
plt.title('Reward Mean over Episodes')
plt.legend()
plt.show()
