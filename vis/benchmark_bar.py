import json
import matplotlib.pyplot as plt
import argparse

def load_log_data(log_file):
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def visualize_performance(log_file):
    data = load_log_data(log_file)
    
    models = list(set(item['model_name'] for item in data))
    datasets = list(set(item['dataset'] for item in data))
    
    _, _ = plt.subplots(figsize=(12, 6))
    
    bar_width = 0.35
    opacity = 0.8
    
    for i, dataset in enumerate(datasets):
        avg_times = [next(item['average_time'] for item in data if item['model_name'] == model and item['dataset'] == dataset) for model in models]
        
        plt.bar([x + i*bar_width for x in range(len(models))], avg_times, bar_width,
                alpha=opacity, label=dataset)
    
    plt.xlabel('Models')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title('Model Performance Comparison')
    plt.xticks([x + bar_width/2 for x in range(len(models))], models)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('performance_benchmark.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str,required=True)
    args = parser.parse_args()
    visualize_performance(args.log_file)
