import os
import yaml
import pickle
import matplotlib.pyplot as plt

def plot_pareto_for_runs(output_dir='output', section_key="Pr_list"):
    folders = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
               if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('output_')]
    
    all_runs = []
    for folder in folders:
        # Gather meta info from parameters file
        parameters_file = os.path.join(folder, 'parameters_indep.yml')
        meta = {}
        if os.path.exists(parameters_file):
            with open(parameters_file, 'r') as f:
                params = yaml.safe_load(f)
                # meta['gamma'] = params['Optimizer']['gamma']
                # meta['nu'] = params['Optimizer']['nu']
                # meta['stopping_crit'] = params['Optimizer']['optim_1']['stopping_crit']
                # Change these keys if your YAML structure differs
                meta['gamma'] = params.get('Optimizer', {}).get('gamma') #params['Optimizer']['gamma']
                meta['nu'] = params.get('Optimizer', {}).get('nu')
                meta['var'] = params.get('Optimizer', {}).get('var')
                meta['eta'] = params.get('Optimizer', {}).get('eta')
                meta['stopping_crit'] = params.get('Optimizer', {}).get('optim_1', {}).get('stopping_crit')
                
        # Grab BayesOpt results from pickle
        results_file = os.path.join(folder, 'results.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                results_dict = pickle.load(f)
                pr_list = results_dict.get(section_key)
                iterations = len(pr_list)
                accuracy = pr_list[-1]
                if iterations is not None and accuracy is not None:
                    all_runs.append({
                        'iterations': iterations,
                        'accuracy': accuracy,
                        'gamma': meta.get('gamma'),
                        'nu': meta.get('nu'),
                        'var': meta.get('var'),
                        'eta': meta.get('eta'),
                        'stopping_crit': meta.get('stopping_crit'),
                        'label': f"γ={meta.get('gamma')}\nν={meta.get('nu')}\nη={meta.get('eta')}\nstop={meta.get('stopping_crit')}"
                        #f"γ={meta.get('gamma')}, ν={meta.get('nu')}, var = {meta.get('var')}, eta = {meta.get('eta')}, stop={meta.get('stopping_crit')}"
                    })
                else:
                    print(f"Key '{section_key}' not found in {results_file}")
    return all_runs

def plot_pareto_from_data(run_data):
    plt.figure(figsize=(10, 7))

    # Group by run label
    labels = set(run['label'] for run in run_data)
    #colors = plt.cm.tab10.colors

    for i, label in enumerate(labels):
        # Filter data for this label/run
        data = [run for run in run_data if run['label'] == label]
        x = [run['iterations'] for run in data]
        y = [run['accuracy'] for run in data]
        plt.scatter(x, y, label=label, s=80, alpha=0.7)
        
        # Annotate each point with its label
        for run in data:
            plt.annotate(
                run['label'], 
                (run['iterations'], run['accuracy']), 
                textcoords="offset points", xytext=(5,-5), ha='left', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
            )
        # # (Optional) Pareto front: keep only points that are not dominated
        # pareto_points = sorted(data, key=lambda r: (r['iterations'], -r['accuracy']))
        # pareto_x, pareto_y = [], []
        # best_y = -float('inf')
        # for pt in pareto_points:
        #     if pt['accuracy'] > best_y:
        #         pareto_x.append(pt['iterations'])
        #         pareto_y.append(pt['accuracy'])
        #         best_y = pt['accuracy']
        # plt.plot(pareto_x, pareto_y, color=colors[i % len(colors)], linewidth=2)
    
    plt.xlabel('Number of Iterations')
    plt.ylabel('End Prediction Accuracy')
    plt.title('Pareto Fronts for BayesOpt PR List Results')
    #plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
