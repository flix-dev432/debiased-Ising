import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
import sys
import time
import pickle
import traceback
from joblib import Parallel, delayed
import debiased_alg_auxiliary_eidf as db  # Our debiased algorithm module
import wandb
try:
    import torch
except ImportError:
    torch = None

# Helper function to load saved results
def load_ising_estimation_results(filepath):
    """
    Load previously saved ising estimation results from a pickle file.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved pickle file
        
    Returns:
    --------
    dict
        The comprehensive ising estimation results object
    """
    with open(filepath, 'rb') as f:
        ising_estimation_results = pickle.load(f)
    
    print(f"Loaded results from: {filepath}")
    print(f"Method: {ising_estimation_results['experiment_config']['method_name']}")
    print(f"Model: {ising_estimation_results['experiment_config']['model_description']}")
    print(f"Temperature: {ising_estimation_results['experiment_config']['temperature']}")
    print(f"Sample size: {ising_estimation_results['experiment_config']['sample_size']}")
    print(f"Timestamp: {ising_estimation_results['experiment_config']['timestamp']}")
    
    return ising_estimation_results

# Helper function to create plots from saved results
def plot_from_saved_results(ising_estimation_results, output_dir=None, save_plot=True):
    """
    Create plots from previously saved ising estimation results.
    
    Parameters:
    -----------
    ising_estimation_results : dict
        The comprehensive ising estimation results object
    output_dir : str, optional
        Directory to save the plot (if None, uses current directory)
    save_plot : bool, optional
        Whether to save the plot to file
    """
    # Extract data from the saved results
    config = ising_estimation_results['experiment_config']
    summary = ising_estimation_results['summary_statistics']
    plotting_data = ising_estimation_results['plotting_data']
    
    # Create the plot
    plt.figure(figsize=(24, 12))
    
    # Plot estimates and confidence intervals
    plt.errorbar(plotting_data['pair_indices'], plotting_data['pair_estimates'], 
                yerr=[
                    np.array(plotting_data['pair_estimates']) - np.array(plotting_data['pair_lower']),
                    np.array(plotting_data['pair_upper']) - np.array(plotting_data['pair_estimates'])
                ],
                fmt='o', capsize=4, markersize=6, elinewidth=1.5, alpha=0.8)
    
    # Plot ground truth and average estimate
    plt.axhline(y=summary['ground_truth'], color='r', linestyle='-', linewidth=3, 
               label=f'Ground truth (1/T = {summary["ground_truth"]:.6f})')
    plt.axhline(y=summary['avg_estimate'], color='g', linestyle='--', linewidth=3, 
               label=f'Average estimate ({summary["avg_estimate"]:.6f})')
    
    # Customize plot
    plt.xlabel('Adjacent Pair Index', fontsize=20, fontweight='bold')
    plt.ylabel('Estimate (adjusted by dividing by 4)', fontsize=20, fontweight='bold')
    
    title = f'{config["method_name"]} Estimates for 2D Ising Model\nT={config["temperature"]}, N={config["sample_size"]}, {config["model_description"]}'
    plt.title(title, fontsize=22, fontweight='bold', pad=20)
    
    plt.xticks(np.arange(0, len(plotting_data['pair_indices']), 10), fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16, loc='upper right')
    
    # Add statistics box
    stats_text = f'Avg SE: {summary["avg_se"]:.6f}\nBias: {summary["bias"]:.6f}\nCoverage: {summary["coverage_rate"]:.1%}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=16, verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_plot:
        if output_dir is None:
            output_dir = '.'
        filename = f"{output_dir}/replotted_ising_results_{config['method']}_{config.get('model_type_theta', 'unknown')}_{config.get('model_type_alpha', 'unknown')}_T{config['temperature']}_{config['sample_size']}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {filename}")
    
    plt.show()
    return plt.gcf()

# Function to identify adjacent pairs in the lattice
def nn_pairs(data_shape):
    ising_size = data_shape[1]
    L_size = int(np.sqrt(data_shape[1]))
    sites_2D = np.asarray(range(data_shape[1])).reshape(L_size, L_size)
    print(f"Lattice sites arranged in {L_size}x{L_size} grid:")
    print(sites_2D)
    
    nn_horizontal = []
    nn_vertical = []
 
    # Horizontal neighbors (with periodic boundary)
    for i in range(0, ising_size):  
        j = ((i+1) % L_size) + int(i/L_size)*L_size
        if i < j:
            nn_horizontal.append([i, j])
        else:
            nn_horizontal.append([j, i])

    # Vertical neighbors (with periodic boundary)
    for i in range(0, ising_size):  
        j = (i + L_size) % ising_size
        if i < j:
            nn_vertical.append([i, j])
        else:
            nn_vertical.append([j, i])
            
    nn_all_pairs = np.asarray(nn_horizontal + nn_vertical)
    return nn_all_pairs

# Function to identify non-adjacent pairs and randomly select the same number as adjacent pairs
def non_adjacent_pairs(data_shape, random_seed=42):
    """
    Generate all non-adjacent pairs and randomly select the same number as adjacent pairs.
    
    Parameters:
    -----------
    data_shape : tuple
        Shape of the data (n_samples, n_sites)
    random_seed : int
        Random seed for reproducible selection
        
    Returns:
    --------
    numpy.ndarray
        Array of randomly selected non-adjacent pairs
    """
    ising_size = data_shape[1]
    L_size = int(np.sqrt(data_shape[1]))
    
    # Get adjacent pairs to know what to exclude
    adjacent_pairs = nn_pairs(data_shape)
    adjacent_set = set(tuple(sorted([p[0], p[1]])) for p in adjacent_pairs)
    
    # Generate all possible pairs
    all_pairs = []
    for i in range(ising_size):
        for j in range(i + 1, ising_size):
            pair_tuple = tuple(sorted([i, j]))
            if pair_tuple not in adjacent_set:
                all_pairs.append([i, j])
    
    print(f"Total number of non-adjacent pairs: {len(all_pairs)}")
    print(f"Number of adjacent pairs: {len(adjacent_pairs)}")
    
    # Randomly select the same number as adjacent pairs
    np.random.seed(random_seed)
    selected_indices = np.random.choice(len(all_pairs), size=len(adjacent_pairs), replace=False)
    selected_non_adjacent = np.asarray([all_pairs[i] for i in selected_indices])
    
    print(f"Randomly selected {len(selected_non_adjacent)} non-adjacent pairs (seed={random_seed})")
    
    return selected_non_adjacent

# Function to process a single pair (adjacent or non-adjacent)
def process_pair(pair_idx, pair, data, pair_number, method='autoTML', model_type_theta='constant', model_type_alpha='linear',
                temperature=None, sample_size=None, pair_type='adjacent', total_pairs=128,
                n_splits=5, lambda_theta=0.001, lambda_alpha=0.01, l1_ratio=0.5,
                theta_epochs=150, alpha_epochs=400, 
                batch_size_theta=512, batch_size_alpha=1024,
                lr_theta=0.01, lr_alpha=0.01, alpha_nn_setup=None):
    y_index, a_index = pair[0], pair[1]
    
    try:
        print(f"Processing {pair_type} pair {pair_number}/{total_pairs}: ({y_index}, {a_index})")
        sys.stdout.flush()
        
        # Set a unique seed for each pair
        seed = 42 + pair_idx
        
        # Create unique run ID and pair info for wandb
        run_id = f"T{temperature}_{sample_size}_{method}_{model_type_theta}_{model_type_alpha}_{pair_type}_pair_{y_index}_{a_index}"
        pair_info = {
            'y': y_index,
            'a': a_index,
            'pair_number': pair_number,
            'pair_type': pair_type,
            'temperature': temperature,
            'sample_size': sample_size
        }
        
        print(f"About to call debiased_binary_estimator for pair ({y_index}, {a_index}) with method {method}")
        sys.stdout.flush()
        
        # Run the debiased binary estimator with specified method and model types
        # Using proper GPU context handling for multi-worker scenarios
        result = db.debiased_binary_estimator(
            data, 
            y_index=y_index, 
            a_index=a_index, 
            method=method,  # Use parameter from main function
            model_type_theta=model_type_theta,  # Use parameter from main function
            model_type_alpha=model_type_alpha,  # Use parameter from main function
            n_splits=n_splits, 
            lambda_theta=lambda_theta, 
            lambda_alpha=lambda_alpha, 
            l1_ratio=l1_ratio,  # Pass l1_ratio parameter for polynomial alpha models
            theta_epochs=theta_epochs,  
            alpha_epochs=alpha_epochs, 
            batch_size_theta=batch_size_theta,    
            batch_size_alpha=batch_size_alpha,     
            lr_theta=lr_theta, 
            lr_alpha=lr_alpha, 
            seed=seed, 
            alpha_nn_setup=alpha_nn_setup,  # Pass alpha neural network architecture
            run_id=run_id,
            pair_info=pair_info
        )
        
        print(f"debiased_binary_estimator completed for pair ({y_index}, {a_index})")
        sys.stdout.flush()
        
        # Adjust the estimates by dividing by 4 (to match ground truth 1/T)
        adjusted_result = {
            'pair': pair,
            'pair_idx': pair_idx,
            'pair_type': pair_type,
            'psi_hat': result['psi_hat'] / 4.0,
            'se_hat': result['se_hat'] / 4.0,
            'ci_lower': result['ci_lower'] / 4.0,
            'ci_upper': result['ci_upper'] / 4.0,
            'epsilon_n': result.get('epsilon_n', None),
            'avg_eif': result.get('avg_eif', None),
            'avg_eif_term': result.get('avg_eif_term', None),
            'execution_time': result.get('execution_time', None),
            'theta_training_time': result.get('theta_training_time', None),
            'alpha_training_time': result.get('alpha_training_time', None),
            'unadjusted_psi_hat': result['psi_hat'],  # Keep original for reference
            'theta_learning_curves': result.get('theta_learning_curves', []),
            'alpha_learning_curves': result.get('alpha_learning_curves', []),
            'alpha_validation_losses': result.get('alpha_validation_losses', []),
            'mean_alpha_validation_loss': result.get('mean_alpha_validation_loss', None),
            'y_index': y_index,
            'a_index': a_index
        }
        
        return adjusted_result
    
    except Exception as e:
        print(f"Error processing {pair_type} pair {pair}: {e}")
        return {
            'pair': pair,
            'pair_idx': pair_idx,
            'pair_type': pair_type,
            'error': str(e)
        }

# Function to process a single pair with static GPU assignment
def worker_with_gpu(i, pair, data, pair_number, method, model_type_theta, model_type_alpha,
                    temperature, sample_size, pair_type, total_pairs,
                    n_splits, lambda_theta, lambda_alpha, l1_ratio,
                    theta_epochs, alpha_epochs,
                    batch_size_theta, batch_size_alpha,
                    lr_theta, lr_alpha, alpha_nn_setup, max_workers=10):
    """
    Worker function with static GPU assignment.
    
    GPU assignment strategy:
    - Distributes workers evenly across available GPUs
    - For max_workers=10 and 2 GPUs: workers 0-4 → GPU 0, workers 5-9 → GPU 1
    - Falls back to CPU if no GPUs available
    """
    try:
        print(f"Worker {i}: Starting processing for pair {pair}")
        sys.stdout.flush()
        
        # Get number of available GPUs
        if torch is not None and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 0
        
        # Static GPU assignment if GPUs are available
        if num_gpus > 0:
            # Calculate GPU assignment based on worker index
            # Distribute workers evenly across GPUs
            # Formula: gpu_id = worker_index // workers_per_gpu
            workers_per_gpu = max_workers // num_gpus
            remaining_workers = max_workers % num_gpus
            
            if i < (num_gpus - remaining_workers) * workers_per_gpu:
                # Workers that fit evenly
                gpu_id = i // workers_per_gpu
            else:
                # Handle remaining workers (give extra workers to the last few GPUs)
                adjusted_i = i - (num_gpus - remaining_workers) * workers_per_gpu
                gpu_id = (num_gpus - remaining_workers) + adjusted_i // (workers_per_gpu + 1)
            
            # Ensure gpu_id is within valid range
            gpu_id = min(gpu_id, num_gpus - 1)
            
            print(f"Worker {i}: GPU assignment - {max_workers} workers across {num_gpus} GPUs, assigned to GPU {gpu_id}")
            sys.stdout.flush()
            
            try:
                torch.cuda.set_device(gpu_id)
                # Initialize CUDA context to ensure it's properly set up
                torch.cuda.init()
                # Create a small tensor to ensure context is active
                _ = torch.tensor([1.0], device=f'cuda:{gpu_id}')
                print(f"Worker {i}: assigned to GPU {gpu_id} (out of {num_gpus} GPUs)")
                sys.stdout.flush()
            except Exception as e:
                print(f"Worker {i}: Failed to set GPU {gpu_id}, falling back to CPU. Error: {e}")
                sys.stdout.flush()
        else:
            print(f"Worker {i}: No GPUs available, using CPU")
            sys.stdout.flush()
            
    except Exception as e:
        print(f"Worker {i}: GPU setup failed, using CPU. Error: {e}")
        sys.stdout.flush()
    
    # Run the actual pair processing
    try:
        print(f"Worker {i}: About to call process_pair for {pair}")
        sys.stdout.flush()
        
        result = process_pair(i, pair, data, pair_number, method, model_type_theta, model_type_alpha,
                            temperature, sample_size, pair_type, total_pairs,
                            n_splits, lambda_theta, lambda_alpha, l1_ratio,
                            theta_epochs, alpha_epochs,
                            batch_size_theta, batch_size_alpha,
                            lr_theta, lr_alpha, alpha_nn_setup)
        
        print(f"Worker {i}: Completed processing for pair {pair}")
        sys.stdout.flush()
        return result
        
    except Exception as e:
        print(f"Worker {i}: Error in process_pair. Error: {e}")
        print(f"Worker {i}: Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        return {
            'pair': pair,
            'pair_idx': i,
            'pair_type': pair_type,
            'error': str(e)
        }

# Function to plot individual learning curves for each pair using wandb
def plot_individual_learning_curves_wandb(results, method, model_type_theta, model_type_alpha):
    """
    Plot individual learning curves for each pair using Weights & Biases
    Creates separate plots for each pair with all 5 folds clearly labeled
    Organizes plots by pair type (adjacent vs non-adjacent) into separate folders
    
    Parameters:
    -----------
    results : list
        List of results from each pair processing
    method : str
        Method used (autoDML or autoTML)
    model_type_theta : str
        Theta model type used
    model_type_alpha : str
        Alpha model type used
    """
    try:
        import matplotlib.pyplot as plt
        
        # Filter out errored results and collect learning curves
        valid_results = [r for r in results if 'error' not in r and 'theta_learning_curves' in r]
        
        if not valid_results:
            print("No valid learning curves to plot")
            return
        
        # Separate results by pair type
        adjacent_results = [r for r in valid_results if r.get('pair_type') == 'adjacent']
        non_adjacent_results = [r for r in valid_results if r.get('pair_type') == 'non_adjacent']
        
        print(f"Creating learning curve plots:")
        print(f"  - Adjacent pairs: {len(adjacent_results)}")
        print(f"  - Non-adjacent pairs: {len(non_adjacent_results)}")
        
        # Create individual plots for each pair
        plots_created = 0
        batch_size = 10  # Log plots in batches to improve efficiency
        batch_logs = {}
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Distinct colors for 5 folds
        
        # Process adjacent pairs first
        for result in adjacent_results:
            pair_y = result['y_index']
            pair_a = result['a_index']
            theta_curves = result['theta_learning_curves']
            alpha_curves = result['alpha_learning_curves']
            
            # Skip if no learning curves available
            if not theta_curves or not alpha_curves:
                continue
            
            # Create theta learning curve plot for this adjacent pair
            fig_theta, ax_theta = plt.subplots(figsize=(10, 6))
            
            for fold_idx, losses in enumerate(theta_curves):
                if losses:  # Check if losses is not empty
                    epochs = range(1, len(losses) + 1)
                    color = colors[fold_idx % len(colors)]
                    ax_theta.plot(epochs, losses, 
                                label=f'Fold {fold_idx + 1}', 
                                color=color,
                                linewidth=2.0,
                                alpha=0.8)
            
            ax_theta.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax_theta.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
            ax_theta.set_title(f'Theta Training - {method} - Adjacent Pair ({pair_y}, {pair_a}) - Theta:{model_type_theta}, Alpha:{model_type_alpha}', 
                              fontsize=14, fontweight='bold')
            ax_theta.legend(fontsize=10, loc='best')
            ax_theta.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create alpha learning curve plot for this adjacent pair
            fig_alpha, ax_alpha = plt.subplots(figsize=(10, 6))
            
            for fold_idx, losses in enumerate(alpha_curves):
                if losses:  # Check if losses is not empty
                    epochs = range(1, len(losses) + 1)
                    color = colors[fold_idx % len(colors)]
                    ax_alpha.plot(epochs, losses, 
                                label=f'Fold {fold_idx + 1}', 
                                color=color,
                                linewidth=2.0,
                                alpha=0.8)
            
            ax_alpha.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax_alpha.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
            ax_alpha.set_title(f'Alpha Training - {method} - Adjacent Pair ({pair_y}, {pair_a}) - Theta:{model_type_theta}, Alpha:{model_type_alpha}', 
                              fontsize=14, fontweight='bold')
            ax_alpha.legend(fontsize=10, loc='best')
            ax_alpha.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add to batch for logging with separate folders
            batch_logs[f"theta_adjacent_learning_curves/pair_{pair_y}_{pair_a}"] = wandb.Image(fig_theta)
            batch_logs[f"alpha_adjacent_learning_curves/pair_{pair_y}_{pair_a}"] = wandb.Image(fig_alpha)
            
            # Close figures to save memory
            plt.close(fig_theta)
            plt.close(fig_alpha)
            
            plots_created += 1
            
            # Log batch when it reaches batch_size
            if len(batch_logs) >= batch_size * 2:
                wandb.log(batch_logs)
                batch_logs = {}  # Clear the batch
                print(f"  📊 Logged adjacent learning curves for pairs up to #{plots_created}")
        
        # Process non-adjacent pairs
        for result in non_adjacent_results:
            pair_y = result['y_index']
            pair_a = result['a_index']
            theta_curves = result['theta_learning_curves']
            alpha_curves = result['alpha_learning_curves']
            
            # Skip if no learning curves available
            if not theta_curves or not alpha_curves:
                continue
            
            # Create theta learning curve plot for this non-adjacent pair
            fig_theta, ax_theta = plt.subplots(figsize=(10, 6))
            
            for fold_idx, losses in enumerate(theta_curves):
                if losses:  # Check if losses is not empty
                    epochs = range(1, len(losses) + 1)
                    color = colors[fold_idx % len(colors)]
                    ax_theta.plot(epochs, losses, 
                                label=f'Fold {fold_idx + 1}', 
                                color=color,
                                linewidth=2.0,
                                alpha=0.8)
            
            ax_theta.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax_theta.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
            ax_theta.set_title(f'Theta Training - {method} - Non-Adjacent Pair ({pair_y}, {pair_a}) - Theta:{model_type_theta}, Alpha:{model_type_alpha}', 
                              fontsize=14, fontweight='bold')
            ax_theta.legend(fontsize=10, loc='best')
            ax_theta.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Create alpha learning curve plot for this non-adjacent pair
            fig_alpha, ax_alpha = plt.subplots(figsize=(10, 6))
            
            for fold_idx, losses in enumerate(alpha_curves):
                if losses:  # Check if losses is not empty
                    epochs = range(1, len(losses) + 1)
                    color = colors[fold_idx % len(colors)]
                    ax_alpha.plot(epochs, losses, 
                                label=f'Fold {fold_idx + 1}', 
                                color=color,
                                linewidth=2.0,
                                alpha=0.8)
            
            ax_alpha.set_xlabel('Epochs', fontsize=12, fontweight='bold')
            ax_alpha.set_ylabel('Average Loss', fontsize=12, fontweight='bold')
            ax_alpha.set_title(f'Alpha Training - {method} - Non-Adjacent Pair ({pair_y}, {pair_a}) - Theta:{model_type_theta}, Alpha:{model_type_alpha}', 
                              fontsize=14, fontweight='bold')
            ax_alpha.legend(fontsize=10, loc='best')
            ax_alpha.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add to batch for logging with separate folders
            batch_logs[f"theta_non_adjacent_learning_curves/pair_{pair_y}_{pair_a}"] = wandb.Image(fig_theta)
            batch_logs[f"alpha_non_adjacent_learning_curves/pair_{pair_y}_{pair_a}"] = wandb.Image(fig_alpha)
            
            # Close figures to save memory
            plt.close(fig_theta)
            plt.close(fig_alpha)
            
            plots_created += 1
            
            # Log batch when it reaches batch_size
            if len(batch_logs) >= batch_size * 2:
                wandb.log(batch_logs)
                batch_logs = {}  # Clear the batch
                print(f"  📊 Logged non-adjacent learning curves for pairs up to #{plots_created}")
        
        # Log any remaining plots in the last batch
        if batch_logs:
            wandb.log(batch_logs)
            print(f"  📊 Logged final batch of learning curve plots")
        
        # Log summary statistics
        wandb.log({
            "learning_curves_summary/total_pairs_plotted": plots_created,
            "learning_curves_summary/total_valid_pairs": len(valid_results),
            "learning_curves_summary/adjacent_pairs_plotted": len(adjacent_results),
            "learning_curves_summary/non_adjacent_pairs_plotted": len(non_adjacent_results),
            "learning_curves_summary/total_theta_plots": plots_created,
            "learning_curves_summary/total_alpha_plots": plots_created
        })
        
        print(f"✅ Successfully created {plots_created * 2} learning curve plots:")
        print(f"   - Adjacent: {len(adjacent_results)} theta + {len(adjacent_results)} alpha plots")
        print(f"   - Non-adjacent: {len(non_adjacent_results)} theta + {len(non_adjacent_results)} alpha plots")
        print(f"📊 Check wandb for organized plots:")
        print(f"   - 'theta_adjacent_learning_curves/' ({len(adjacent_results)} plots)")
        print(f"   - 'alpha_adjacent_learning_curves/' ({len(adjacent_results)} plots)")
        print(f"   - 'theta_non_adjacent_learning_curves/' ({len(non_adjacent_results)} plots)")
        print(f"   - 'alpha_non_adjacent_learning_curves/' ({len(non_adjacent_results)} plots)")
        
    except Exception as e:
        print(f"Warning: Could not create wandb learning curve plots: {e}")
        import traceback
        traceback.print_exc()

# Main function to run Ising model analysis
def analyze_ising_model(data_file, output_dir='ising_results', max_workers=4, method='autoTML', 
                       model_type_theta='constant', model_type_alpha='linear', 
                       pair_selection='adjacent', non_adjacent_seed=42,
                       n_splits=5, lambda_theta=0.001, lambda_alpha=0.01, l1_ratio=0.5,
                       theta_epochs=150, alpha_epochs=400, 
                       batch_size_theta=512, batch_size_alpha=1024,  
                       lr_theta=0.01, lr_alpha=0.01, alpha_nn_setup=None):
    print(f"Loading data from {data_file}...")
    
    # Load the data
    csvdata = np.loadtxt(data_file, delimiter=",", skiprows=1)
    data = csvdata.reshape(-1, 64)
    
    print(f"Data loaded: {data.shape[0]} samples, {data.shape[1]} positions")
    
    # System resource detection
    print("\n" + "="*50)
    print("SYSTEM RESOURCE DETECTION")
    print("="*50)
    
    # Detect CPUs
    try:
        num_cpus = os.cpu_count()
        print(f"CPUs detected: {num_cpus}")
    except Exception as e:
        print(f"CPUs detected: Unable to determine ({e})")
    
    # Detect GPUs
    try:
        if torch is not None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print(f"GPUs detected: {num_gpus}")
                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                    print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                print("GPUs detected: 0 (CUDA not available)")
        else:
            print("GPUs detected: 0 (PyTorch not available)")
    except Exception as e:
        print(f"GPUs detected: Unable to determine ({e})")
    
    print(f"Parallel workers configured: {max_workers}")
    print("="*50 + "\n")
    
    # Extract method, temperature, and sample size information for flexible use
    method_name = method.upper()  # Use the actual method parameter
    
    # Extract temperature from data file name
    if "T1.8" in data_file:
        temperature = 1.8
    elif "T2.3" in data_file:
        temperature = 2.3
    elif "T3.0" in data_file:
        temperature = 3.0
    else:
        raise ValueError(f"Cannot extract temperature from data file name: {data_file}. Expected 'T1.8', 'T2.3', or 'T3.0' in filename.")
    
    # Extract sample size from data file name
    if "10k" in data_file:
        sample_size = "10k"
    elif "100k" in data_file:
        sample_size = "100k"
    elif "1M" in data_file:
        sample_size = "1M"
    else:
        sample_size = f"{data.shape[0]}samples"
    
    # Initialize wandb for this run
    wandb_run_name = f"{method_name}_{model_type_theta}_{model_type_alpha}_{pair_selection}_T{temperature}_{sample_size}"
    if pair_selection in ['non_adjacent', 'both']:
        wandb_run_name += f"_seed{non_adjacent_seed}"
    print(f"Initializing Weights & Biases run: {wandb_run_name}")
    
    wandb.init(
        project="2d-ising-autodml",
        name=wandb_run_name,
        config={
            "method": method,
            "model_type_theta": model_type_theta,
            "model_type_alpha": model_type_alpha,
            "temperature": temperature,
            "sample_size": sample_size,
            "data_file": data_file,
            "max_workers": max_workers,
            "pair_selection": pair_selection,
            "non_adjacent_seed": non_adjacent_seed if pair_selection in ['non_adjacent', 'both'] else None,
            "n_samples": data.shape[0],
            "n_positions": data.shape[1]
        },
        tags=[method, f"theta_{model_type_theta}", f"alpha_{model_type_alpha}", f"T{temperature}", sample_size, pair_selection]
    )
    
    # Create model description based on model types
    theta_desc = {
        'constant': 'Constant Theta',
        'linear': 'Linear Theta',
        'nn': 'NN Theta',
        'X_only': 'X-only Theta'
    }.get(model_type_theta, f"{model_type_theta.capitalize()} Theta")
    
    alpha_desc = {
        'linear': 'Linear Alpha',
        'nn': 'NN Alpha',
        'polynomial': 'Polynomial Alpha'
    }.get(model_type_alpha, f"{model_type_alpha.capitalize()} Alpha")
    
    model_description = f"{theta_desc}, {alpha_desc}"
    
    # Calculate ground truth based on pair type(s)
    # For non-adjacent pairs, ground truth is 0; for adjacent pairs, it's 1/temperature
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which pairs to process based on pair_selection
    print(f"Pair selection mode: {pair_selection}")
    
    if pair_selection == 'adjacent':
        # Process only adjacent pairs
        pairs_to_process = nn_pairs(data.shape)
        pair_types = ['adjacent'] * len(pairs_to_process)
        ground_truths = [1/temperature] * len(pairs_to_process)
        print(f"Processing {len(pairs_to_process)} adjacent pairs")
        
    elif pair_selection == 'non_adjacent':
        # Process only randomly selected non-adjacent pairs
        pairs_to_process = non_adjacent_pairs(data.shape, random_seed=non_adjacent_seed)
        pair_types = ['non_adjacent'] * len(pairs_to_process)
        ground_truths = [0.0] * len(pairs_to_process)
        print(f"Processing {len(pairs_to_process)} non-adjacent pairs (seed={non_adjacent_seed})")
        
    elif pair_selection == 'both':
        # Process both adjacent and non-adjacent pairs
        adjacent_pairs = nn_pairs(data.shape)
        non_adj_pairs = non_adjacent_pairs(data.shape, random_seed=non_adjacent_seed)
        
        pairs_to_process = np.vstack([adjacent_pairs, non_adj_pairs])
        pair_types = ['adjacent'] * len(adjacent_pairs) + ['non_adjacent'] * len(non_adj_pairs)
        ground_truths = [1/temperature] * len(adjacent_pairs) + [0.0] * len(non_adj_pairs)
        
        print(f"Processing {len(adjacent_pairs)} adjacent pairs + {len(non_adj_pairs)} non-adjacent pairs")
        print(f"Total pairs to process: {len(pairs_to_process)}")
        
    else:
        raise ValueError(f"Invalid pair_selection: {pair_selection}. Must be 'adjacent', 'non_adjacent', or 'both'")
    
    # Process pairs using joblib's Parallel approach
    start_time = time.time()
    
    # Check GPU availability for information (static assignment will be done in workers)
    try:
        use_gpu = torch is not None and torch.cuda.is_available()
        num_gpus = torch.cuda.device_count() if (torch is not None and use_gpu) else 0
    except:
        use_gpu = False
        num_gpus = 0
    
    if num_gpus > 0:
        print(f"GPU setup: {num_gpus} GPUs detected. Using static assignment.")
        
        # Calculate and display worker distribution across GPUs
        workers_per_gpu = max_workers // num_gpus
        remaining_workers = max_workers % num_gpus
        
        print(f"Worker distribution: {workers_per_gpu} workers per GPU base")
        if remaining_workers > 0:
            print(f"Last {remaining_workers} GPU(s) get 1 extra worker")
        
        # Show detailed assignment
        for gpu_id in range(num_gpus):
            if gpu_id < (num_gpus - remaining_workers):
                start_worker = gpu_id * workers_per_gpu
                end_worker = start_worker + workers_per_gpu - 1
                worker_count = workers_per_gpu
            else:
                # GPUs that get extra workers
                base_workers = (num_gpus - remaining_workers) * workers_per_gpu
                extra_gpu_idx = gpu_id - (num_gpus - remaining_workers)
                start_worker = base_workers + extra_gpu_idx * (workers_per_gpu + 1)
                end_worker = start_worker + workers_per_gpu  # +1 extra worker
                worker_count = workers_per_gpu + 1
            
            print(f"GPU {gpu_id}: Workers {start_worker}-{end_worker} ({worker_count} workers)")
    else:
        print("GPU setup: No GPUs available. All workers will use CPU.")

    # Using joblib's Parallel with static GPU assignment
    print("Starting parallel processing...")
    print(f"About to process {len(pairs_to_process)} pairs with {max_workers} workers")
    sys.stdout.flush()  # Force output immediately
    
    results = Parallel(n_jobs=max_workers, verbose=10)(
        delayed(worker_with_gpu)(i, pairs_to_process[i], data, i+1, method, model_type_theta, model_type_alpha, 
                            temperature, sample_size, pair_types[i], len(pairs_to_process),
                            n_splits, lambda_theta, lambda_alpha, l1_ratio,
                            theta_epochs, alpha_epochs, 
                            batch_size_theta, batch_size_alpha,
                            lr_theta, lr_alpha, alpha_nn_setup, max_workers) 
        for i in range(len(pairs_to_process))
    )
    
    print(f"Parallel processing completed! Got {len(results)} results")
    sys.stdout.flush()
            
    # Print results as they complete
    for result in results:
        if 'error' not in result:
            pair = result['pair']
            print(f"Completed pair ({pair[0]}, {pair[1]}): "
                  f"estimate={result['psi_hat']:.6f}, "
                  f"se={result['se_hat']:.6f}, "
                  f"95% CI=[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Sort results by pair index
    results = sorted(results, key=lambda x: x.get('pair_idx', 999))
    
    # Save results to file
    with open(f"{output_dir}/ising_results.txt", 'w') as f:
        f.write("2D Ising Model Analysis Results\n")
        f.write("=============================\n\n")
        f.write(f"Data file: {data_file}\n")
        f.write(f"Temperature T = {temperature}\n")
        f.write(f"Data shape: {data.shape[0]} samples, {data.shape[1]} positions\n")
        f.write(f"Pair selection: {pair_selection}\n")
        
        if pair_selection == 'adjacent':
            f.write(f"Number of adjacent pairs: {len(pairs_to_process)}\n")
            f.write(f"Ground truth for adjacent pairs (1/T): {1/temperature:.6f}\n")
        elif pair_selection == 'non_adjacent':
            f.write(f"Number of non-adjacent pairs: {len(pairs_to_process)}\n")
            f.write(f"Non-adjacent random seed: {non_adjacent_seed}\n")
            f.write(f"Ground truth for non-adjacent pairs: 0.000000\n")
        elif pair_selection == 'both':
            n_adj = sum(1 for pt in pair_types if pt == 'adjacent')
            n_non_adj = sum(1 for pt in pair_types if pt == 'non_adjacent')
            f.write(f"Number of adjacent pairs: {n_adj}\n")
            f.write(f"Number of non-adjacent pairs: {n_non_adj}\n")
            f.write(f"Non-adjacent random seed: {non_adjacent_seed}\n")
            f.write(f"Ground truth for adjacent pairs (1/T): {1/temperature:.6f}\n")
            f.write(f"Ground truth for non-adjacent pairs: 0.000000\n")
            
        f.write(f"Method: {method_name} with {model_description}\n\n")
        
        f.write("Results (adjusted by dividing by 4):\n")
        f.write("----------------------------------\n")
        
        # Calculate average estimates by pair type
        valid_results = [r for r in results if 'error' not in r]
        
        if pair_selection == 'both':
            # Separate statistics for adjacent and non-adjacent pairs
            adj_results = [r for r in valid_results if r.get('pair_type') == 'adjacent']
            non_adj_results = [r for r in valid_results if r.get('pair_type') == 'non_adjacent']
            
            if adj_results:
                adj_avg_estimate = np.mean([r['psi_hat'] for r in adj_results])
                adj_avg_se = np.mean([r['se_hat'] for r in adj_results])
                adj_ground_truth = 1/temperature
                adj_bias = adj_avg_estimate - adj_ground_truth
                adj_coverage_count = sum(1 for r in adj_results if r['ci_lower'] <= adj_ground_truth <= r['ci_upper'])
                adj_coverage_rate = adj_coverage_count / len(adj_results)
                
                f.write(f"ADJACENT PAIRS ({len(adj_results)} pairs):\n")
                f.write(f"  Average estimate: {adj_avg_estimate:.6f} (SE: {adj_avg_se:.6f})\n")
                f.write(f"  Ground truth (1/T): {adj_ground_truth:.6f}\n")
                f.write(f"  Bias from ground truth: {adj_bias:.6f}\n")
                f.write(f"  Coverage rate: {adj_coverage_rate:.2%} ({adj_coverage_count}/{len(adj_results)} CIs contain ground truth)\n\n")
            
            if non_adj_results:
                non_adj_avg_estimate = np.mean([r['psi_hat'] for r in non_adj_results])
                non_adj_avg_se = np.mean([r['se_hat'] for r in non_adj_results])
                non_adj_ground_truth = 0.0
                non_adj_bias = non_adj_avg_estimate - non_adj_ground_truth
                non_adj_coverage_count = sum(1 for r in non_adj_results if r['ci_lower'] <= non_adj_ground_truth <= r['ci_upper'])
                non_adj_coverage_rate = non_adj_coverage_count / len(non_adj_results)
                
                f.write(f"NON-ADJACENT PAIRS ({len(non_adj_results)} pairs):\n")
                f.write(f"  Average estimate: {non_adj_avg_estimate:.6f} (SE: {non_adj_avg_se:.6f})\n")
                f.write(f"  Ground truth: {non_adj_ground_truth:.6f}\n")
                f.write(f"  Bias from ground truth: {non_adj_bias:.6f}\n")
                f.write(f"  Coverage rate: {non_adj_coverage_rate:.2%} ({non_adj_coverage_count}/{len(non_adj_results)} CIs contain ground truth)\n\n")
        else:
            # Single type of pairs
            avg_estimate = np.mean([r['psi_hat'] for r in valid_results])
            avg_se = np.mean([r['se_hat'] for r in valid_results])
            ground_truth = ground_truths[0]  # All pairs have same ground truth
            bias = avg_estimate - ground_truth
            coverage_count = sum(1 for r in valid_results if r['ci_lower'] <= ground_truth <= r['ci_upper'])
            coverage_rate = coverage_count / len(valid_results)
            
            f.write(f"Average estimate across all pairs: {avg_estimate:.6f} (SE: {avg_se:.6f})\n")
            f.write(f"Ground truth: {ground_truth:.6f}\n")
            f.write(f"Bias from ground truth: {bias:.6f}\n")
            f.write(f"Coverage rate: {coverage_rate:.2%} ({coverage_count}/{len(valid_results)} CIs contain ground truth)\n\n")
        
        f.write("Individual pair results:\n")
        
        # Group results by pair type for better organization
        if pair_selection == 'both':
            # Write adjacent pairs first
            f.write("\n--- ADJACENT PAIRS ---\n")
            for i, result in enumerate(results):
                if 'error' in result or result.get('pair_type') != 'adjacent':
                    continue
                    
                pair = result['pair']
                pair_type = result.get('pair_type', 'unknown')
                pair_ground_truth = ground_truths[i] if i < len(ground_truths) else 0.0
                
                # Calculate additional statistics
                pair_bias = result['psi_hat'] - pair_ground_truth
                bias_se_ratio = pair_bias / result['se_hat'] if result['se_hat'] != 0 else float('inf')
                ci_width = result['ci_upper'] - result['ci_lower']
                
                f.write(f"Pair ({pair[0]}, {pair[1]}) [{pair_type}]: ")
                f.write(f"adjusted_estimate={result['psi_hat']:.6f} (original={result['unadjusted_psi_hat']:.6f}), ")
                f.write(f"se={result['se_hat']:.6f}, ")
                f.write(f"bias={pair_bias:.6f}, ")
                f.write(f"bias/SE={bias_se_ratio:.6f}, ")
                f.write(f"CI_width={ci_width:.6f}, ")
                f.write(f"95% CI=[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}], ")
                
                if result['epsilon_n'] is not None:
                    f.write(f"epsilon_n={result['epsilon_n']:.6f}, ")
                
                # Add autoTML-specific outputs
                if method.lower() == 'autotml':
                    if result.get('avg_eif_term') is not None:
                        f.write(f"avg_eif_term={result['avg_eif_term']:.6f}, ")
                    if result.get('avg_eif') is not None:
                        f.write(f"avg_eif_value={result['avg_eif']:.6f}, ")
                
                # Add validation loss information
                if result.get('mean_alpha_validation_loss') is not None:
                    f.write(f"mean_alpha_val_loss={result['mean_alpha_validation_loss']:.6f}, ")
                if result.get('alpha_validation_losses'):
                    val_losses_str = "[" + ", ".join([f"{loss:.6f}" for loss in result['alpha_validation_losses']]) + "]"
                    f.write(f"alpha_val_losses={val_losses_str}, ")
                
                contains_truth = result['ci_lower'] <= pair_ground_truth <= result['ci_upper']
                f.write(f"Contains truth: {contains_truth}, ")
                f.write(f"execution_time={result['execution_time']:.2f}s, ")
                f.write(f"theta_training_time={result.get('theta_training_time', 0):.2f}s, ")
                f.write(f"alpha_training_time={result.get('alpha_training_time', 0):.2f}s\n")
            
            # Add spacing and write non-adjacent pairs
            f.write("\n\n--- NON-ADJACENT PAIRS ---\n")
            for i, result in enumerate(results):
                if 'error' in result or result.get('pair_type') != 'non_adjacent':
                    continue
                    
                pair = result['pair']
                pair_type = result.get('pair_type', 'unknown')
                pair_ground_truth = ground_truths[i] if i < len(ground_truths) else 0.0
                
                # Calculate additional statistics
                pair_bias = result['psi_hat'] - pair_ground_truth
                bias_se_ratio = pair_bias / result['se_hat'] if result['se_hat'] != 0 else float('inf')
                ci_width = result['ci_upper'] - result['ci_lower']
                
                f.write(f"Pair ({pair[0]}, {pair[1]}) [{pair_type}]: ")
                f.write(f"adjusted_estimate={result['psi_hat']:.6f} (original={result['unadjusted_psi_hat']:.6f}), ")
                f.write(f"se={result['se_hat']:.6f}, ")
                f.write(f"bias={pair_bias:.6f}, ")
                f.write(f"bias/SE={bias_se_ratio:.6f}, ")
                f.write(f"CI_width={ci_width:.6f}, ")
                f.write(f"95% CI=[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}], ")
                
                if result['epsilon_n'] is not None:
                    f.write(f"epsilon_n={result['epsilon_n']:.6f}, ")
                
                # Add autoTML-specific outputs
                if method.lower() == 'autotml':
                    if result.get('avg_eif_term') is not None:
                        f.write(f"avg_eif_term={result['avg_eif_term']:.6f}, ")
                    if result.get('avg_eif') is not None:
                        f.write(f"avg_eif_value={result['avg_eif']:.6f}, ")
                
                # Add validation loss information
                if result.get('mean_alpha_validation_loss') is not None:
                    f.write(f"mean_alpha_val_loss={result['mean_alpha_validation_loss']:.6f}, ")
                if result.get('alpha_validation_losses'):
                    val_losses_str = "[" + ", ".join([f"{loss:.6f}" for loss in result['alpha_validation_losses']]) + "]"
                    f.write(f"alpha_val_losses={val_losses_str}, ")
                
                contains_truth = result['ci_lower'] <= pair_ground_truth <= result['ci_upper']
                f.write(f"Contains truth: {contains_truth}, ")
                f.write(f"execution_time={result['execution_time']:.2f}s, ")
                f.write(f"theta_training_time={result.get('theta_training_time', 0):.2f}s, ")
                f.write(f"alpha_training_time={result.get('alpha_training_time', 0):.2f}s\n")
            
            # Write errors if any
            error_results = [result for result in results if 'error' in result]
            if error_results:
                f.write("\n\n--- ERRORS ---\n")
                for result in error_results:
                    f.write(f"Error processing pair {result['pair']}: {result['error']}\n")
            
            # Add total execution time at the end of the file for 'both' case
            f.write(f"\nTotal execution time: {end_time - start_time:.2f} seconds\n")
        
        else:
            # Single pair type - write all results normally
            for i, result in enumerate(results):
                if 'error' in result:
                    f.write(f"Error processing pair {result['pair']}: {result['error']}\n")
                    continue
                    
                pair = result['pair']
                pair_type = result.get('pair_type', 'unknown')
                pair_ground_truth = ground_truths[i] if i < len(ground_truths) else 0.0
                
                # Calculate additional statistics
                pair_bias = result['psi_hat'] - pair_ground_truth
                bias_se_ratio = pair_bias / result['se_hat'] if result['se_hat'] != 0 else float('inf')
                ci_width = result['ci_upper'] - result['ci_lower']
                
                f.write(f"Pair ({pair[0]}, {pair[1]}) [{pair_type}]: ")
                f.write(f"adjusted_estimate={result['psi_hat']:.6f} (original={result['unadjusted_psi_hat']:.6f}), ")
                f.write(f"se={result['se_hat']:.6f}, ")
                f.write(f"bias={pair_bias:.6f}, ")
                f.write(f"bias/SE={bias_se_ratio:.6f}, ")
                f.write(f"CI_width={ci_width:.6f}, ")
                f.write(f"95% CI=[{result['ci_lower']:.6f}, {result['ci_upper']:.6f}], ")
                
                if result['epsilon_n'] is not None:
                    f.write(f"epsilon_n={result['epsilon_n']:.6f}, ")
                
                # Add autoTML-specific outputs
                if method.lower() == 'autotml':
                    if result.get('avg_eif_term') is not None:
                        f.write(f"avg_eif_term={result['avg_eif_term']:.6f}, ")
                    if result.get('avg_eif') is not None:
                        f.write(f"avg_eif_value={result['avg_eif']:.6f}, ")
                
                # Add validation loss information
                if result.get('mean_alpha_validation_loss') is not None:
                    f.write(f"mean_alpha_val_loss={result['mean_alpha_validation_loss']:.6f}, ")
                if result.get('alpha_validation_losses'):
                    val_losses_str = "[" + ", ".join([f"{loss:.6f}" for loss in result['alpha_validation_losses']]) + "]"
                    f.write(f"alpha_val_losses={val_losses_str}, ")
                
                contains_truth = result['ci_lower'] <= pair_ground_truth <= result['ci_upper']
                f.write(f"Contains truth: {contains_truth}, ")
                f.write(f"execution_time={result['execution_time']:.2f}s, ")
                f.write(f"theta_training_time={result.get('theta_training_time', 0):.2f}s, ")
                f.write(f"alpha_training_time={result.get('alpha_training_time', 0):.2f}s\n")
        
        # Add total execution time at the end of the file
        f.write(f"\nTotal execution time: {end_time - start_time:.2f} seconds\n")
    
    # Now handle plotting - different approach based on pair_selection
    if pair_selection == 'both':
        # Create separate plots for adjacent and non-adjacent pairs
        valid_results = [r for r in results if 'error' not in r]
        adj_results = [r for r in valid_results if r.get('pair_type') == 'adjacent']
        non_adj_results = [r for r in valid_results if r.get('pair_type') == 'non_adjacent']
        
        # Plot adjacent pairs
        if adj_results:
            plt.figure(figsize=(24, 12))
            adj_indices = [i for i, r in enumerate(adj_results)]
            adj_estimates = [r['psi_hat'] for r in adj_results]
            adj_lower = [r['ci_lower'] for r in adj_results]
            adj_upper = [r['ci_upper'] for r in adj_results]
            adj_ground_truth = 1/temperature
            adj_avg_estimate = np.mean(adj_estimates)
            
            plt.errorbar(adj_indices, adj_estimates, 
                        yerr=[
                            np.array(adj_estimates) - np.array(adj_lower),
                            np.array(adj_upper) - np.array(adj_estimates)
                        ],
                        fmt='o', capsize=4, markersize=6, elinewidth=1.5, alpha=0.8)
            
            plt.axhline(y=adj_ground_truth, color='r', linestyle='-', linewidth=3, 
                       label=f'Ground truth (1/T = {adj_ground_truth:.6f})')
            plt.axhline(y=adj_avg_estimate, color='g', linestyle='--', linewidth=3, 
                       label=f'Average estimate ({adj_avg_estimate:.6f})')
            
            plt.xlabel('Adjacent Pair Index', fontsize=20, fontweight='bold')
            plt.ylabel('Estimate (adjusted by dividing by 4)', fontsize=20, fontweight='bold')
            title = f'{method_name} Adjacent Pairs - 2D Ising Model\nT={temperature}, N={sample_size}, {model_description}'
            plt.title(title, fontsize=22, fontweight='bold', pad=20)
            
            plt.xticks(np.arange(0, len(adj_indices), 10), fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=16, loc='upper right')
            
            adj_avg_se = np.mean([r['se_hat'] for r in adj_results])
            adj_bias = adj_avg_estimate - adj_ground_truth
            adj_coverage_count = sum(1 for r in adj_results if r['ci_lower'] <= adj_ground_truth <= r['ci_upper'])
            adj_coverage_rate = adj_coverage_count / len(adj_results)
            
            stats_text = f'Avg SE: {adj_avg_se:.6f}\nBias: {adj_bias:.6f}\nCoverage: {adj_coverage_rate:.1%}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                     fontsize=16, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ising_results_adjacent.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot non-adjacent pairs
        if non_adj_results:
            plt.figure(figsize=(24, 12))
            non_adj_indices = [i for i, r in enumerate(non_adj_results)]
            non_adj_estimates = [r['psi_hat'] for r in non_adj_results]
            non_adj_lower = [r['ci_lower'] for r in non_adj_results]
            non_adj_upper = [r['ci_upper'] for r in non_adj_results]
            non_adj_ground_truth = 0.0
            non_adj_avg_estimate = np.mean(non_adj_estimates)
            
            plt.errorbar(non_adj_indices, non_adj_estimates, 
                        yerr=[
                            np.array(non_adj_estimates) - np.array(non_adj_lower),
                            np.array(non_adj_upper) - np.array(non_adj_estimates)
                        ],
                        fmt='o', capsize=4, markersize=6, elinewidth=1.5, alpha=0.8, color='orange')
            
            plt.axhline(y=non_adj_ground_truth, color='r', linestyle='-', linewidth=3, 
                       label=f'Ground truth = {non_adj_ground_truth:.6f}')
            plt.axhline(y=non_adj_avg_estimate, color='g', linestyle='--', linewidth=3, 
                       label=f'Average estimate ({non_adj_avg_estimate:.6f})')
            
            plt.xlabel('Non-Adjacent Pair Index', fontsize=20, fontweight='bold')
            plt.ylabel('Estimate (adjusted by dividing by 4)', fontsize=20, fontweight='bold')
            title = f'{method_name} Non-Adjacent Pairs - 2D Ising Model\nT={temperature}, N={sample_size}, {model_description}'
            plt.title(title, fontsize=22, fontweight='bold', pad=20)
            
            plt.xticks(np.arange(0, len(non_adj_indices), 10), fontsize=16)
            plt.yticks(fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=16, loc='upper right')
            
            non_adj_avg_se = np.mean([r['se_hat'] for r in non_adj_results])
            non_adj_bias = non_adj_avg_estimate - non_adj_ground_truth
            non_adj_coverage_count = sum(1 for r in non_adj_results if r['ci_lower'] <= non_adj_ground_truth <= r['ci_upper'])
            non_adj_coverage_rate = non_adj_coverage_count / len(non_adj_results)
            
            stats_text = f'Avg SE: {non_adj_avg_se:.6f}\nBias: {non_adj_bias:.6f}\nCoverage: {non_adj_coverage_rate:.1%}'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                     fontsize=16, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ising_results_non_adjacent.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    else:
        # Single plot for either adjacent or non-adjacent pairs
        valid_results = [r for r in results if 'error' not in r]
        avg_estimate = np.mean([r['psi_hat'] for r in valid_results])
        avg_se = np.mean([r['se_hat'] for r in valid_results])
        ground_truth = ground_truths[0]  # All pairs have same ground truth
        bias = avg_estimate - ground_truth
        coverage_count = sum(1 for r in valid_results if r['ci_lower'] <= ground_truth <= r['ci_upper'])
        coverage_rate = coverage_count / len(valid_results)
        
        plt.figure(figsize=(24, 12))
        
        pair_indices = [i for i, r in enumerate(results) if 'error' not in r]
        pair_estimates = [r['psi_hat'] for r in results if 'error' not in r]
        pair_lower = [r['ci_lower'] for r in results if 'error' not in r]
        pair_upper = [r['ci_upper'] for r in results if 'error' not in r]
        
        color = 'blue' if pair_selection == 'adjacent' else 'orange'
        plt.errorbar(pair_indices, pair_estimates, 
                    yerr=[
                        np.array(pair_estimates) - np.array(pair_lower),
                        np.array(pair_upper) - np.array(pair_estimates)
                    ],
                    fmt='o', capsize=4, markersize=6, elinewidth=1.5, alpha=0.8, color=color)
        
        plt.axhline(y=ground_truth, color='r', linestyle='-', linewidth=3, 
                   label=f'Ground truth = {ground_truth:.6f}')
        plt.axhline(y=avg_estimate, color='g', linestyle='--', linewidth=3, 
                   label=f'Average estimate ({avg_estimate:.6f})')
        
        pair_type_title = 'Adjacent' if pair_selection == 'adjacent' else 'Non-Adjacent'
        plt.xlabel(f'{pair_type_title} Pair Index', fontsize=20, fontweight='bold')
        plt.ylabel('Estimate (adjusted by dividing by 4)', fontsize=20, fontweight='bold')
        title = f'{method_name} {pair_type_title} Pairs - 2D Ising Model\nT={temperature}, N={sample_size}, {model_description}'
        plt.title(title, fontsize=22, fontweight='bold', pad=20)
        
        plt.xticks(np.arange(0, len(pair_indices), 10), fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=16, loc='upper right')
        
        stats_text = f'Avg SE: {avg_se:.6f}\nBias: {bias:.6f}\nCoverage: {coverage_rate:.1%}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 fontsize=16, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        suffix = '_adjacent' if pair_selection == 'adjacent' else '_non_adjacent'
        plt.savefig(f"{output_dir}/ising_results{suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create comprehensive results object for future reuse
    ising_estimation_results = {
        # Experiment configuration
        'experiment_config': {
            'data_file': data_file,
            'method': method,
            'method_name': method_name,
            'model_type_theta': model_type_theta,
            'model_type_alpha': model_type_alpha,
            'model_description': model_description,
            'temperature': temperature,
            'sample_size': sample_size,
            'max_workers': max_workers,
            'pair_selection': pair_selection,
            'non_adjacent_seed': non_adjacent_seed if pair_selection in ['non_adjacent', 'both'] else None,
            'execution_time_total': end_time - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        
        # Data information
        'data_info': {
            'data_shape': data.shape,
            'n_samples': data.shape[0],
            'n_positions': data.shape[1],
            'lattice_size': int(np.sqrt(data.shape[1])),
            'pair_selection': pair_selection,
            'n_total_pairs': len(pairs_to_process),
            'pairs_processed': pairs_to_process.tolist(),
            'pair_types': pair_types
        },
        
        # Ground truth and summary statistics (updated for pair types)
        'summary_statistics': {},
        
        # Individual pair results (all details)
        'individual_results': results,
        
        # Plotting data (extracted for convenience)
        'plotting_data': {},
        
        # Additional diagnostics (if available)
        'diagnostics': {
            'execution_times': [r.get('execution_time') for r in results if 'error' not in r and r.get('execution_time') is not None],
            'theta_training_times': [r.get('theta_training_time') for r in results if 'error' not in r and r.get('theta_training_time') is not None],
            'alpha_training_times': [r.get('alpha_training_time') for r in results if 'error' not in r and r.get('alpha_training_time') is not None],
            'mean_alpha_validation_losses': [r.get('mean_alpha_validation_loss') for r in results if 'error' not in r and r.get('mean_alpha_validation_loss') is not None],
            'alpha_validation_losses': [r.get('alpha_validation_losses') for r in results if 'error' not in r and r.get('alpha_validation_losses') is not None],
        }
    }
    
    # Save the comprehensive results object to disk
    pair_suffix = f"_{pair_selection}"
    if pair_selection in ['non_adjacent', 'both']:
        pair_suffix += f"_seed{non_adjacent_seed}"
    results_filename = f"{output_dir}/ising_estimation_results_{method}_{model_type_theta}_{model_type_alpha}_T{temperature}_{sample_size}{pair_suffix}.pkl"
    with open(results_filename, 'wb') as f:
        pickle.dump(ising_estimation_results, f)
    
    print(f"Comprehensive results object saved to {results_filename}")
    print(f"Results saved to {output_dir}")
    
    # Plot individual learning curves for each pair in wandb before finishing
    print("Creating individual learning curve plots for each pair in wandb...")
    plot_individual_learning_curves_wandb(results, method, model_type_theta, model_type_alpha)
    
    # Finish wandb run
    wandb.finish()
    
    # Return both the original results and the comprehensive object
    return results, ising_estimation_results

if __name__ == "__main__":
    # Path to the data file
    data_file = "./data/states_T1.8_L8_10k.txt"
    
    # Configuration parameters
    method = 'autoTML'  # Can be changed to 'autoDML'
    model_type_theta = 'constant'  # Can be changed to 'linear', 'nn', 'constant', or 'X_only'
    model_type_alpha = 'nn'  # Can be changed to 'linear', 'nn', or 'polynomial'
    
    # Pair selection parameters
    pair_selection = 'both'      # Can be 'adjacent', 'non_adjacent', or 'both'
    non_adjacent_seed = 42       # Random seed for non-adjacent pair selection (for reproducibility)
    
    # Set the number of CPU workers for parallel processing
    max_workers = 13  
    
    # Training hyperparameters
    n_splits = 5             # number of splits for cross-fitting
    lambda_theta = 0.001     # regularization for theta model
    lambda_alpha = 0.1       # regularization for alpha model
    l1_ratio = 1             # L1/L2 regularization ratio for polynomial alpha model (1 for L1, 0 for L2)
    theta_epochs = 30       # training epochs for theta model
    alpha_epochs = 350       # training epochs for alpha model
    batch_size_theta = 512   # batch size for theta training
    batch_size_alpha = 128   # batch size for alpha training
    lr_theta = 0.01          # learning rate for theta model
    lr_alpha = 0.001         # learning rate for alpha model
    alpha_nn_setup = [64]    # Neural network architecture for alpha model (only used when model_type_alpha='nn')
                               # Example: [64, 64] means two hidden layers with 64 nodes each

    # Run the analysis
    results, ising_estimation_results = analyze_ising_model(
        data_file, 
        max_workers=max_workers, 
        method=method, 
        model_type_theta=model_type_theta, 
        model_type_alpha=model_type_alpha,
        pair_selection=pair_selection,
        non_adjacent_seed=non_adjacent_seed,
        n_splits=n_splits,
        lambda_theta=lambda_theta,
        lambda_alpha=lambda_alpha,
        l1_ratio=l1_ratio,
        theta_epochs=theta_epochs,
        alpha_epochs=alpha_epochs,
        batch_size_theta=batch_size_theta,
        batch_size_alpha=batch_size_alpha,
        lr_theta=lr_theta,
        lr_alpha=lr_alpha,
        alpha_nn_setup=alpha_nn_setup
    )
    
    # Print the final summary based on pair selection
    valid_results = [r for r in results if 'error' not in r]
    
    # Extract temperature for ground truth calculation
    if "T1.8" in data_file:
        temperature = 1.8
    elif "T2.3" in data_file:
        temperature = 2.3
    elif "T3.0" in data_file:
        temperature = 3.0
    else:
        temperature = 3.0
    
    # Extract sample size from data file name
    if "10k" in data_file:
        sample_size = "10k"
    elif "100k" in data_file:
        sample_size = "100k"
    elif "1M" in data_file:
        sample_size = "1M"
    else:
        sample_size = f"{len(valid_results)}pairs"
    
    print("\nFinal Summary:")
    print(f"Pair selection: {pair_selection}")
    
    if pair_selection == 'both':
        # Separate summaries for adjacent and non-adjacent pairs
        adj_results = [r for r in valid_results if r.get('pair_type') == 'adjacent']
        non_adj_results = [r for r in valid_results if r.get('pair_type') == 'non_adjacent']
        
        if adj_results:
            adj_avg_estimate = np.mean([r['psi_hat'] for r in adj_results])
            adj_ground_truth = 1/temperature
            adj_coverage_count = sum(1 for r in adj_results if r['ci_lower'] <= adj_ground_truth <= r['ci_upper'])
            
            print(f"\nADJACENT PAIRS ({len(adj_results)} pairs):")
            print(f"  Average estimate: {adj_avg_estimate:.6f}")
            print(f"  Ground truth (1/T): {adj_ground_truth:.6f}")
            print(f"  Difference: {adj_avg_estimate - adj_ground_truth:.6f}")
            print(f"  Relative error: {100 * abs(adj_avg_estimate - adj_ground_truth) / adj_ground_truth:.2f}%")
            print(f"  Coverage rate: {adj_coverage_count/len(adj_results):.2%} ({adj_coverage_count}/{len(adj_results)} CIs contain ground truth)")
        
        if non_adj_results:
            non_adj_avg_estimate = np.mean([r['psi_hat'] for r in non_adj_results])
            non_adj_ground_truth = 0.0
            non_adj_coverage_count = sum(1 for r in non_adj_results if r['ci_lower'] <= non_adj_ground_truth <= r['ci_upper'])
            
            print(f"\nNON-ADJACENT PAIRS ({len(non_adj_results)} pairs):")
            print(f"  Average estimate: {non_adj_avg_estimate:.6f}")
            print(f"  Ground truth: {non_adj_ground_truth:.6f}")
            print(f"  Difference: {non_adj_avg_estimate - non_adj_ground_truth:.6f}")
            print(f"  Coverage rate: {non_adj_coverage_count/len(non_adj_results):.2%} ({non_adj_coverage_count}/{len(non_adj_results)} CIs contain ground truth)")
    
    else:
        # Single type summary
        avg_estimate = np.mean([r['psi_hat'] for r in valid_results])
        ground_truth = 1/temperature if pair_selection == 'adjacent' else 0.0
        coverage_count = sum(1 for r in valid_results if r['ci_lower'] <= ground_truth <= r['ci_upper'])
        
        pair_type_name = 'Adjacent' if pair_selection == 'adjacent' else 'Non-Adjacent'
        print(f"{pair_type_name.upper()} PAIRS ({len(valid_results)} pairs):")
        print(f"  Average estimate: {avg_estimate:.6f}")
        if pair_selection == 'adjacent':
            print(f"  Ground truth (1/T): {ground_truth:.6f}")
            print(f"  Relative error: {100 * abs(avg_estimate - ground_truth) / ground_truth:.2f}%")
        else:
            print(f"  Ground truth: {ground_truth:.6f}")
        print(f"  Difference: {avg_estimate - ground_truth:.6f}")
        print(f"  Coverage rate: {coverage_count/len(valid_results):.2%} ({coverage_count}/{len(valid_results)} CIs contain ground truth)")
    
    # Generate filename suffix for saving
    pair_suffix = f"_{pair_selection}"
    if pair_selection in ['non_adjacent', 'both']:
        pair_suffix += f"_seed{non_adjacent_seed}"
    
    print("\n" + "="*60)
    print("COMPREHENSIVE RESULTS OBJECT SAVED")
    print("="*60)
    print(f"Object name: 'ising_estimation_results'")
    print(f"Saved to file: ising_estimation_results_{method}_{model_type_theta}_{model_type_alpha}_T{temperature}_{sample_size}{pair_suffix}.pkl")
    print(f"Contains:")
    print(f"  - Experiment configuration (including pair selection: {pair_selection})")
    print(f"  - Data information") 
    print(f"  - Summary statistics")
    print(f"  - Individual pair results")
    print(f"  - Plotting data")
    print(f"  - Diagnostics")
    print(f"\nTo reuse later:")
    print(f"  results = load_ising_estimation_results('ising_results/ising_estimation_results_{method}_{model_type_theta}_{model_type_alpha}_T{temperature}_{sample_size}{pair_suffix}.pkl')")
    print(f"  plot_from_saved_results(results)")
    print("="*60)