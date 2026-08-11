"""
Proposed算法参数优化脚本
支持网格搜索和贝叶斯优化两种方法
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from itertools import product

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_base_config
from main import run_proposed


# ============================================================
# 参数搜索空间定义
# ============================================================

# 网格搜索参数空间
GRID_SPACE = {
    1: {  # 第1轮：粗搜索
        'lrmc_rank_ratio': [0.5, 0.7, 0.8, 0.9, 0.95],
        'compensation_weight': [0.3, 0.5, 0.7, 0.9],
        'buffer_size': [3, 5, 7],
        'lrmc_max_iter': [20, 30],
        'lrmc_tol': [1e-3, 1e-4],
    },
    2: {  # 第2轮：细搜索（基于第1轮结果调整）
        'lrmc_rank_ratio': [0.75, 0.80, 0.85],
        'compensation_weight': [0.6, 0.7, 0.8],
        'buffer_size': [4, 5, 6],
        'lrmc_max_iter': [25, 30, 35],
        'lrmc_tol': [5e-5, 1e-4, 5e-4],
    },
    3: {  # 第3轮：微调
        'lrmc_rank_ratio': [0.78, 0.80, 0.82],
        'compensation_weight': [0.65, 0.70, 0.75],
        'buffer_size': [4, 5, 6],
        'lrmc_max_iter': [28, 30, 32],
        'lrmc_tol': [8e-5, 1e-4, 2e-4],
    },
}

# 贝叶斯优化搜索边界
BAYESIAN_BOUNDS = {
    'lrmc_rank_ratio': (0.5, 0.95),
    'compensation_weight': (0.1, 0.95),
    'buffer_size': (2, 10),       # 整数
    'lrmc_max_iter': (10, 50),    # 整数
    'lrmc_tol': (-5, -2),         # log10尺度
}


# ============================================================
# 工具函数
# ============================================================

def create_experiment_dir(method, round_num=None):
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if method == 'grid':
        dir_name = f"grid_round{round_num}_{timestamp}"
    else:
        dir_name = f"bayesian_{timestamp}"
    
    exp_dir = os.path.join('optimizer', dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def save_result(exp_dir, exp_id, params, results):
    """保存单个实验结果"""
    result = {
        'experiment_id': exp_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        'params': params,
        'results': results,
    }
    
    filename = f"{exp_id}.json"
    filepath = os.path.join(exp_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return filepath


def run_single_experiment(config, params, exp_id, exp_dir):
    """运行单个实验"""
    # 更新配置
    for key, value in params.items():
        config[key] = value
    
    # 运行Proposed算法
    start_time = time.time()
    try:
        metrics = run_proposed(config)
        elapsed = time.time() - start_time
        
        # 提取结果
        acc_history = metrics.get('acc', [])
        loss_history = metrics.get('loss', [])
        
        if len(acc_history) == 0:
            return None
        
        # 计算评估指标
        final_acc = acc_history[-1]
        final_loss = loss_history[-1] if loss_history else 0.0
        avg_acc = np.mean(acc_history[-5:]) if len(acc_history) >= 5 else final_acc
        
        # 计算收敛轮次（首次达到90%最终值）
        threshold = 0.9 * final_acc
        convergence_round = next(
            (i for i, acc in enumerate(acc_history) if acc >= threshold),
            len(acc_history)
        )
        
        results = {
            'final_acc': float(final_acc),
            'final_loss': float(final_loss),
            'avg_acc': float(avg_acc),
            'convergence_round': int(convergence_round),
            'latency_per_round': float(elapsed / len(acc_history)),
            'total_time': float(elapsed),
            'acc_history': [float(x) for x in acc_history],
            'loss_history': [float(x) for x in loss_history],
        }
        
        # 保存结果
        save_result(exp_dir, exp_id, params, results)
        
        return results
        
    except Exception as e:
        print(f"[Error] 实验 {exp_id} 失败: {e}")
        return None


def print_progress(exp_id, total, params, results):
    """打印进度"""
    if results:
        print(f"[{exp_id}/{total}] Acc={results['final_acc']:.2f}% | "
              f"Loss={results['final_loss']:.4f} | "
              f"Time={results['latency_per_round']:.2f}s/round | "
              f"Params: ratio={params['lrmc_rank_ratio']}, "
              f"weight={params['compensation_weight']}, "
              f"buffer={params['buffer_size']}")
    else:
        print(f"[{exp_id}/{total}] FAILED")


# ============================================================
# 网格搜索
# ============================================================

def grid_search(config, round_num, exp_dir):
    """网格搜索"""
    space = GRID_SPACE[round_num]
    
    # 生成所有参数组合
    param_names = list(space.keys())
    param_values = list(space.values())
    combinations = list(product(*param_values))
    
    total = len(combinations)
    print(f"\n{'='*60}")
    print(f"网格搜索 第{round_num}轮")
    print(f"参数空间: {len(combinations)} 组合")
    print(f"{'='*60}\n")
    
    best_acc = 0.0
    best_params = None
    all_results = []
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        exp_id = f"grid_{i+1:04d}"
        
        results = run_single_experiment(config.copy(), params, exp_id, exp_dir)
        print_progress(f"{i+1}/{total}", total, params, results)
        
        if results and results['final_acc'] > best_acc:
            best_acc = results['final_acc']
            best_params = params.copy()
            all_results.append((params, results))
    
    # 保存最佳结果
    if best_params:
        best_result = {
            'best_params': best_params,
            'best_acc': best_acc,
            'total_experiments': total,
            'round': round_num,
        }
        
        with open(os.path.join(exp_dir, 'best_result.json'), 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"第{round_num}轮搜索完成！")
        print(f"最佳准确率: {best_acc:.2f}%")
        print(f"最佳参数: {best_params}")
        print(f"结果保存至: {exp_dir}")
        print(f"{'='*60}\n")
    
    return best_params, best_acc


# ============================================================
# 贝叶斯优化
# ============================================================

def bayesian_optimization(config, n_iterations, exp_dir):
    """贝叶斯优化"""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    except ImportError:
        print("[Error] 贝叶斯优化需要 scikit-learn，请安装: pip install scikit-learn")
        return None, 0.0
    
    print(f"\n{'='*60}")
    print(f"贝叶斯优化")
    print(f"初始采样: 5 个点")
    print(f"迭代次数: {n_iterations} 次")
    print(f"{'='*60}\n")
    
    # 初始化高斯过程
    kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
    
    # 存储观测数据
    X_observed = []
    y_observed = []
    
    best_acc = 0.0
    best_params = None
    
    def sample_random_params():
        """随机采样参数"""
        params = {
            'lrmc_rank_ratio': np.random.uniform(0.5, 0.95),
            'compensation_weight': np.random.uniform(0.1, 0.95),
            'buffer_size': int(np.random.choice([2, 3, 5, 7, 10])),
            'lrmc_max_iter': int(np.random.choice([10, 20, 30, 50])),
            'lrmc_tol': float(10 ** np.random.uniform(-5, -2)),
        }
        return params
    
    def params_to_vector(params):
        """将参数转换为向量"""
        return np.array([
            params['lrmc_rank_ratio'],
            params['compensation_weight'],
            params['buffer_size'],
            params['lrmc_max_iter'],
            np.log10(params['lrmc_tol']),
        ])
    
    def vector_to_params(vector):
        """将向量转换为参数"""
        return {
            'lrmc_rank_ratio': float(np.clip(vector[0], 0.5, 0.95)),
            'compensation_weight': float(np.clip(vector[1], 0.1, 0.95)),
            'buffer_size': int(np.clip(np.round(vector[2]), 2, 10)),
            'lrmc_max_iter': int(np.clip(np.round(vector[3]), 10, 50)),
            'lrmc_tol': float(10 ** np.clip(vector[4], -5, -2)),
        }
    
    def expected_improvement(X, gp, y_best, xi=0.01):
        """计算Expected Improvement采集函数"""
        from scipy.stats import norm
        
        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        Z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    # 初始随机采样
    print("初始采样阶段...")
    for i in range(5):
        params = sample_random_params()
        exp_id = f"bayesian_init_{i+1}"
        
        results = run_single_experiment(config.copy(), params, exp_id, exp_dir)
        
        if results:
            X_observed.append(params_to_vector(params))
            y_observed.append(results['final_acc'])
            
            if results['final_acc'] > best_acc:
                best_acc = results['final_acc']
                best_params = params.copy()
            
            print_progress(f"Init {i+1}/5", 5, params, results)
    
    # 迭代优化
    print("\n迭代优化阶段...")
    for iteration in range(n_iterations):
        # 拟合高斯过程
        X = np.array(X_observed)
        y = np.array(y_observed)
        gp.fit(X, y)
        
        # 网格搜索采集函数最大值
        n_candidates = 1000
        X_candidates = np.random.uniform(
            [0.5, 0.1, 2, 10, -5],
            [0.95, 0.95, 10, 50, -2],
            size=(n_candidates, 5)
        )
        
        # 计算EI
        ei = expected_improvement(X_candidates, gp, best_acc)
        best_idx = np.argmax(ei)
        next_params = vector_to_params(X_candidates[best_idx])
        
        # 运行实验
        exp_id = f"bayesian_iter_{iteration+1}"
        results = run_single_experiment(config.copy(), next_params, exp_id, exp_dir)
        
        if results:
            X_observed.append(params_to_vector(next_params))
            y_observed.append(results['final_acc'])
            
            if results['final_acc'] > best_acc:
                best_acc = results['final_acc']
                best_params = next_params.copy()
            
            print_progress(f"Iter {iteration+1}/{n_iterations}", n_iterations, next_params, results)
    
    # 保存最佳结果
    if best_params:
        best_result = {
            'best_params': best_params,
            'best_acc': best_acc,
            'total_experiments': 5 + n_iterations,
            'method': 'bayesian',
        }
        
        with open(os.path.join(exp_dir, 'best_result.json'), 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"贝叶斯优化完成！")
        print(f"最佳准确率: {best_acc:.2f}%")
        print(f"最佳参数: {best_params}")
        print(f"结果保存至: {exp_dir}")
        print(f"{'='*60}\n")
    
    return best_params, best_acc


# ============================================================
# 结果分析
# ============================================================

def analyze_results(exp_dir):
    """分析实验结果"""
    results = []
    
    for filename in os.listdir(exp_dir):
        if filename.endswith('.json') and filename != 'best_result.json':
            filepath = os.path.join(exp_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
    
    if not results:
        print("没有找到实验结果")
        return
    
    # 按准确率排序
    results.sort(key=lambda x: x['results']['final_acc'], reverse=True)
    
    print(f"\n{'='*60}")
    print(f"实验结果分析")
    print(f"总实验数: {len(results)}")
    print(f"{'='*60}\n")
    
    # 打印Top 10
    print("Top 10 最佳参数组合:")
    print("-" * 60)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. Acc={r['results']['final_acc']:.2f}% | "
              f"ratio={r['params']['lrmc_rank_ratio']}, "
              f"weight={r['params']['compensation_weight']}, "
              f"buffer={r['params']['buffer_size']}, "
              f"iter={r['params']['lrmc_max_iter']}, "
              f"tol={r['params']['lrmc_tol']}")
    
    # 统计信息
    accs = [r['results']['final_acc'] for r in results]
    print(f"\n统计信息:")
    print(f"  平均准确率: {np.mean(accs):.2f}%")
    print(f"  标准差: {np.std(accs):.2f}%")
    print(f"  最小值: {np.min(accs):.2f}%")
    print(f"  最大值: {np.max(accs):.2f}%")


def compare_methods():
    """对比不同优化方法的结果"""
    optimizer_dir = 'optimizer'
    
    print(f"\n{'='*60}")
    print(f"优化方法对比")
    print(f"{'='*60}\n")
    
    for method_dir in os.listdir(optimizer_dir):
        method_path = os.path.join(optimizer_dir, method_dir)
        if os.path.isdir(method_path):
            best_file = os.path.join(method_path, 'best_result.json')
            if os.path.exists(best_file):
                with open(best_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"{method_dir}:")
                    print(f"  最佳准确率: {data['best_acc']:.2f}%")
                    print(f"  最佳参数: {data['best_params']}")
                    print()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Proposed算法参数优化')
    parser.add_argument('--method', type=str, default='grid',
                        choices=['grid', 'bayesian', 'analyze', 'compare'],
                        help='优化方法: grid(网格搜索), bayesian(贝叶斯优化), analyze(分析结果), compare(对比方法)')
    parser.add_argument('--round', type=int, default=1,
                        help='网格搜索轮次 (1, 2, 3)')
    parser.add_argument('--n_iterations', type=int, default=15,
                        help='贝叶斯优化迭代次数')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='实验结果目录（用于分析）')
    
    args = parser.parse_args()
    
    # 获取基础配置
    config = get_base_config()
    
    # 减少通信轮数以加速优化
    config['global_rounds'] = 10
    
    if args.method == 'grid':
        exp_dir = create_experiment_dir('grid', args.round)
        grid_search(config, args.round, exp_dir)
        
    elif args.method == 'bayesian':
        exp_dir = create_experiment_dir('bayesian')
        bayesian_optimization(config, args.n_iterations, exp_dir)
        
    elif args.method == 'analyze':
        if args.exp_dir:
            analyze_results(args.exp_dir)
        else:
            # 找到最新的实验目录
            dirs = [d for d in os.listdir('optimizer') if os.path.isdir(os.path.join('optimizer', d))]
            if dirs:
                latest_dir = os.path.join('optimizer', sorted(dirs)[-1])
                analyze_results(latest_dir)
            else:
                print("没有找到实验目录")
                
    elif args.method == 'compare':
        compare_methods()


if __name__ == '__main__':
    main()
