"""
Proposed算法参数优化脚本（使用scikit-optimize）
真正调用run_proposed()运行联邦学习实验
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_base_config
from main import run_proposed

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("[Warning] scikit-optimize 未安装，贝叶斯优化不可用")
    print("请运行: pip install scikit-optimize")


# ============================================================
# 搜索空间定义
# ============================================================

SEARCH_SPACE = [
    Real(0.5, 0.99, name='lrmc_rank_ratio'),           # 对齐problem.md §5.1
    Real(0.1, 0.95, name='compensation_weight'),
    Integer(2, 10, name='buffer_size'),
    Integer(5, 50, name='lrmc_max_iter'),                # 从5开始
    Real(1e-5, 1e-2, name='lrmc_tol', prior='log-uniform'),  # 对数均匀分布
]

# 时延权重（控制对时延的惩罚力度）
LAMBDA_LATENCY = 0.001


# ============================================================
# 工具函数
# ============================================================

def create_experiment_dir():
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join('optimizer', f'bayesian_{timestamp}')
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
    filepath = os.path.join(exp_dir, f'{exp_id}.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return filepath


# ============================================================
# 贝叶斯优化
# ============================================================

def run_bayesian_optimization(n_calls=20, n_initial_points=5):
    """运行贝叶斯优化"""
    if not SKOPT_AVAILABLE:
        print("[Error] scikit-optimize 未安装，请运行: pip install scikit-optimize")
        return
    
    exp_dir = create_experiment_dir()
    
    print(f"\n{'='*60}")
    print(f"贝叶斯优化 (scikit-optimize)")
    print(f"初始采样: {n_initial_points} 个点")
    print(f"总迭代次数: {n_calls} 次")
    print(f"时延权重: {LAMBDA_LATENCY}")
    print(f"结果保存至: {exp_dir}")
    print(f"{'='*60}\n")
    
    # 读取基础配置（从config.py）
    base_config = get_base_config()
    print(f"数据集: {base_config['dataset']}")
    print(f"全局轮数: {base_config['global_rounds']}")
    print(f"掉队率: {base_config['straggler_ratio']}")
    print(f"{'='*60}\n")
    
    # 实验计数器
    experiment_counter = [0]
    
    @use_named_args(SEARCH_SPACE)
    def objective(lrmc_rank_ratio, compensation_weight, buffer_size, lrmc_max_iter, lrmc_tol):
        """目标函数：最小化 -Acc + λ*Latency"""
        experiment_counter[0] += 1
        exp_id = f"bayesian_{experiment_counter[0]:04d}"
        
        # 从config.py获取基础配置，只覆盖LRMC参数
        config = get_base_config()
        config['lrmc_rank_ratio'] = float(lrmc_rank_ratio)
        config['compensation_weight'] = float(compensation_weight)
        config['buffer_size'] = int(buffer_size)
        config['lrmc_max_iter'] = int(lrmc_max_iter)
        config['lrmc_tol'] = float(lrmc_tol)
        
        params = {
            'lrmc_rank_ratio': float(lrmc_rank_ratio),
            'compensation_weight': float(compensation_weight),
            'buffer_size': int(buffer_size),
            'lrmc_max_iter': int(lrmc_max_iter),
            'lrmc_tol': float(lrmc_tol),
        }
        
        print(f"\n[{experiment_counter[0]}/{n_calls}] 测试参数:")
        print(f"  ratio={lrmc_rank_ratio:.4f}, weight={compensation_weight:.4f}, "
              f"buffer={buffer_size}, max_iter={lrmc_max_iter}, tol={lrmc_tol}")
        
        try:
            # 真正运行联邦学习
            start_time = time.time()
            metrics = run_proposed(config)
            elapsed = time.time() - start_time
            
            # 提取结果
            acc_history = metrics.get('acc', [])
            loss_history = metrics.get('loss', [])
            
            if len(acc_history) == 0:
                print(f"  [Error] 实验返回空结果")
                return 9999.0
            
            final_acc = acc_history[-1]
            final_loss = loss_history[-1] if loss_history else 0.0
            latency_per_round = elapsed / len(acc_history)
            
            results = {
                'final_acc': float(final_acc),
                'final_loss': float(final_loss),
                'latency_per_round': float(latency_per_round),
                'total_time': float(elapsed),
                'acc_history': [float(x) for x in acc_history],
                'loss_history': [float(x) for x in loss_history],
            }
            
            # 保存结果
            save_result(exp_dir, exp_id, params, results)
            
            # 计算目标函数值：-Acc + λ*Latency + 稳定性惩罚
            acc_std = np.std(acc_history[-50:]) if len(acc_history) >= 50 else 0
            score = -final_acc + LAMBDA_LATENCY * latency_per_round + 0.1 * acc_std
            
            print(f"  -> Acc={final_acc:.2f}%, Loss={final_loss:.4f}, "
                  f"Time={latency_per_round:.2f}s/round, Score={score:.4f}")
            
            return score
            
        except Exception as e:
            print(f"  [Error] 实验失败: {e}")
            return 9999.0
    
    # 运行贝叶斯优化
    result = gp_minimize(
        func=objective,
        dimensions=SEARCH_SPACE,
        n_calls=n_calls,
        n_initial_points=n_initial_points,
        acq_func='EI',
        xi=0.05,
        noise='gaussian',
        random_state=42,
        verbose=False,
    )
    
    # 提取最佳结果
    best_params = {
        'lrmc_rank_ratio': float(result.x[0]),
        'compensation_weight': float(result.x[1]),
        'buffer_size': int(result.x[2]),
        'lrmc_max_iter': int(result.x[3]),
        'lrmc_tol': float(result.x[4]),
    }
    best_acc = -result.fun  # 转换回准确率
    
    # 保存最佳结果
    best_result = {
        'best_params': best_params,
        'best_acc': float(best_acc),
        'total_experiments': experiment_counter[0],
        'method': 'bayesian_skopt',
        'config': {
            'dataset': base_config['dataset'],
            'global_rounds': base_config['global_rounds'],
            'straggler_ratio': base_config['straggler_ratio'],
        },
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
    """分析实验结果，包含参数敏感性分析"""
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
              f"ratio={r['params']['lrmc_rank_ratio']:.4f}, "
              f"weight={r['params']['compensation_weight']:.4f}, "
              f"buffer={r['params']['buffer_size']}, "
              f"iter={r['params']['lrmc_max_iter']}, "
              f"tol={r['params']['lrmc_tol']:.2e}")
    
    # 统计信息
    accs = [r['results']['final_acc'] for r in results]
    print(f"\n统计信息:")
    print(f"  平均准确率: {np.mean(accs):.2f}%")
    print(f"  标准差: {np.std(accs):.2f}%")
    print(f"  最小值: {np.min(accs):.2f}%")
    print(f"  最大值: {np.max(accs):.2f}%")
    
    # 参数敏感性分析（problem.md §5.6）
    if len(results) >= 5:
        print(f"\n参数敏感性分析 (Pearson相关系数):")
        print("-" * 60)
        param_names = ['lrmc_rank_ratio', 'compensation_weight', 'buffer_size', 'lrmc_max_iter', 'lrmc_tol']
        for param in param_names:
            values = [r['params'][param] for r in results]
            if len(set(values)) > 1:  # 需要有变化才能计算相关系数
                corr = np.corrcoef(values, accs)[0, 1]
                direction = "↑" if corr > 0 else "↓"
                strength = "强" if abs(corr) > 0.5 else "中" if abs(corr) > 0.3 else "弱"
                print(f"  {param:25s}: r={corr:+.3f} ({strength}正相关 {direction})" if corr > 0 
                      else f"  {param:25s}: r={corr:+.3f} ({strength}负相关 {direction})")


def plot_optimization_results(exp_dir):
    """绘制参数-准确率散点图和敏感性分析"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Warning] matplotlib 未安装，无法绘图")
        print("请运行: pip install matplotlib")
        return
    
    results = []
    for filename in os.listdir(exp_dir):
        if filename.endswith('.json') and filename != 'best_result.json':
            filepath = os.path.join(exp_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append(data)
    
    if len(results) < 3:
        print("实验结果不足，无法绘图")
        return
    
    param_names = ['lrmc_rank_ratio', 'compensation_weight', 'buffer_size', 'lrmc_max_iter', 'lrmc_tol']
    param_labels = ['Rank Ratio (ρ)', 'Compensation Weight (α)', 'Buffer Size (B)', 
                    'Max Iterations (T_max)', 'Tolerance (ε)']
    accs = [r['results']['final_acc'] for r in results]
    
    # 创建2x3子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('LRMC Parameter Sensitivity Analysis', fontsize=14)
    
    for idx, (param, label) in enumerate(zip(param_names, param_labels)):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        values = [r['params'][param] for r in results]
        
        # 对tol参数使用对数坐标
        if param == 'lrmc_tol':
            ax.scatter(values, accs, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
            ax.set_xscale('log')
        else:
            ax.scatter(values, accs, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
        
        # 添加趋势线
        if len(set(values)) > 1:
            z = np.polyfit(values, accs, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(values), max(values), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'trend')
            
            # 计算相关系数
            corr = np.corrcoef(values, accs)[0, 1]
            ax.set_title(f'{label}\nr = {corr:.3f}')
        else:
            ax.set_title(f'{label}')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    axes[1][2].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(exp_dir, 'sensitivity_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"图片已保存至: {plot_path}")
    plt.show()


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Proposed算法参数优化（参考problem.md）')
    parser.add_argument('--method', type=str, default='bayesian',
                        choices=['bayesian', 'analyze', 'plot'],
                        help='优化方法: bayesian(贝叶斯优化), analyze(分析结果), plot(绘制图表)')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='贝叶斯优化迭代次数')
    parser.add_argument('--exp_dir', type=str, default=None,
                        help='实验结果目录（用于分析/绘图）')
    
    args = parser.parse_args()
    
    def get_latest_exp_dir():
        """获取最新的实验目录"""
        if os.path.exists('optimizer'):
            dirs = [d for d in os.listdir('optimizer') if os.path.isdir(os.path.join('optimizer', d))]
            if dirs:
                return os.path.join('optimizer', sorted(dirs)[-1])
        return None
    
    if args.method == 'bayesian':
        run_bayesian_optimization(n_calls=args.n_iterations, n_initial_points=5)
        
    elif args.method == 'analyze':
        exp_dir = args.exp_dir or get_latest_exp_dir()
        if exp_dir:
            analyze_results(exp_dir)
        else:
            print("没有找到实验目录")
            
    elif args.method == 'plot':
        exp_dir = args.exp_dir or get_latest_exp_dir()
        if exp_dir:
            plot_optimization_results(exp_dir)
        else:
            print("没有找到实验目录")


if __name__ == '__main__':
    main()
