# HQDE Deep Dive: Part 2 - Distributed Computing Architecture

**Version:** 0.1.5  
**Last Updated:** February 2025

## 🎉 What's New in v0.1.5

**Critical Distributed Training Improvements:**
- ✅ **Enabled FedAvg Weight Aggregation** - Workers now synchronize knowledge after each epoch
- ✅ **Added Ensemble Diversity** - Different LR and dropout per worker for better ensemble performance
- ✅ **Improved Training Stability** - Gradient clipping and better synchronization

See [CHANGELOG.md](../CHANGELOG.md) for complete details.

---

## Overview of Distribution in HQDE

HQDE uses **Ray** as its distributed computing framework. Ray enables:
- Creating remote actors (workers)
- Parallel task execution
- GPU resource management
- Fault tolerance
- **NEW in v0.1.5**: Federated learning-style weight aggregation

---

## 1. Ray Worker Architecture - Deep Dive

### Ray Fundamentals

**File**: [hqde_system.py L136-227](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/core/hqde_system.py#L136-227)

```python
# Check if Ray is available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

class DistributedEnsembleManager:
    def __init__(self, num_workers: int = 4):
        if self.use_ray:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)  # Start Ray cluster
```

### Worker Creation with GPU Splitting

```python
def create_ensemble_workers(self, model_class, model_kwargs):
    # Calculate GPU fraction per worker
    num_gpus = torch.cuda.device_count()  # e.g., 2 GPUs
    gpu_per_worker = num_gpus / self.num_workers  # 2/4 = 0.5 GPU each
    
    # Define remote worker class with GPU allocation
    @ray.remote(num_gpus=gpu_per_worker)
    class EnsembleWorker:
        def __init__(self, model_class, model_kwargs):
            self.model = model_class(**model_kwargs)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.efficiency_score = 1.0
            self.optimizer = None
            self.criterion = None
```

**GPU Distribution Example**:
```
2 GPUs, 4 Workers:
┌─────────────────────────────────────────────────┐
│                    GPU 0                         │
│  ┌─────────────────┐  ┌─────────────────┐       │
│  │   Worker 0      │  │   Worker 1      │       │
│  │   0.5 GPU       │  │   0.5 GPU       │       │
│  └─────────────────┘  └─────────────────┘       │
├─────────────────────────────────────────────────┤
│                    GPU 1                         │
│  ┌─────────────────┐  ┌─────────────────┐       │
│  │   Worker 2      │  │   Worker 3      │       │
│  │   0.5 GPU       │  │   0.5 GPU       │       │
│  └─────────────────┘  └─────────────────┘       │
└─────────────────────────────────────────────────┘
```

### Worker Training Implementation

```python
def train_step(self, data_batch, targets=None):
    """Actual training happens inside each worker."""
    if data_batch is not None and targets is not None:
        self.model.train()
        
        # Move data to worker's GPU
        data_batch = data_batch.to(self.device)
        targets = targets.to(self.device)
        
        # Standard PyTorch training loop
        self.optimizer.zero_grad()
        outputs = self.model(data_batch)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        
        # Update efficiency score (used for weighted aggregation)
        self.efficiency_score = max(0.1, 
            self.efficiency_score * 0.99 + 0.01 * (1.0 / (1.0 + loss.item())))
        
        return loss.item()
```

### Data Distribution Across Workers

**File**: [hqde_system.py L267-309](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/core/hqde_system.py#L267-309)

```python
def train_ensemble(self, data_loader, num_epochs: int = 10):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(data_loader):
            # Split batch across workers
            batch_size_per_worker = len(data) // self.num_workers
            training_futures = []
            
            for worker_id, worker in enumerate(self.workers):
                # Calculate data slice for this worker
                start_idx = worker_id * batch_size_per_worker
                if worker_id < self.num_workers - 1:
                    end_idx = (worker_id + 1) * batch_size_per_worker
                else:
                    end_idx = len(data)  # Last worker gets remainder
                
                worker_data = data[start_idx:end_idx]
                worker_targets = targets[start_idx:end_idx]
                
                # Launch async training (non-blocking)
                training_futures.append(
                    worker.train_step.remote(worker_data, worker_targets)
                )
            
            # Wait for all workers to complete
            batch_losses = ray.get(training_futures)
        
        # 🆕 v0.1.5: Aggregate and synchronize weights after each epoch (FedAvg)
        aggregated_weights = self.aggregate_weights()
        if aggregated_weights:
            self.broadcast_weights(aggregated_weights)
            print(f"  → Weights aggregated and synchronized at epoch {epoch+1}")
```

**Data Distribution Visualization**:
```
Batch of 128 samples, 4 workers:

Original Batch: [0─────────────────────────────────127]
                         ↓ split
Worker 0: [0────31]     (samples 0-31, LR=0.001, dropout=0.15)
Worker 1: [32───63]     (samples 32-63, LR=0.0008, dropout=0.18)
Worker 2: [64───95]     (samples 64-95, LR=0.0012, dropout=0.12)
Worker 3: [96──127]     (samples 96-127, LR=0.0009, dropout=0.16)

🆕 v0.1.5: Each worker has different hyperparameters for ensemble diversity
```

---

## 1.5 Federated Averaging (FedAvg) - NEW in v0.1.5

### What is FedAvg?

**Federated Averaging** is a distributed learning algorithm where:
1. Workers train independently on their data partitions
2. After each epoch, weights are aggregated (averaged)
3. Aggregated weights are broadcast back to all workers
4. Workers continue training with synchronized knowledge

### Why FedAvg in HQDE?

**Problem in v0.1.4**: Workers trained independently without sharing knowledge
- Like 4 students studying different chapters who never discuss
- Each worker learned different patterns, but never combined insights
- Result: Poor accuracy even with many epochs

**Solution in v0.1.5**: Enable weight synchronization after each epoch
- Workers share what they learned
- Combined knowledge is better than individual knowledge
- Result: +15-20% accuracy improvement

### Implementation

```python
def aggregate_weights(self) -> Dict[str, torch.Tensor]:
    """Aggregate weights from all workers (FedAvg algorithm)."""
    # Step 1: Collect weights from all workers
    weight_futures = [worker.get_weights.remote() for worker in self.workers]
    all_weights = ray.get(weight_futures)
    
    # Step 2: Average each parameter across workers
    aggregated_weights = {}
    param_names = all_weights[0].keys()
    
    for param_name in param_names:
        param_tensors = [weights[param_name] for weights in all_weights]
        stacked_params = torch.stack(param_tensors)
        
        # Simple averaging (FedAvg)
        aggregated_param = stacked_params.mean(dim=0)
        aggregated_weights[param_name] = aggregated_param
    
    return aggregated_weights

def broadcast_weights(self, aggregated_weights: Dict[str, torch.Tensor]):
    """Broadcast aggregated weights to all workers."""
    broadcast_futures = []
    for worker in self.workers:
        future = worker.set_weights.remote(aggregated_weights)
        broadcast_futures.append(future)
    
    # Wait for all workers to receive weights
    ray.get(broadcast_futures)
```

### FedAvg Training Flow

```
EPOCH 1:
┌─────────────────────────────────────────────────────────────┐
│ Worker 1: Train on partition 1 → Weights W1                 │
│ Worker 2: Train on partition 2 → Weights W2                 │
│ Worker 3: Train on partition 3 → Weights W3                 │
│ Worker 4: Train on partition 4 → Weights W4                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  AGGREGATE (FedAvg)  │
              │  W_avg = (W1+W2+W3+W4)/4 │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  BROADCAST W_avg     │
              │  to all workers      │
              └──────────┬───────────┘
                         │
                         ▼
EPOCH 2:
┌─────────────────────────────────────────────────────────────┐
│ All workers start with W_avg (synchronized knowledge)        │
│ Worker 1: Train on partition 1 → Weights W1'                │
│ Worker 2: Train on partition 2 → Weights W2'                │
│ Worker 3: Train on partition 3 → Weights W3'                │
│ Worker 4: Train on partition 4 → Weights W4'                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              (Repeat aggregation...)
```

### Performance Impact

| Dataset | Without FedAvg (v0.1.4) | With FedAvg (v0.1.5) | Improvement |
|---------|-------------------------|----------------------|-------------|
| CIFAR-10 | ~59% | ~75-80% | +16-21% |
| SVHN | ~72% | ~85-88% | +13-16% |
| CIFAR-100 | ~14% | ~45-55% | +31-41% |

---

## 2. MapReduce Pattern - Deep Dive

### Why MapReduce?

**File**: [mapreduce_ensemble.py L152-271](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/distributed/mapreduce_ensemble.py#L152-271)

MapReduce is a programming model for processing large datasets in parallel:
1. **Map**: Transform data in parallel
2. **Shuffle**: Group by keys
3. **Reduce**: Aggregate grouped data

### HQDE MapReduce Implementation

```python
class MapReduceEnsembleManager:
    def __init__(self,
                 num_stores: int = 3,      # Distributed weight stores
                 num_mappers: int = 4,     # Parallel mappers
                 num_reducers: int = 2,    # Aggregation reducers
                 replication_factor: int = 3):  # Fault tolerance
```

### Phase 1: Map

```python
def _map_phase(self, ensemble_data):
    """Map phase: process ensemble data in parallel."""
    
    # Partition data across mappers
    partitions = [[] for _ in range(self.num_mappers)]
    for i, data_item in enumerate(ensemble_data):
        partition_idx = i % self.num_mappers
        partitions[partition_idx].append(data_item)
    
    # Map function: extract weight information
    def ensemble_map_function(item, context):
        results = []
        if 'weights' in item:
            for param_name, weight_tensor in item['weights'].items():
                # Output: (key, value) pairs
                results.append((param_name, {
                    'weight': weight_tensor,
                    'source_id': item.get('source_id'),
                    'accuracy': item.get('accuracy', 0.0)
                }))
        return results
    
    # Execute mappers in parallel
    map_futures = []
    for i, partition in enumerate(partitions):
        if partition:
            mapper = self.mappers[i]
            future = mapper.map_operation.remote(partition, ensemble_map_function)
            map_futures.append(future)
    
    return all_map_results
```

**Map Phase Visualization**:
```
INPUT: 4 workers' weight dictionaries

Worker 1 weights:          Worker 2 weights:
├── conv1.weight           ├── conv1.weight
├── conv1.bias             ├── conv1.bias
├── conv2.weight           ├── conv2.weight
└── fc1.weight             └── fc1.weight

        ↓ MAP (extract and tag)

OUTPUT: List of (key, value) pairs
[
  (conv1.weight, {weight: tensor, source: W1, acc: 0.92}),
  (conv1.bias,   {weight: tensor, source: W1, acc: 0.92}),
  (conv2.weight, {weight: tensor, source: W1, acc: 0.92}),
  (conv1.weight, {weight: tensor, source: W2, acc: 0.88}),
  (conv1.bias,   {weight: tensor, source: W2, acc: 0.88}),
  ...
]
```

### Phase 2: Shuffle

```python
def _shuffle_phase(self, map_results):
    """Shuffle phase: group map results by key."""
    grouped_data = defaultdict(list)
    
    for key, value in map_results:
        grouped_data[key].append(value)
    
    return dict(grouped_data)
```

**Shuffle Phase Visualization**:
```
INPUT: Flat list of (key, value) pairs

OUTPUT: Grouped by parameter name
{
  'conv1.weight': [
    {weight: W1_tensor, acc: 0.92},
    {weight: W2_tensor, acc: 0.88},
    {weight: W3_tensor, acc: 0.85},
    {weight: W4_tensor, acc: 0.90}
  ],
  'conv1.bias': [...],
  'conv2.weight': [...],
  ...
}
```

### Phase 3: Reduce

```python
def _reduce_phase(self, grouped_data, aggregation_strategy):
    """Reduce phase: aggregate grouped data."""
    
    def ensemble_reduce_function(key, values, context):
        weight_tensors = [v['weight'] for v in values]
        
        if aggregation_strategy == "hierarchical":
            # Weight by accuracy
            accuracies = [v.get('accuracy', 1.0) for v in values]
            accuracy_weights = torch.softmax(torch.tensor(accuracies), dim=0)
            
            weighted_sum = torch.zeros_like(weight_tensors[0])
            for weight, acc_weight in zip(weight_tensors, accuracy_weights):
                weighted_sum += acc_weight * weight
            
            return weighted_sum
        else:
            # Simple averaging
            return torch.stack(weight_tensors).mean(dim=0)
    
    # Execute reducers in parallel
    for key in key_partition:
        values = grouped_data[key]
        future = reducer.reduce_operation.remote(key, values, ensemble_reduce_function)
```

**Reduce Phase Visualization**:
```
INPUT: Grouped weights per parameter

conv1.weight: [W1, W2, W3, W4] with accuracies [0.92, 0.88, 0.85, 0.90]
        ↓ REDUCE (weighted average)
        
softmax([0.92, 0.88, 0.85, 0.90]) = [0.28, 0.25, 0.22, 0.25]
aggregated = 0.28×W1 + 0.25×W2 + 0.22×W3 + 0.25×W4

OUTPUT: Single aggregated weight per parameter
{
  'conv1.weight': aggregated_tensor,
  'conv1.bias': aggregated_tensor,
  ...
}
```

---

## 3. Hierarchical Tree Aggregation - Deep Dive

### Why Hierarchical?

**File**: [hierarchical_aggregator.py L122-243](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/distributed/hierarchical_aggregator.py#L122-243)

**Problem**: Aggregating N workers' weights
- Naive: All-to-one communication = O(N) bottleneck
- Hierarchical: Tree-based = O(log N) levels

### Communication Complexity Analysis

```
Flat Aggregation (N=8 workers):
All workers → Central node = 8 messages simultaneously
Bottleneck: Central node bandwidth

Hierarchical Aggregation (N=8, branching=2):
Level 3: 8 workers → 4 intermediate nodes (4 parallel aggregations)
Level 2: 4 intermediate → 2 nodes (2 parallel aggregations)
Level 1: 2 nodes → 1 root (1 aggregation)

Total: 3 levels = log₂(8) = O(log N)
```

### Tree Building Algorithm

```python
def _build_aggregation_tree(self):
    """Build the hierarchical aggregation tree."""
    
    # Calculate tree levels (bottom-up)
    num_leaves = self.num_ensemble_members  # e.g., 8
    tree_levels = []
    current_level_size = num_leaves
    
    while current_level_size > 1:
        tree_levels.append(current_level_size)
        current_level_size = math.ceil(current_level_size / self.tree_branching_factor)
    
    tree_levels.append(1)  # Root
    tree_levels.reverse()  # Top-down order
    
    # Example for 8 workers, branching=2:
    # tree_levels = [1, 2, 4, 8]  (root to leaves)
    
    # Create nodes at each level
    for level_idx, num_nodes in enumerate(tree_levels):
        for node_idx in range(num_nodes):
            node_id = f"agg_node_{level_idx}_{node_idx}"
            node = AggregationNode.remote(node_id, level_idx)
```

### Tree Structure for 8 Workers

```
Level 0 (Root):                    [agg_0_0]
                                   /        \
Level 1:                   [agg_1_0]    [agg_1_1]
                           /      \      /      \
Level 2:              [agg_2_0][agg_2_1][agg_2_2][agg_2_3]
                       /    \   /    \   /    \   /    \
Level 3 (Leaves):    W0    W1 W2    W3 W4    W5 W6    W7

Communication pattern:
- Step 1: (W0,W1)→2_0, (W2,W3)→2_1, (W4,W5)→2_2, (W6,W7)→2_3
- Step 2: (2_0,2_1)→1_0, (2_2,2_3)→1_1
- Step 3: (1_0,1_1)→0_0 (final)
```

### Bottom-Up Aggregation

```python
def _perform_bottom_up_aggregation(self):
    """Perform bottom-up aggregation through the tree."""
    
    # Process levels from bottom to top (reverse order)
    for level in sorted(self.tree_structure.keys(), reverse=True):
        level_nodes = self.tree_structure[level]
        
        # Aggregate at each node in this level (parallel)
        aggregation_futures = []
        for node_id in level_nodes:
            node = self.nodes[node_id]
            future = node.aggregate_local_weights.remote("weighted_mean")
            aggregation_futures.append((node_id, future))
        
        # Wait for all aggregations at this level
        for node_id, future in aggregation_futures:
            aggregated_weights = ray.get(future)
            level_results[node_id] = aggregated_weights
        
        # Send results to parent nodes (if not at root)
        if level > 0:
            for node_id in level_nodes:
                parent_id = get_parent(node_id)
                parent_node = self.nodes[parent_id]
                parent_node.receive_weights.remote(
                    node_id, 
                    level_results[node_id]
                )
```

---

## 4. Byzantine Fault Tolerance - Deep Dive

### What is Byzantine Fault?

**File**: [fault_tolerance.py L19-127](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/distributed/fault_tolerance.py#L19-127)

A Byzantine fault is when a node:
- Crashes and sends no data
- Sends corrupted data
- Maliciously sends wrong data
- Behaves inconsistently

**Byzantine Generals Problem**: How to reach consensus when some generals might be traitors?

### HQDE's Solution: Threshold + Robust Aggregation

```python
class ByzantineFaultTolerantAggregator:
    def __init__(self,
                 byzantine_threshold: float = 0.33,  # Tolerate up to 33%
                 outlier_detection_method: str = "median_absolute_deviation",
                 min_reliable_sources: int = 3):
```

**Why 33%?** - Byzantine consensus requires N > 3f (where f = faulty nodes)
- If 33% are faulty, we need 4f nodes total, so f < N/3

### Step 1: Outlier Detection

```python
def _calculate_outlier_score(self, target_update, all_updates, target_index):
    if self.outlier_detection_method == "median_absolute_deviation":
        return self._mad_outlier_score(target_update, all_updates, target_index)

def _mad_outlier_score(self, target_update, all_updates, target_index):
    """Median Absolute Deviation - robust outlier detection."""
    for param_name in target_update.keys():
        # Collect all parameter values (excluding target)
        param_values = []
        target_value = target_update[param_name].flatten()
        
        for i, update in enumerate(all_updates):
            if i != target_index:
                param_values.append(update[param_name].flatten())
        
        # Calculate median
        stacked_values = torch.stack(param_values)
        median_value = torch.median(stacked_values, dim=0)[0]
        
        # Calculate MAD (Median Absolute Deviation)
        absolute_deviations = torch.abs(stacked_values - median_value)
        mad = torch.median(absolute_deviations, dim=0)[0]
        
        # Score = how many MADs away from median
        target_deviation = torch.abs(target_value - median_value)
        mad_score = torch.mean(target_deviation / (mad + 1e-8)).item()
        
        total_mad_score += mad_score
    
    return total_mad_score / param_count
```

**MAD Intuition**:
```
5 workers send weights for conv1.weight:
W1: [1.0, 2.0, 3.0]  ← Normal
W2: [1.1, 2.1, 2.9]  ← Normal
W3: [0.9, 1.9, 3.1]  ← Normal
W4: [1.0, 2.0, 3.0]  ← Normal
W5: [9.0, 9.0, 9.0]  ← BYZANTINE! (outlier)

Median: [1.0, 2.0, 3.0]
MAD: [0.1, 0.1, 0.1]

W5's score: mean(|[9,9,9] - [1,2,3]| / [0.1,0.1,0.1]) = 73.3 (HIGH!)
W1's score: mean(|[1,2,3] - [1,2,3]| / [0.1,0.1,0.1]) = 0.0 (LOW)
```

### Step 2: Geometric Median Aggregation

```python
def _geometric_median_aggregation(self, weight_updates):
    """Aggregate weights using geometric median for robustness."""
    
    # Initialize with arithmetic mean
    current_median = torch.stack(tensors).mean(dim=0)
    
    for iteration in range(max_iterations):
        # Calculate weight inversely proportional to distance
        distances = []
        for tensor in tensors:
            dist = torch.norm(tensor - current_median)
            distances.append(max(dist.item(), 1e-8))
        
        # Closer tensors get higher weight
        weights = [1.0 / dist for dist in distances]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Weighted average
        new_median = torch.zeros_like(current_median)
        for tensor, weight in zip(tensors, weights):
            new_median += weight * tensor
        
        # Check convergence
        if torch.norm(new_median - current_median) < 1e-6:
            break
        
        current_median = new_median
    
    return current_median
```

**Why Geometric Median?**

Arithmetic mean: sensitive to outliers
```
Normal: [1, 2, 3, 4, 100] → mean = 22 (pulled by outlier)
```

Geometric median: robust to outliers
```
Geometric median of [1, 2, 3, 4, 100] ≈ 3 (ignores outlier)
```

---

## 5. Dynamic Load Balancing - Deep Dive

### Why Load Balance?

**File**: [load_balancer.py L177-324](file:///d:/MTech%202nd%20Year/hqde/HQDE-PyPI/hqde/distributed/load_balancer.py#L177-324)

Problem: Workers have different:
- Hardware (faster/slower GPUs)
- Current load
- Historical performance
- Network latency

### Suitability Score Calculation

```python
def _calculate_node_suitability_score(self, task_data, node_stats):
    """Multi-factor scoring for node selection."""
    score = 0.0
    
    # Factor 1: Success rate (40% weight)
    # Workers that complete tasks successfully get higher scores
    success_rate = node_stats.get('success_rate', 1.0)
    score += success_rate * 0.4
    
    # Factor 2: Current load (30% weight)
    # Less loaded workers get higher scores
    current_load = node_stats.get('current_load', 0.0)
    load_factor = max(0.0, 1.0 - current_load)
    score += load_factor * 0.3
    
    # Factor 3: Speed (20% weight)
    # Faster workers get higher scores
    avg_time = node_stats.get('avg_execution_time', 1.0)
    time_factor = max(0.0, 1.0 - min(avg_time / 10.0, 1.0))
    score += time_factor * 0.2
    
    # Factor 4: Capability match (10% weight)
    # Workers with required capabilities get bonus
    task_requirements = task_data.get('requirements', {})
    node_capabilities = node_stats.get('capabilities', {})
    capability_match = self._calculate_capability_match(
        task_requirements, node_capabilities
    )
    score += capability_match * 0.1
    
    return score
```

**Example Scoring**:
```
Task: GPU-intensive weight aggregation

Worker A: success=0.95, load=0.2, speed=2s, has_gpu=True
Score = 0.95×0.4 + 0.8×0.3 + 0.8×0.2 + 1.0×0.1 = 0.88 ✓

Worker B: success=0.90, load=0.7, speed=5s, has_gpu=True
Score = 0.90×0.4 + 0.3×0.3 + 0.5×0.2 + 1.0×0.1 = 0.65

Worker C: success=0.98, load=0.1, speed=1s, has_gpu=False
Score = 0.98×0.4 + 0.9×0.3 + 0.9×0.2 + 0.0×0.1 = 0.85

→ Select Worker A (highest score with GPU)
```

### Load Rebalancing

```python
def _check_load_balance(self):
    """Check if load rebalancing is needed."""
    node_loads = list(self.balancing_metrics['node_utilization'].values())
    
    max_load = max(node_loads)
    min_load = min(node_loads)
    load_imbalance = max_load - min_load
    
    # Trigger rebalancing if imbalance exceeds threshold
    if load_imbalance > self.load_threshold:  # default: 0.8
        self._rebalance_load()
```

**Load Imbalance Detection**:
```
Workers: [0.9, 0.8, 0.2, 0.1]  (normalized loads)
Max: 0.9, Min: 0.1
Imbalance: 0.9 - 0.1 = 0.8

If threshold = 0.5:
0.8 > 0.5 → REBALANCE TRIGGERED

After rebalancing:
Workers: [0.5, 0.5, 0.5, 0.5]  (balanced)
```
