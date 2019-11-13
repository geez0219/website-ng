

### per_replica_to_global
```python
per_replica_to_global(data)
```
Combine data from "per-replica" values.For multi-GPU training, data are distributed using `tf.distribute.Strategy.experimental_distribute_dataset`. Thismethod collects data from all replicas and combine them into one.

#### Args:

* **data** :  Distributed data.

#### Returns:

* **obj** :  Combined data from all replicas.