"""Attack-configuration grids for the evasion sweeps.

An experimental INSTANCE is a (base_config, seed) pair: the base_config fixes the attack's operating
point (e.g. the C2 check-in interval or the exfil payload volume) and the seed is the RNG draw within
it. The earlier sweeps fixed ONE base_config and varied only the seed, so every "instance" was the same
operating point redrawn — a methodology gap (no parameter diversity). Diversifying the base_config
across a grid lets a result be reported across operating points instead of one repeated configuration,
while the seed still gives independent draws within each.

build_configs() takes fixed fields plus per-axis value lists and returns the Cartesian product of base
configs (ordered and reproducible). expand_instances() crosses those with seeds and returns both the
flat instance list and the label arrays the sweep .npz stores (config_table / inst_config_idx /
inst_seed), so offline scoring can group results by config.
"""
import json
from itertools import product


def build_configs(base, **axes):
    """base: dict of fixed theta fields shared by every config. axes: name -> list of values to vary,
    combined as a Cartesian product. Returns an ordered list of complete base-config dicts. With no
    axes it returns [base] (the single-config case, identical to the old seed-only sweeps)."""
    if not axes:
        return [dict(base)]
    names = list(axes)
    out = []
    for combo in product(*(axes[n] for n in names)):
        cfg = dict(base)
        cfg.update(zip(names, combo))
        out.append(cfg)
    return out


def expand_instances(configs, seeds_per_config):
    """Cross each config with `seeds_per_config` RNG seeds. Seeds are globally unique across instances
    (config i, seed s -> i*seeds_per_config + s) so no two captures share an RNG draw. Returns
    (instances, labels): instances is a list of (config_idx, config_dict, seed); labels holds parallel
    lists for the .npz — config_table (one JSON string per config), inst_config_idx, inst_seed."""
    instances, inst_config_idx, inst_seed = [], [], []
    for ci, cfg in enumerate(configs):
        for s in range(seeds_per_config):
            seed = ci * seeds_per_config + s
            instances.append((ci, cfg, seed))
            inst_config_idx.append(ci)
            inst_seed.append(seed)
    labels = {
        "config_table": [json.dumps(c, sort_keys=True) for c in configs],
        "inst_config_idx": inst_config_idx,
        "inst_seed": inst_seed,
    }
    return instances, labels
