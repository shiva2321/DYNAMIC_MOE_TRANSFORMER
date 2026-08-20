import json
import numpy as np

with open("experiments/frozen_inference_routing_probe_results.json", "r") as f:
    data = json.load(f)

for seed_key in ["seed_1337", "seed_42"]:
    res = data[seed_key]
    print(f"\n=======================================================")
    print(f"  RESIDUAL AUXILIARY ROUTING ANALYSIS: {seed_key.upper()}")
    print(f"=======================================================")
    
    r_mat = res["routing_matrix"]
    domains = ["fineweb_edu", "python_code", "wikitext_facts", "natural_stories"]
    
    for l_str in ["0", "1", "2", "3"]:
        l = int(l_str)
        # Find global top shared expert for this layer (average over domains)
        layer_dist = np.array([r_mat[l_str][d] for d in domains]) # [4, num_exp]
        mean_exp_usage = layer_dist.mean(axis=0)
        top_shared_exp = int(np.argmax(mean_exp_usage))
        shared_mass = mean_exp_usage[top_shared_exp]
        
        # Strip top shared expert and re-normalize residual
        residual_dist = np.delete(layer_dist, top_shared_exp, axis=1)
        residual_norm = residual_dist / (residual_dist.sum(axis=1, keepdims=True) + 1e-8)
        
        # Compute pairwise cosine similarities on residual auxiliary distributions
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            
        p_res = residual_norm[1] # python
        w_res = residual_norm[2] # wiki
        s_res = residual_norm[3] # stories
        f_res = residual_norm[0] # web
        
        sim_code_wiki = cos_sim(p_res, w_res)
        sim_code_stories = cos_sim(p_res, s_res)
        sim_wiki_stories = cos_sim(w_res, s_res)
        
        print(f"\nLayer {l} (Top Shared Backbone: E{top_shared_exp} absorbing {shared_mass:.1f}% avg mass):")
        print(f"  • Raw Cosine Sim:      Py vs Wiki = {res['overlap_metrics'][f'layer_{l}']['code_vs_wiki']:.4f} | Py vs Stories = {res['overlap_metrics'][f'layer_{l}']['code_vs_stories']:.4f}")
        print(f"  • Residual Aux CosSim: Py vs Wiki = {sim_code_wiki:.4f} | Py vs Stories = {sim_code_stories:.4f} | Wiki vs Stories = {sim_wiki_stories:.4f}")
