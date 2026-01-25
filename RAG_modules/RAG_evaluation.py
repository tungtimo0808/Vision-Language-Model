
import os
import sys
import json
from rag_modules.engine_v2 import rag_engine_v2

def evaluate_all_ground_truth():
    """Evaluate with all ground truth files"""
    
    gt_folder = "/storage/student6/GalLens_student6/knowledge_base/Ground-truth"
    
    if not os.path.exists(gt_folder):
        print(f"Ground truth folder not found: {gt_folder}")
        return
    
    # Find all ground truth JSON files
    gt_files = []
    for file in os.listdir(gt_folder):
        if file.endswith('.json') and 'ground_truth' in file.lower():
            gt_files.append(os.path.join(gt_folder, file))
    
    if not gt_files:
        print("No ground truth files found")
        return
    
    print(f"Found {len(gt_files)} ground truth files:\n")
    
    all_results = {}
    
    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        print(f"\nEvaluating: {filename}")
        print("-" * 70)
        
        metrics = rag_engine_v2.evaluate_with_ground_truth(gt_file)
        all_results[filename] = metrics
        
        if metrics:
            print(f" Hit@1: {metrics['hit_at_1']:.2%}")
            print(f" Hit@3: {metrics['hit_at_3']:.2%}")
            print(f" Hit@5: {metrics['hit_at_5']:.2%}")
            print(f" MRR: {metrics['mrr']:.4f}")
            
            if metrics.get('failed_queries'):
                print(f"\n Failed Queries: {len(metrics['failed_queries'])}")
                for i, fail in enumerate(metrics['failed_queries'][:3], 1):
                    print(f"      {i}. {fail['query'][:60]}...")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    if all_results:
        avg_hit_1 = sum(m['hit_at_1'] for m in all_results.values() if m) / len(all_results)
        avg_hit_3 = sum(m['hit_at_3'] for m in all_results.values() if m) / len(all_results)
        avg_hit_5 = sum(m['hit_at_5'] for m in all_results.values() if m) / len(all_results)
        avg_mrr = sum(m['mrr'] for m in all_results.values() if m) / len(all_results)
        
        print(f"Average Hit@1: {avg_hit_1:.2%}")
        print(f"Average Hit@3: {avg_hit_3:.2%}")
        print(f"Average Hit@5: {avg_hit_5:.2%}")
        print(f"Average MRR: {avg_mrr:.4f}")
    
    # Save results
    output_file = "/storage/student6/GalLens_student6/rag_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    # Check if DB is populated
    stats = rag_engine_v2.get_stats()
    if stats['db_size'] == 0:
        print("Vector database is empty")
        print("run ingest.py first")
        sys.exit(1)
    
    print(f"Database ready with {stats['db_size']} documents\n")
    evaluate_all_ground_truth()
