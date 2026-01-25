"""
Test Confidence-Aware RAG System
Verify that low-confidence queries are properly handled
"""

from rag_modules import rag_engine

def test_confidence_system():
    
    # Test cases with expected confidence levels
    test_cases = [
        # HIGH CONFIDENCE - Specific medical queries
        {
            "query": "What is bumblefoot disease?",
            "expected": "HIGH",
            "category": "Disease Definition"
        },
        {
            "query": "How do you treat bumblefoot in chickens?",
            "expected": "HIGH",
            "category": "Treatment Protocol"
        },
        {
            "query": "What are the symptoms of CRD chronic respiratory disease?",
            "expected": "HIGH",
            "category": "Clinical Symptoms"
        },
        {
            "query": "What causes fowlpox in poultry?",
            "expected": "HIGH",
            "category": "Disease Etiology"
        },
        {
            "query": "How to prevent scalyleg mites?",
            "expected": "HIGH",
            "category": "Prevention Strategy"
        },
        
        # MEDIUM CONFIDENCE - General queries
        {
            "query": "What vitamins help chicken immunity?",
            "expected": "MEDIUM",
            "category": "Nutrition/General"
        },
        {
            "query": "How often should I check chicken health?",
            "expected": "MEDIUM",
            "category": "Management Practice"
        },
        {
            "query": "What are common signs of illness in poultry?",
            "expected": "MEDIUM",
            "category": "General Health"
        },
        
        # LOW CONFIDENCE - Vague/Behavioral
        {
            "query": "Why do chickens sometimes look tired?",
            "expected": "LOW/MEDIUM",
            "category": "Behavioral/Vague"
        },
        {
            "query": "What makes chickens happy?",
            "expected": "LOW/MEDIUM",
            "category": "Behavioral/Vague"
        },
        
        # VERY LOW - Off-topic
        {
            "query": "What is the weather like today?",
            "expected": "VERY LOW",
            "category": "Off-topic"
        },
        {
            "query": "How to make chicken soup?",
            "expected": "VERY LOW",
            "category": "Off-topic"
        },
        {
            "query": "What is machine learning?",
            "expected": "VERY LOW",
            "category": "Off-topic"
        }
    ]
    
    from rag_modules.config import RAG_CONFIDENCE_THRESHOLD, RERANK_MODEL
    print(f"\nConfidence Threshold: {RAG_CONFIDENCE_THRESHOLD}")
    print(f"Re-ranker Model: {RERANK_MODEL}\n")
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {test['description']}")
        print(f"{'='*80}")
        print(f"Query: {test['query']}")
        print(f"Expected confidence: {test['expected']}")
        
        # Call confidence-aware search
        result = rag_engine.search_with_confidence(
            query=test['query'],
            k=3,
            disease_context=None
        )
        
        # Display results
        print(f"\nResults:")
        print(f"   - Retrieved chunks: {len(result['results'])}")
        print(f"   - Top score: {result['top_score']:.3f}")
        print(f"   - Is reliable: {result['is_reliable']}")
        
        # Show first chunk preview
        if result['results']:
            preview = result['results'][0][:200].replace('\n', ' ')
            print(f"   - First chunk preview: {preview}...")
        
        # Determine actual confidence level
        if result['top_score'] > 5.0:
            actual = "HIGH"
        elif result['top_score'] > 0:
            actual = "MEDIUM"
        else:
            actual = "LOW"
        
        print(f"\nActual confidence: {actual}")
        
        # Store for summary
        results.append({
            'query': test['query'],
            'expected': test['expected'],
            'actual': actual,
            'score': result['top_score'],
            'reliable': result['is_reliable']
        })
    
    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Query':<50} {'Expected':<15} {'Actual':<10} {'Score':<10} {'Reliable'}")
    print("-" * 100)
    
    for r in results:
        query_short = r['query'][:47] + "..." if len(r['query']) > 50 else r['query']
        reliable_icon = "✅" if r['reliable'] else "⚠️"
        print(f"{query_short:<50} {r['expected']:<15} {r['actual']:<10} {r['score']:<10.3f} {reliable_icon}")
    
    # Count reliable vs unreliable
    reliable_count = sum(1 for r in results if r['reliable'])
    unreliable_count = len(results) - reliable_count
    
    print(f"\nReliability Stats:")
    print(f"   - Reliable queries: {reliable_count}/{len(results)} ({reliable_count/len(results)*100:.1f}%)")
    print(f"   - Unreliable queries: {unreliable_count}/{len(results)} ({unreliable_count/len(results)*100:.1f}%)")
    
    # Check if off-topic queries are properly caught
    off_topic = [r for r in results if "weather" in r['query'].lower() or "cook" in r['query'].lower()]
    off_topic_caught = sum(1 for r in off_topic if not r['reliable'])
    
    print(f"\nOff-topic Detection:")
    print(f"   - Off-topic queries: {len(off_topic)}")
    print(f"   - Correctly flagged as unreliable: {off_topic_caught}/{len(off_topic)}")
    
    if off_topic_caught == len(off_topic):
        print(f"   - good news - all off-topic queries properly caught")
    else:
        print(f"   -  bad news - some off-topic queries marked as reliable")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_confidence_system()
