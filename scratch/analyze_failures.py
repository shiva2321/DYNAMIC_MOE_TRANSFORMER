import json

def analyze():
    data = json.load(open('artifacts/benchmark_grounded_results.json'))
    details = data['gsm8k']['details']
    print(f"Total: {len(details)} | Correct: {data['gsm8k']['correct']}/{data['gsm8k']['num_samples']}")
    
    truncated = 0
    contains_gold = 0
    for it in details:
        idx = it['idx']
        corr = it['is_correct']
        gold = str(it['gold_num'])
        pred = str(it['pred_num'])
        raw = it['raw_output']
        ends_with_period = raw.strip().endswith(('.', '}', ')', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
        in_text = gold in raw
        if in_text:
            contains_gold += 1
        
        # Check if truncated mid-sentence
        last_chars = raw.strip()[-30:]
        is_cut = not (('####' in raw) or ('boxed' in raw) or ('answer is' in raw[-100:].lower()) or ('therefore' in raw[-80:].lower()))
        if is_cut:
            truncated += 1
            
        print(f"Q{idx:02d} | Corr: {str(corr):5s} | Gold: {gold:>7s} | Pred: {pred:>7s} | InText: {str(in_text):5s} | Cut: {is_cut} | End: {last_chars!r}")

    print(f"\nSummary:")
    print(f"  Total with Gold number in raw text: {contains_gold}/{len(details)} ({contains_gold/len(details)*100:.1f}%)")
    print(f"  Total potentially truncated before final answer: {truncated}/{len(details)}")

if __name__ == '__main__':
    analyze()
