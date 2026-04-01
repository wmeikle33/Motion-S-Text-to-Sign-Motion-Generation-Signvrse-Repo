samples_data = []

if DATASET_ROOT.exists():
    sample_dirs = [d for d in DATASET_ROOT.iterdir() if d.is_dir()]
    
    for sample_dir in tqdm(sample_dirs, desc="Loading samples"):
        sample_id = sample_dir.name
        metadata = load_metadata(sample_dir)
        bvh_count = count_bvh_files(sample_dir)
        
        if metadata:
            glosses = parse_glosses(metadata['gloss'])
            fingerspell_count = sum(1 for g in glosses if is_fingerspelling(g))
            
            samples_data.append({
                'sample_id': sample_id,
                'sentence': metadata['sentence'],
                'gloss': metadata['gloss'],
                'gloss_list': glosses,
                'gloss_count': len(glosses),
                'fingerspell_count': fingerspell_count,
                'bvh_count': bvh_count,
                'sentence_word_count': len(metadata['sentence'].split()),
                'sentence_char_count': len(metadata['sentence']),
            })
        else:
            samples_data.append({
                'sample_id': sample_id,
                'sentence': None,
                'gloss': None,
                'gloss_list': [],
                'gloss_count': 0,
                'fingerspell_count': 0,
                'bvh_count': bvh_count,
                'sentence_word_count': 0,
                'sentence_char_count': 0,
            })

df = pd.DataFrame(samples_data)
print(f"Loaded {len(df)} samples")
