def load_metadata(sample_dir: Path) -> dict:
    metadata_path = sample_dir / "metadata.txt"
    if not metadata_path.exists():
        return None
    
    result = {}
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith("SENTENCE:"):
                    result['sentence'] = line.replace("SENTENCE:", "").strip()
                elif line.startswith("GLOSS:"):
                    result['gloss'] = line.replace("GLOSS:", "").strip()
    except Exception as e:
        return None
    
    return result if 'sentence' in result and 'gloss' in result else None

def parse_glosses(gloss_str: str) -> list:
    """Parse gloss string into list of individual glosses."""
    cleaned = gloss_str.replace("//", "").strip()
    return [g.strip() for g in cleaned.split() if g.strip()]

def count_bvh_files(sample_dir: Path) -> int:
    """Count BVH files in a sample directory."""
    return len(list(sample_dir.glob("*.bvh")))

def is_fingerspelling(gloss: str) -> bool:
    """Check if a gloss is a fingerspelled letter (single uppercase letter)."""
    return len(gloss) == 1 and gloss.isupper()
