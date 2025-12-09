# ai_api_call_uma

## Using Gemini API
```bash
export GEMINI_API_KEY="your_key_here"
python generator_ai_training.py --api gemini --model large --num_samples 10 --out generated_training.csv
```
## Using Mistral API
```bash
export MISTRAL_API_KEY="your_key_here"
python generator_ai_training.py --api mistral --model medium --num_samples 5 --out generated_training.csv

## Or provide key directly
python generator_ai_training.py --api gemini --api_key YOUR_KEY --model medium --num_samples 100
```
# How It Works:

1. Loads all your CSV catalogs (umas, tracks, cards, skills)
2. For each sample:

    - Randomly selects a horse (uma), track, and support cards
    - Creates a detailed prompt explaining the training scenario
    - Calls the AI API to simulate training
    - AI returns final stats and acquired skills in JSON format


3. Outputs CSV with columns matching your predictive model's expected format

## To Get Free API Keys:

- Gemini: https://makersuite.google.com/app/apikey
- Mistral: https://console.mistral.ai/

## Output Format:
The generated CSV includes:

-  race_id, uma_id, uma_name, style, material, distance
- Speed, Stamina, Power, Guts, Wit (final trained stats)
- skills_offense, skills_endurance, skills_tactics
- card_ids, mood, finish_pos (for your predictive model)