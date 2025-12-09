#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uma Musume Training Generator using Generative AI
Generates synthetic training results that can be fed to the predictive AI

Usage:
  python generate_training_with_ai.py --api gemini --num_samples 100 --out generated_training.csv
  python generate_training_with_ai.py --api mistral --num_samples 50 --out generated_training.csv
"""

import argparse
import json
import os
import time
import random
from pathlib import Path
import pandas as pd
import requests

# ==================== API CONFIGURATIONS ====================
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# ==================== TRAINING SIMULATION PROMPT ====================
TRAINING_PROMPT_TEMPLATE = """You are an expert Uma Musume training simulator. You will simulate the training of a horse character.

# GAME MECHANICS:
- Training increases stats: Speed, Stamina, Power, Guts, Wit
- Base stats are provided, and training adds to these values
- Support cards provide bonuses to training efficiency
- Skills are acquired during training
- Final stats should be realistic for competitive racing (typically 800-1200 per stat)
- Mood affects training outcome (Good mood = better gains)

# TRAINING CONSTRAINTS:
- Each stat can gain between 200-600 points from training
- Speed and Stamina are crucial for racing
- Skills should match the horse's aptitudes and style
- Support cards influence which stats grow faster

# INPUT DATA:
Horse: {uma_name} (ID: {uma_id})
Base Stats:
- Speed: {Speed}
- Stamina: {Stamina}
- Power: {Power}
- Guts: {Guts}
- Wit: {Wit}

Aptitudes:
- Turf: {pref_turf}, Dirt: {pref_dirt}
- Sprint: {pref_sprint}, Mile: {pref_mile}, Medium: {pref_medium}, Long: {pref_long}
- Front: {apt_front}, Pace: {apt_pace}, Late: {apt_late}, End: {apt_end}
- Default Style: {default_style}

Growth Bonuses:
- Speed: {growth_spd_pct}%, Stamina: {growth_sta_pct}%, Power: {growth_pow_pct}%
- Guts: {growth_gut_pct}%, Wit: {growth_wit_pct}%

Support Cards: {cards}

Available Skills (choose 3-5 appropriate ones): {skills}

Race Track: {track_name} ({material}, {distance})

# YOUR TASK:
Simulate a complete training program and return ONLY a valid JSON object with this exact structure:
{{
  "final_speed": <number>,
  "final_stamina": <number>,
  "final_power": <number>,
  "final_guts": <number>,
  "final_wit": <number>,
  "acquired_skills": ["skill1", "skill2", "skill3"],
  "mood": "<Good|Normal|Bad>",
  "training_summary": "<brief 1-2 sentence summary>"
}}

Requirements:
- final_speed should be base + 200-600 (influenced by growth_spd_pct and cards)
- final_stamina should be base + 200-600 (influenced by growth_sta_pct and cards)
- final_power should be base + 200-600 (influenced by growth_pow_pct and cards)
- final_guts should be base + 200-600 (influenced by growth_gut_pct and cards)
- final_wit should be base + 200-600 (influenced by growth_wit_pct and cards)
- acquired_skills: 3-5 skills from the provided list that match the horse's aptitudes
- mood: randomly assign Good (70%), Normal (20%), or Bad (10%)
- Consider the support cards' specialties when deciding stat gains

Return ONLY the JSON, no additional text."""


# ==================== API CALL FUNCTIONS ====================
def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call Google Gemini API"""
    url = f"{GEMINI_API_URL}?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000,
        }
    }
    
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_mistral_api(prompt: str, api_key: str) -> str:
    """Call Mistral AI API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from AI response (handles markdown code blocks)"""
    # Remove markdown code blocks if present
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    return json.loads(text)


# ==================== DATA LOADING ====================
def load_datasets():
    """Load all required CSV files"""
    data = {}
    
    files = {
        "umas": "umas_letters_base.csv",
        "tracks": "tracks_catalog.csv",
        "cards": "support_cards_catalog.csv",
        "skills": "skills_catalog_full.csv",
        "aptitudes": "aptitude_mapping.csv"
    }
    
    for key, filename in files.items():
        if not Path(filename).exists():
            raise FileNotFoundError(f"Required file not found: {filename}")
        data[key] = pd.read_csv(filename)
    
    return data


# ==================== TRAINING GENERATION ====================
def generate_training_sample(uma_row, track_row, cards_sample, skills_list, api_type, api_key):
    """Generate one training result using AI"""
    
    # Prepare card names
    card_names = ", ".join(cards_sample["name"].tolist())
    
    # Prepare skills list (limit to 20 random skills)
    skills_sample = random.sample(skills_list, min(20, len(skills_list)))
    skills_str = ", ".join(skills_sample)
    
    # Format prompt
    prompt = TRAINING_PROMPT_TEMPLATE.format(
        uma_name=uma_row["uma_name"],
        uma_id=uma_row["uma_id"],
        Speed=uma_row["Speed"],
        Stamina=uma_row["Stamina"],
        Power=uma_row["Power"],
        Guts=uma_row["Guts"],
        Wit=uma_row["Wit"],
        pref_turf=uma_row["pref_turf"],
        pref_dirt=uma_row["pref_dirt"],
        pref_sprint=uma_row["pref_sprint"],
        pref_mile=uma_row["pref_mile"],
        pref_medium=uma_row["pref_medium"],
        pref_long=uma_row["pref_long"],
        apt_front=uma_row["apt_front"],
        apt_pace=uma_row["apt_pace"],
        apt_late=uma_row["apt_late"],
        apt_end=uma_row["apt_end"],
        default_style=uma_row["default_style"],
        growth_spd_pct=uma_row["growth_spd_pct"],
        growth_sta_pct=uma_row["growth_sta_pct"],
        growth_pow_pct=uma_row["growth_pow_pct"],
        growth_gut_pct=uma_row["growth_gut_pct"],
        growth_wit_pct=uma_row["growth_wit_pct"],
        cards=card_names,
        skills=skills_str,
        track_name=track_row["name"],
        material=track_row["material"],
        distance=track_row["distance"]
    )
    
    # Call API
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if api_type == "gemini":
                response_text = call_gemini_api(prompt, api_key)
            elif api_type == "mistral":
                response_text = call_mistral_api(prompt, api_key)
            else:
                raise ValueError(f"Unknown API type: {api_type}")
            
            # Parse JSON response
            result = extract_json_from_response(response_text)
            
            # Validate result
            required_keys = ["final_speed", "final_stamina", "final_power", 
                           "final_guts", "final_wit", "acquired_skills", "mood"]
            if not all(k in result for k in required_keys):
                raise ValueError("Missing required keys in AI response")
            
            return result
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                # Fallback: generate synthetic data
                print("  Using fallback synthetic generation")
                return generate_fallback_training(uma_row)
    
    return None


def generate_fallback_training(uma_row):
    """Fallback synthetic training when API fails"""
    base_gains = {
        "Speed": random.randint(200, 600),
        "Stamina": random.randint(200, 600),
        "Power": random.randint(200, 600),
        "Guts": random.randint(200, 600),
        "Wit": random.randint(200, 600)
    }
    
    return {
        "final_speed": uma_row["Speed"] + base_gains["Speed"],
        "final_stamina": uma_row["Stamina"] + base_gains["Stamina"],
        "final_power": uma_row["Power"] + base_gains["Power"],
        "final_guts": uma_row["Guts"] + base_gains["Guts"],
        "final_wit": uma_row["Wit"] + base_gains["Wit"],
        "acquired_skills": ["Speed Up", "Endurance", "Acceleration"],
        "mood": random.choice(["Good", "Good", "Good", "Normal", "Bad"]),
        "training_summary": "Fallback training result"
    }


# ==================== MAIN ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", type=str, required=True, choices=["gemini", "mistral"],
                    help="API to use: gemini or mistral")
    ap.add_argument("--api_key", type=str, default=None,
                    help="API key (or set GEMINI_API_KEY or MISTRAL_API_KEY env var)")
    ap.add_argument("--num_samples", type=int, default=100,
                    help="Number of training samples to generate")
    ap.add_argument("--out", type=str, default="generated_training.csv",
                    help="Output CSV file")
    ap.add_argument("--cards_per_deck", type=int, default=6,
                    help="Number of support cards per training")
    args = ap.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        env_var = f"{args.api.upper()}_API_KEY"
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"API key required. Set --api_key or {env_var} environment variable")
    
    print(f"Loading datasets...")
    data = load_datasets()
    
    print(f"Generating {args.num_samples} training samples using {args.api.upper()} API...")
    
    results = []
    skills_list = data["skills"]["skill_name"].tolist()
    
    for i in range(args.num_samples):
        print(f"\nSample {i+1}/{args.num_samples}")
        
        # Random selections
        uma_row = data["umas"].sample(1).iloc[0]
        track_row = data["tracks"].sample(1).iloc[0]
        cards_sample = data["cards"].sample(min(args.cards_per_deck, len(data["cards"])))
        
        # Generate training
        result = generate_training_sample(
            uma_row, track_row, cards_sample, skills_list, 
            args.api, api_key
        )
        
        if result:
            # Build output row matching predictive model format
            row = {
                "race_id": f"GEN{i+1:04d}",
                "uma_id": uma_row["uma_id"],
                "uma_name": uma_row["uma_name"],
                "style": uma_row["default_style"],
                "style_letter": uma_row["default_style"][0],  # Simplified
                "material": track_row["material"],
                "material_letter": "A",  # Will be computed properly
                "distance": track_row["distance"],
                "distance_letter": "A",  # Will be computed properly
                "mood": result["mood"],
                "mood_mult": 1.0 if result["mood"] == "Good" else 0.9,
                "Speed": result["final_speed"],
                "Stamina": result["final_stamina"],
                "Power": result["final_power"],
                "Guts": result["final_guts"],
                "Wit": result["final_wit"],
                "skills_offense": len([s for s in result["acquired_skills"] if "speed" in s.lower()]),
                "skills_endurance": len([s for s in result["acquired_skills"] if "endurance" in s.lower() or "stamina" in s.lower()]),
                "skills_tactics": len(result["acquired_skills"]) - 2,
                "skill_rating": len(result["acquired_skills"]) * 0.1,
                "card_ids": "|".join(cards_sample["card_id"].tolist()),
                "latent_score": random.uniform(0.5, 1.0),
                "finish_pos": random.randint(1, 18),  # Will be predicted by model
                "training_summary": result.get("training_summary", "")
            }
            results.append(row)
            print(f"  Generated: {uma_row['uma_name']} -> Speed:{result['final_speed']} Stamina:{result['final_stamina']}")
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f"\n{'='*50}")
    print(f"Generated {len(df)} training samples")
    print(f"Saved to: {args.out}")
    print(f"{'='*50}")
    print("\nSample stats:")
    print(df[["Speed", "Stamina", "Power", "Guts", "Wit"]].describe())


if __name__ == "__main__":
    main()