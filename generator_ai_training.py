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
# Updated Gemini API endpoint (correct as of Dec 2024)
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# ==================== TRAINING SIMULATION PROMPT ====================
TRAINING_PROMPT_TEMPLATE = """You are an expert Uma Musume training simulator. You will simulate a complete 3-year training journey for a horsegirl.

# GAME MECHANICS OVERVIEW:
The training system in Umamusume: Pretty Derby involves preparing a horsegirl over a limited number of days to compete in races and ultimately qualify for the URA Finale.

## LEGACY UMAMUSUME (Inheritance):
- Two Legacy Umamusume are selected at the start
- They grant stat boosts and skill hints based on their own stats and aptitudes
- Choose legacies that align with the trainee's target strengths (e.g., Sprint legacy for Sprint racer)
- Veterans with excellent Speed, Power, Stamina, and A rank in Mile/Front are generally safe choices

## SUPPORT CARDS (6-card deck):
- 5 cards from gacha + 1 borrowed card
- Provide bonuses: better starting stats, more fans, better mood, extra race stats
- Shape skill development through skill hints during training
- SSR/SR cards with relevant stats are preferred
- Card level matters significantly for effectiveness

## TRAINING SESSIONS:
Each training session:
- Uses 1 day and drains energy
- Focuses on one stat: Speed, Stamina, Power, Guts, or Wit
- Gains bonus when Support card friends join (visible in top right)
- Early training priority: Speed, Stamina, Wit + sessions with most Support friends

## ENERGY & REST:
- Low energy reduces training effectiveness
- Use Rest option to recover energy
- Pushing too hard risks injury

## MOOD SYSTEM:
- Mood affects performance and stat gains significantly
- Bad mood = reduced gains and poor race performance
- Improve mood with recreational outings (costs 1 day)
- Injury worsens mood → visit Infirmary before outings
- Strategic conversations help maintain good mood

## STAT TARGET RANGES (for URA Finale):
- Speed: ~900 points (determines running speed)
- Stamina: 300-800 (depends on race distance - Long needs more)
- Power: 460-600 (affects acceleration)
- Guts: 200-350 (willpower to hang on late in race)
- Wit: 320-400 (focus and smart decisions mid-race)

## SKILLS:
- Unlocked using Skill Points throughout training
- Prioritize Speed and Stamina boosting skills
- Skills with Hint discounts are especially valuable
- Skill hints come from Support cards in deck

## RACE REQUIREMENTS:
- Must meet fan count requirements
- Enter extra races through Races menu to gain fans
- Win goal races to progress toward URA Finale

# TRAINING CONSTRAINTS:
- Training typically adds 200-600 points per stat from base values
- Speed and Stamina are most crucial
- Skills should match horse's aptitudes and racing style
- Support card specialties heavily influence which stats grow faster
- Good mood throughout training = higher final stats
- Balanced training across all 5 stats is important (don't neglect Guts/Wit)

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
Simulate a complete 3-year training journey considering all the mechanics above, and return ONLY a valid JSON object with this exact structure:
{{
  "final_speed": <number>,
  "final_stamina": <number>,
  "final_power": <number>,
  "final_guts": <number>,
  "final_wit": <number>,
  "acquired_skills": ["skill1", "skill2", "skill3", "skill4"],
  "mood": "<Good|Normal|Bad>",
  "training_summary": "<brief 2-3 sentence narrative of the training journey>",
  "injuries_count": <0-3>,
  "rest_days_used": <number>
}}

Requirements for final stats (base + training gains):
- final_speed: aim for ~900 total (base + 200-600 gain, boosted by growth_spd_pct% and Speed support cards)
- final_stamina: 300-800 total depending on distance preference (base + 200-600, boosted by growth_sta_pct% and Stamina cards)
  * Long distance needs 600-800 stamina
  * Medium distance needs 450-650 stamina  
  * Mile/Sprint needs 300-500 stamina
- final_power: 460-600 total (base + 200-400, boosted by growth_pow_pct% and Power cards)
- final_guts: 200-350 total (base + 100-300, boosted by growth_gut_pct% and Guts cards)
- final_wit: 320-400 total (base + 150-350, boosted by growth_wit_pct% and Wit cards)

Skills selection (choose 4-6 appropriate skills):
- Prioritize Speed and Stamina boosting skills
- Match skills to horse's aptitudes (e.g., Sprint skills for pref_sprint=A)
- Match skills to racing style (Front/Pace/Late/End)
- Consider skills that the Support cards would provide hints for
- Include skills with strategic value (hint discounts, recovery, acceleration)

Mood determination:
- Good (70% probability): well-managed training, few injuries, good card synergy
- Normal (20% probability): average training with some setbacks
- Bad (10% probability): multiple injuries or poor energy management

Training narrative:
- Describe key moments: early training focus, injuries/setbacks, breakthrough moments
- Mention which Support cards were most helpful
- Note if legacy inheritance was particularly beneficial

Realism factors:
- Support cards specializing in Speed → higher Speed gains
- Support cards specializing in Stamina → higher Stamina gains (etc.)
- Distance preference affects stamina needs (Long distance = more stamina training)
- Style preference (Front/Pace/Late/End) affects Power and Wit balance
- Good mood throughout = higher overall stat totals
- Injuries (1-3 during training) reduce final stats slightly

Return ONLY the JSON object, no markdown formatting, no additional text."""


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
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        # Print detailed error message
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", str(e))
            print(f"    Gemini API Error: {error_msg}")
        except:
            print(f"    HTTP Error: {e}")
        raise
    except Exception as e:
        print(f"    Request Error: {e}")
        raise


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
    
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        # Print detailed error message
        try:
            error_data = response.json()
            error_msg = error_data.get("message", str(e))
            print(f"    Mistral API Error: {error_msg}")
        except:
            print(f"    HTTP Error: {e}")
        raise
    except Exception as e:
        print(f"    Request Error: {e}")
        raise


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
    
    # Prepare card names and specialties (coerce to strings, handle missing values)
    card_names = ", ".join(cards_sample["name"].astype(str).tolist())
    # Ensure specialties are strings and remove empty/null entries
    specialties = cards_sample["stat_specialty"].fillna("").astype(str).tolist()
    specialties = [s for s in specialties if s and s.lower() != "nan"]
    card_specialties = ", ".join(specialties) if specialties else "None"

    # Determine distance preference for stamina calculation
    distance_prefs = {
        "pref_long": "Long",
        "pref_medium": "Medium", 
        "pref_mile": "Mile",
        "pref_sprint": "Sprint"
    }
    best_distance = "Medium"
    best_letter = "E"
    for col, dist_name in distance_prefs.items():
        letter = uma_row[col]
        if letter in ["A", "B", "S"]:
            if letter > best_letter or best_letter == "E":
                best_distance = dist_name
                best_letter = letter
    
    # Filter skills by rarity (prefer higher quality)
    skills_df = pd.DataFrame({"skill_name": skills_list})
    skills_sample = random.sample(skills_list, min(25, len(skills_list)))
    skills_str = ", ".join(skills_sample)
    
    # Format prompt with enhanced information
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
        cards=f"{card_names} (Specialties: {card_specialties})",
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
            
            # Add default values for new fields if missing
            if "injuries_count" not in result:
                result["injuries_count"] = random.randint(0, 2)
            if "rest_days_used" not in result:
                result["rest_days_used"] = random.randint(5, 20)
            
            return result
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                # Fallback: generate synthetic data
                print("  Using fallback synthetic generation")
                return generate_fallback_training(uma_row, cards_sample, best_distance)
    
    return None


def generate_fallback_training(uma_row, cards_sample, distance_pref):
    """Fallback synthetic training when API fails"""
    
    # Determine stamina needs based on distance preference
    stamina_targets = {
        "Long": (600, 800),
        "Medium": (450, 650),
        "Mile": (350, 500),
        "Sprint": (300, 450)
    }
    stamina_range = stamina_targets.get(distance_pref, (400, 600))
    
    # Base gains with some randomness
    speed_gain = random.randint(250, 550) + int(uma_row["growth_spd_pct"] * 5)
    stamina_gain = random.randint(200, 500) + int(uma_row["growth_sta_pct"] * 5)
    power_gain = random.randint(200, 400) + int(uma_row["growth_pow_pct"] * 3)
    guts_gain = random.randint(100, 300) + int(uma_row["growth_gut_pct"] * 3)
    wit_gain = random.randint(150, 350) + int(uma_row["growth_wit_pct"] * 3)
    
    # Apply support card bonuses (simplified)
    card_specialties = cards_sample["stat_specialty"].value_counts()
    if "Speed" in card_specialties.index:
        speed_gain += card_specialties["Speed"] * 30
    if "Stamina" in card_specialties.index:
        stamina_gain += card_specialties["Stamina"] * 30
    if "Power" in card_specialties.index:
        power_gain += card_specialties["Power"] * 20
    
    # Adjust stamina to target range
    target_stamina = random.randint(*stamina_range)
    stamina_gain = max(50, target_stamina - uma_row["Stamina"])
    
    # Mood probability
    mood = random.choices(
        ["Good", "Normal", "Bad"],
        weights=[0.7, 0.2, 0.1]
    )[0]
    
    # Reduce gains if bad mood
    if mood == "Bad":
        speed_gain = int(speed_gain * 0.85)
        stamina_gain = int(stamina_gain * 0.85)
        power_gain = int(power_gain * 0.85)
    
    # Generate appropriate skills based on style and aptitudes
    default_skills = []
    style = uma_row["default_style"]
    if style == "Front":
        default_skills = ["Lead Start", "Quick Acceleration", "Top Speed Boost"]
    elif style == "Pace":
        default_skills = ["Steady Pace", "Mid-Race Surge", "Tactical Positioning"]
    elif style == "Late":
        default_skills = ["Late Charge", "Final Sprint", "Closing Kick"]
    elif style == "End":
        default_skills = ["Last Stand", "Endurance Master", "Final Push"]
    
    default_skills.append("Stamina Conservation")
    
    return {
        "final_speed": uma_row["Speed"] + speed_gain,
        "final_stamina": uma_row["Stamina"] + stamina_gain,
        "final_power": uma_row["Power"] + power_gain,
        "final_guts": uma_row["Guts"] + guts_gain,
        "final_wit": uma_row["Wit"] + wit_gain,
        "acquired_skills": default_skills[:4],
        "mood": mood,
        "training_summary": f"Fallback training for {style} style racer with focus on {distance_pref} distances.",
        "injuries_count": random.randint(0, 2),
        "rest_days_used": random.randint(5, 15)
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
                "style_letter": uma_row["default_style"][0],
                "material": track_row["material"],
                "material_letter": uma_row[f"pref_{'turf' if track_row['material']=='Turf' else 'dirt'}"],
                "distance": track_row["distance"],
                "distance_letter": uma_row[f"pref_{track_row['distance'].lower()}"],
                "mood": result["mood"],
                "mood_mult": 1.0 if result["mood"] == "Good" else (0.95 if result["mood"] == "Normal" else 0.85),
                "Speed": int(result["final_speed"]),
                "Stamina": int(result["final_stamina"]),
                "Power": int(result["final_power"]),
                "Guts": int(result["final_guts"]),
                "Wit": int(result["final_wit"]),
                "skills_offense": len([s for s in result["acquired_skills"] if any(word in s.lower() for word in ["speed", "acceleration", "surge"])]),
                "skills_endurance": len([s for s in result["acquired_skills"] if any(word in s.lower() for word in ["stamina", "endurance", "recovery"])]),
                "skills_tactics": len([s for s in result["acquired_skills"] if any(word in s.lower() for word in ["position", "tactical", "focus", "wit"])]),
                "skill_rating": round(len(result["acquired_skills"]) * 0.08, 2),
                "card_ids": "|".join(cards_sample["card_id"].astype(str).tolist()),
                "latent_score": round(random.uniform(0.6, 0.95), 3),
                "finish_pos": random.randint(1, 18),  # Will be predicted by model
                "finish_time": round(random.uniform(90.0, 140.0), 2),  # Placeholder
                "training_summary": result.get("training_summary", ""),
                "injuries_count": result.get("injuries_count", 0),
                "rest_days_used": result.get("rest_days_used", 10)
            }
            results.append(row)
            print(f"  Generated: {uma_row['uma_name']} -> Speed:{result['final_speed']} Stamina:{result['final_stamina']} (Mood: {result['mood']})")
        
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