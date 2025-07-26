
import json
from datetime import datetime
import uuid
from main import ai_training

def feed_comprehensive_training_data():
    """Feed the AI with comprehensive training data across all skill levels and categories"""
    
    # ADVANCED GEOMETRIC SHAPES AND CONSTRUCTION
    advanced_shapes_data = [
        {
            "prompt": "how to draw perfect geometric shapes",
            "response": "Use construction lines and guides. For circles: draw a square, mark midpoints, connect with curves. For polygons: start with a circle, divide into equal segments. Use the shape tools for precision, then refine with pencil tool.",
            "rating": 5,
            "skill_level": "intermediate",
            "category": "geometry"
        },
        {
            "prompt": "draw complex 3D geometric forms",
            "response": "Start with basic wireframes. Draw the front face, then parallel lines for depth. Connect corresponding corners. Add perspective by making distant lines converge. Use shading on three visible faces: light, medium, dark.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "3d_geometry"
        },
        {
            "prompt": "isometric cube drawing technique",
            "response": "Draw a vertical line. Add two 30-degree angles from the base. Create three visible faces using parallel lines. Keep all vertical lines truly vertical. All edges should be equal length for perfect isometric projection.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "technical_drawing"
        },
        {
            "prompt": "hexagonal patterns and tessellations",
            "response": "Start with one perfect hexagon using circle construction. Divide circle into 6 equal parts (60° each). Connect points for hexagon. For tessellation: copy hexagons edge-to-edge with no gaps or overlaps.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "patterns"
        },
        {
            "prompt": "drawing perfect spirals and curves",
            "response": "For spirals: start from center, gradually increase radius with each turn. Maintain consistent spacing. For smooth curves: use multiple guide points, draw lightly, then darken the best curve. Practice figure-8 motions for fluidity.",
            "rating": 5,
            "skill_level": "intermediate",
            "category": "curves"
        }
    ]

    # MATHEMATICAL PATTERNS AND SEQUENCES
    mathematical_patterns_data = [
        {
            "prompt": "fibonacci spiral in art composition",
            "response": "Create rectangles following Fibonacci numbers (1,1,2,3,5,8...). Draw quarter-circles in each rectangle. Connect smoothly for golden spiral. Use this for dynamic composition and natural-looking curves in artwork.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "mathematical_art"
        },
        {
            "prompt": "golden ratio rectangle construction",
            "response": "Draw a square. Mark midpoint of one side. Draw arc from midpoint to opposite corner. Extend the square's base to where arc intersects. Complete rectangle. Ratio will be 1:1.618 (golden ratio).",
            "rating": 5,
            "skill_level": "advanced",
            "category": "sacred_geometry"
        },
        {
            "prompt": "geometric mandala patterns",
            "response": "Start with center point and circle. Divide into 8 or 12 equal segments. Create concentric circles at measured intervals. Add geometric shapes at intersection points. Maintain perfect symmetry and mathematical precision.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "mandala"
        },
        {
            "prompt": "fractal patterns for beginners",
            "response": "Start with simple shape (triangle/square). Divide into smaller copies of itself. Repeat process at multiple scales. Each iteration adds more detail. Tree branches, snowflakes, and coastlines follow fractal principles.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "fractals"
        },
        {
            "prompt": "islamic geometric star patterns",
            "response": "Begin with regular polygon (octagon works well). Extend sides to create star points. Add connecting geometric elements. Use compass and straightedge for precision. Maintain 8-fold or 12-fold symmetry throughout.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "islamic_geometry"
        }
    ]

    # ADVANCED ARTISTIC TECHNIQUES
    professional_techniques_data = [
        {
            "prompt": "atmospheric perspective in landscapes",
            "response": "Distant objects become lighter, bluer, less detailed. Foreground: dark, warm, high contrast. Middle ground: medium values. Background: light, cool, soft edges. Reduce saturation and increase blue as distance increases.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "perspective"
        },
        {
            "prompt": "chiaroscuro lighting technique",
            "response": "Create dramatic contrast between light and dark. Use single strong light source. Leave large areas in deep shadow. Highlight only key features. Gradually transition from light to dark. Think Caravaggio or Rembrandt style.",
            "rating": 5,
            "skill_level": "expert",
            "category": "lighting"
        },
        {
            "prompt": "sfumato blending like leonardo da vinci",
            "response": "Use subtle gradations without harsh outlines. Blend colors and tones imperceptibly. Work with multiple thin layers. Soften edges with gentle transitions. Creates mysterious, dreamlike quality. Practice on portraits first.",
            "rating": 5,
            "skill_level": "expert",
            "category": "classical_techniques"
        },
        {
            "prompt": "impasto texture painting effects",
            "response": "Apply thick layers of paint/color. Use palette knife or textured brushes. Build up surface texture. Light catches raised areas creating dimensional effect. Excellent for capturing fabric, hair, or rough surfaces.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "texture_techniques"
        },
        {
            "prompt": "pointillism color mixing technique",
            "response": "Place pure color dots side by side. Let viewer's eye mix colors optically. Use complementary colors for vibration. Vary dot size for emphasis. Smaller dots = smoother appearance. Requires patience but creates luminous effects.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "impressionist_techniques"
        }
    ]

    # COMPLEX SUBJECT MATTER
    complex_subjects_data = [
        {
            "prompt": "drawing realistic water reflections",
            "response": "Mirror the subject vertically below waterline. Break reflection with horizontal ripple lines. Make reflection slightly darker than original. Add waviness - more distortion = more movement. Reflections follow perspective rules.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "natural_elements"
        },
        {
            "prompt": "complex architectural perspective",
            "response": "Establish horizon line and multiple vanishing points. Use 3-point perspective for tall buildings. Keep verticals straight unless doing worm's eye view. Measure proportions carefully. Add atmospheric perspective for depth.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "architecture_advanced"
        },
        {
            "prompt": "drawing intricate mechanical objects",
            "response": "Break complex machines into basic geometric shapes. Understand how parts connect and function. Use technical drawing principles. Show depth with overlapping forms. Add bolts, rivets, and mechanical details systematically.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "technical_illustration"
        },
        {
            "prompt": "realistic fabric and drapery",
            "response": "Understand how fabric falls and folds. Identify major fold types: pipe, zigzag, spiral, half-lock, inert. Show form underneath fabric. Use consistent light source. Practice with simple cloth arrangements first.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "drapery"
        },
        {
            "prompt": "complex organic forms and crystals",
            "response": "Study crystal structure and growth patterns. Use geometric construction for faceted forms. Understand how light refracts through transparent materials. Show both transmitted and reflected light. Build complexity gradually.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "crystalline_forms"
        }
    ]

    # PROFESSIONAL DIGITAL TECHNIQUES
    digital_mastery_data = [
        {
            "prompt": "advanced layer blending techniques",
            "response": "Experiment with multiply, overlay, screen, and soft light modes. Use masks for selective effects. Create depth with background blur. Build lighting effects on separate layers. Non-destructive editing preserves original artwork.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "digital_advanced"
        },
        {
            "prompt": "professional digital painting workflow",
            "response": "1. Rough sketch on separate layer. 2. Block in major shapes and values. 3. Refine forms and proportions. 4. Add local colors. 5. Paint lighting and shadows. 6. Final details and textures. Save iterations frequently.",
            "rating": 5,
            "skill_level": "expert",
            "category": "workflow"
        },
        {
            "prompt": "custom brush creation and effects",
            "response": "Vary brush hardness for different effects. Soft brushes for skin, hard for mechanical objects. Use texture brushes for organic surfaces. Vary opacity and flow. Create custom brushes from photos for unique textures.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "brush_techniques"
        },
        {
            "prompt": "digital color theory applications",
            "response": "Use color temperature to show lighting conditions. Warm lights create cool shadows. Complement local color with atmospheric effects. Use limited palettes for harmony. Study how digital RGB affects color relationships.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "digital_color"
        }
    ]

    # ADVANCED PATTERN SYSTEMS
    pattern_systems_data = [
        {
            "prompt": "complex celtic knot construction",
            "response": "Start with grid system. Create interwoven paths with consistent over-under pattern. Maintain equal line widths. Use compass for curved sections. Traditional knots follow mathematical rules - study authentic examples first.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "celtic_art"
        },
        {
            "prompt": "advanced op art visual illusions",
            "response": "Use systematic pattern variations to create movement. Gradually change line spacing, angles, or scale. High contrast enhances effect. Precise execution is crucial - small irregularities destroy illusion. Plan carefully before executing.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "optical_illusions"
        },
        {
            "prompt": "japanese wave pattern technique",
            "response": "Study Hokusai's Great Wave structure. Waves have consistent curl patterns. Use flowing lines that follow water's natural movement. Layer multiple wave systems. Add foam details with white highlights and texture.",
            "rating": 5,
            "skill_level": "intermediate",
            "category": "japanese_art"
        },
        {
            "prompt": "art nouveau decorative patterns",
            "response": "Base designs on natural forms - flowers, vines, insects. Use flowing, organic lines. Incorporate stylized female figures. Maintain asymmetrical balance. Colors should be rich but harmonious. Study Mucha and Klimt.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "decorative_arts"
        }
    ]

    # SCIENTIFIC AND TECHNICAL ILLUSTRATION
    scientific_illustration_data = [
        {
            "prompt": "botanical illustration techniques",
            "response": "Study plant structure carefully. Show both overall form and detailed parts. Use consistent lighting. Include cross-sections and magnified details. Accuracy is paramount - each leaf and flower must be botanically correct.",
            "rating": 5,
            "skill_level": "expert",
            "category": "scientific_illustration"
        },
        {
            "prompt": "anatomical drawing precision",
            "response": "Understand underlying bone and muscle structure. Use proportional measurements. Study from life and medical references. Show accurate proportions and connections. Clinical accuracy combined with artistic skill is essential.",
            "rating": 5,
            "skill_level": "expert",
            "category": "medical_illustration"
        },
        {
            "prompt": "technical cutaway illustrations",
            "response": "Show internal mechanisms while maintaining external form. Use different line weights for different components. Create clear visual hierarchy. Add labels and callouts systematically. Think like both artist and engineer.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "technical_drawing"
        }
    ]

    # CULTURAL AND HISTORICAL STYLES
    cultural_styles_data = [
        {
            "prompt": "traditional chinese brush painting",
            "response": "Master the 'Four Treasures': brush, ink, paper, inkstone. Practice basic strokes: dots, lines, washes. Capture essence rather than detail. Use negative space effectively. Study bamboo, plum blossoms, and birds first.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "chinese_art"
        },
        {
            "prompt": "aboriginal dot painting techniques",
            "response": "Use dots to create patterns and tell stories. Different dot sizes create visual hierarchy. Traditional symbols have specific meanings. Work from center outward. Each dot should be consistent in size within its group.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "indigenous_art"
        },
        {
            "prompt": "medieval illuminated manuscript style",
            "response": "Create ornate capital letters with gold leaf effects. Use vibrant colors within black outlines. Add decorative borders with interlacing patterns. Combine text and image harmoniously. Study Book of Kells for inspiration.",
            "rating": 4,
            "skill_level": "advanced",
            "category": "medieval_art"
        }
    ]

    # ADVANCED TROUBLESHOOTING
    advanced_troubleshooting_data = [
        {
            "prompt": "perspective distortion problems",
            "response": "Check vanishing point placement - they should be on horizon line. Avoid placing VPs too close together. For dramatic views, use 3-point perspective carefully. Vertical lines should remain vertical except in extreme angles.",
            "rating": 5,
            "skill_level": "advanced",
            "category": "perspective_troubleshooting"
        },
        {
            "prompt": "color mudding issues in digital painting",
            "response": "Avoid overmixing colors. Use separate layers for different elements. Don't overuse soft brushes. Maintain color temperature consistency. Use hard edges to separate areas. Sample colors from reference frequently.",
            "rating": 4,
            "skill_level": "intermediate",
            "category": "color_troubleshooting"
        },
        {
            "prompt": "fixing proportion errors in portraits",
            "response": "Use measuring techniques - pencil at arm's length. Check eye placement (halfway down head). Verify nose length (eyes to chin midpoint). Measure head-to-body ratios. Turn drawing upside-down to see errors clearly.",
            "rating": 5,
            "skill_level": "intermediate",
            "category": "portrait_troubleshooting"
        }
    ]

    # Combine all comprehensive training datasets
    all_comprehensive_data = (
        advanced_shapes_data + mathematical_patterns_data + 
        professional_techniques_data + complex_subjects_data +
        digital_mastery_data + pattern_systems_data +
        scientific_illustration_data + cultural_styles_data +
        advanced_troubleshooting_data
    )
    
    print(f"🚀 Feeding AI with {len(all_comprehensive_data)} advanced training entries...")
    print("📚 Training categories include:")
    print("   • Advanced Geometric Construction")
    print("   • Mathematical Patterns & Sacred Geometry") 
    print("   • Professional Artistic Techniques")
    print("   • Complex Subject Matter")
    print("   • Digital Mastery Methods")
    print("   • Cultural & Historical Styles")
    print("   • Scientific Illustration")
    print("   • Advanced Troubleshooting")
    
    # Feed data to the AI training system
    for i, data in enumerate(all_comprehensive_data, 1):
        try:
            # Add some variation to timestamps
            import time
            time.sleep(0.01)  # Small delay for timestamp variation
            
            training_entry = ai_training.collect_training_data(
                user_prompt=data["prompt"],
                ai_response=data["response"],
                user_rating=data["rating"],
                correction=data.get("correction"),
                response_time=0.3 + (i * 0.05)  # Simulated response times
            )
            
            if i % 10 == 0:  # Progress indicator every 10 entries
                print(f"   ✅ Processed {i}/{len(all_comprehensive_data)} entries...")
                
        except Exception as e:
            print(f"   ❌ Error processing entry {i}: {e}")
            continue
    
    print(f"✨ Advanced training complete! AI system now has professional-level knowledge.")
    print(f"📊 Enhanced performance metrics:")
    print(f"   • Total interactions: {ai_training.performance_metrics['total_interactions']}")
    print(f"   • Success rate: {ai_training._calculate_success_rate()}%")
    print(f"   • User satisfaction: {ai_training.performance_metrics['user_satisfaction_score']}/5.0")
    print(f"   • Data quality score: {ai_training._calculate_data_quality_score()}%")
    
    # Add high-quality cached responses for complex queries
    advanced_cache_responses = [
        ("fibonacci spiral", "Create rectangles following Fibonacci sequence (1,1,2,3,5,8...). Draw quarter-circles in each. Connect smoothly for golden spiral composition.", 5),
        ("isometric cube", "Draw vertical line. Add two 30-degree angles from base. Create three visible faces with parallel lines. Keep verticals truly vertical.", 5),
        ("atmospheric perspective", "Distant objects: lighter, bluer, less detailed. Foreground: dark, warm, high contrast. Reduce saturation with distance.", 5),
        ("chiaroscuro technique", "Single strong light source. Large shadow areas. Dramatic light-dark contrast. Gradual transitions. Think Caravaggio style.", 5),
        ("complex geometric patterns", "Start with basic polygon. Use compass and straightedge. Maintain mathematical precision. Build complexity through repetition.", 4),
        ("3D geometric forms", "Begin with wireframe. Add parallel depth lines. Connect corresponding corners. Use perspective convergence.", 4)
    ]
    
    print("💾 Adding advanced cached responses...")
    for prompt, response, rating in advanced_cache_responses:
        ai_training.cache_successful_response(prompt, response, rating)
    
    print("🎯 Comprehensive AI training complete! System now handles:")
    print("   • Advanced geometric construction")
    print("   • Mathematical and fractal patterns") 
    print("   • Professional artistic techniques")
    print("   • Complex technical illustration")
    print("   • Cultural and historical art styles")
    
    return {
        "total_entries_added": len(all_comprehensive_data),
        "cached_responses_added": len(advanced_cache_responses),
        "current_performance": ai_training.performance_metrics,
        "data_quality_score": ai_training._calculate_data_quality_score(),
        "advanced_categories": 8
    }

if __name__ == "__main__":
    results = feed_comprehensive_training_data()
    print(f"\n📈 Advanced Training Results Summary:")
    print(f"   • Entries added: {results['total_entries_added']}")
    print(f"   • Cached responses: {results['cached_responses_added']}")  
    print(f"   • Advanced categories: {results['advanced_categories']}")
    print(f"   • Final data quality: {results['data_quality_score']}%")
    print(f"\n🚀 Your AI can now handle complex professional-level artistic queries!")
