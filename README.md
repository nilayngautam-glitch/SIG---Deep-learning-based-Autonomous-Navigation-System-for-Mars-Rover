# SIG---Deep-learning-based-Autonomous-Navigation-System-for-Mars-Rover

This project presents an AI-based autonomous navigation system for a Mars rover.
The goal is to enable the rover to analyze terrain, plan safe paths, and make decisions without human intervention.

# How it works

1. Terrain Understanding (AI) :
    A MobileNetV2 CNN model analyzes Mars images
    Classifies terrain into categories like:
     - Bedrock
     - Rocks
     - Sand
     - Pebbles
       
2. Safety & Navigability Analysis :
  Each terrain is assigned:
   - Safety score → how risky it is
   - Navigability score → how easy it is to traverse
     
3. Path Planning :
   - A combined cost map is created using safety + navigability
   - A* algorithm finds the safest and easiest path
     
4. Path Optimization :
    The path is smoothed for realistic rover movement
   
5. Autonomous Decision Making :
    Rover looks ahead on the path and decides:
     - Move forward
     - Slow down
     - Reroute
     - Stop
       
6. Explainability (Grad-CAM) :
    - Shows which parts of the image influenced the AI’s decision
    - Makes the system transparent and trustworthy
