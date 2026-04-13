import warnings
import warnings
import sys
import os
# Suppress any harmless warnings from the experta library
warnings.filterwarnings('ignore')
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# 1. Import Facts and Rules from rules.py 
# (This safely handles the collections.Mapping fix automatically!)
from rule_based_system.rules import PatientFact, HeartDiseaseExpert
# 2. Enhanced Class for UI Integration
class HeartDiseaseExpertSystem(HeartDiseaseExpert):
    def __init__(self):
        super().__init__()
        # This variable holds the direct string (e.g., "HIGH RISK") for the UI
        self.risk_assessment = "N/A"
        
        # This dictionary holds the detailed breakdown
        self.risk_details = {
            "score": 0,
            "level": "N/A",
            "reasons": [],
            "status": "Ready"
        }

    def declare_facts(self, patient_data: dict):
        """
        Receives data from the UI, runs the expert engine, and calculates risk.
        """
        # Reset engine and scores for a fresh run
        self.reset()
        self.risk_score = 0
        self.risk_flags = []
        
        # Declare the patient data as a fact in the Knowledge Base
        self.declare(PatientFact(**patient_data))
        
        # Run the inference mechanism (Fire the rules)
        self.run()
        
        # Calculate final level based on the rules.py logic
        calculated_level = self.compute_risk_level()
        
        # Update variables for UI access
        self.risk_assessment = calculated_level
        self.risk_details = {
            "score": self.risk_score,
            "level": calculated_level,
            "reasons": self.risk_flags,
            "status": "Completed"
        }
        
        return self.risk_assessment

# 3. Terminal Test (To verify logic before running UI)
if __name__ == "__main__":
    # Simulate UI data entry for a high-risk patient
    sample_data = {
        "age": 60, "sex": 1, "cp": 0, "trestbps": 150, "chol": 260,
        "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1,
        "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 3
    }

    print("--> Initializing Expert System Bridge...")
    engine = HeartDiseaseExpertSystem()
    
    # Run inference
    final_result = engine.declare_facts(sample_data)
    
    print("\n" + "="*50)
    print(f"🧠 EXPERT SYSTEM DIAGNOSIS: {final_result}")
    print("="*50)
    print(f"Total Risk Score: {engine.risk_details['score']} points")
    print("Triggered Medical Rules:")
    for rule in engine.risk_details['reasons']:
        print(f" ✔ {rule}")