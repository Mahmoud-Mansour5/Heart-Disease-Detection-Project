# rules.py
import collections

if not hasattr(collections, 'Mapping'):
    import collections.abc
    collections.Mapping = collections.abc.Mapping
    

from experta import *

class PatientFact(Fact):
    """
    Features: age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal, target
    """
    pass

class HeartDiseaseExpert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.risk_score = 0
        self.risk_flags = []
        self.risk_level = None

    @Rule(PatientFact(chol=P(lambda x: x > 240), age=P(lambda x: x > 50)))
    def rule_high_chol_elderly(self):
        self.risk_score += 3
        self.risk_flags.append("[R1] Cholesterol >240 mg/dl & Age >50 -> High risk (+3)")

    @Rule(PatientFact(trestbps=P(lambda x: x > 140)))
    def rule_hypertension(self):
        self.risk_score += 2
        self.risk_flags.append("[R2] Resting BP >140 mmHg -> Hypertension (+2)")

    @Rule(PatientFact(exang=1))
    def rule_exang(self):
        self.risk_score += 3
        self.risk_flags.append("[R3] Exercise-induced angina present -> Strong indicator (+3)")

    @Rule(PatientFact(thalach=P(lambda x: x < 120)))
    def rule_low_thalach(self):
        self.risk_score += 2
        self.risk_flags.append("[R4] Max heart rate <120 bpm -> Poor cardiac performance (+2)")

    @Rule(PatientFact(oldpeak=P(lambda x: x > 2.0)))
    def rule_high_oldpeak(self):
        self.risk_score += 2
        self.risk_flags.append("[R5] ST depression (oldpeak) >2.0 -> Ischemia indicator (+2)")

    @Rule(PatientFact(cp=P(lambda x: x == 0)))
    def rule_typical_angina(self):
        self.risk_score += 2
        self.risk_flags.append("[R6] Typical angina (cp=0) -> Classic symptom (+2)")

    @Rule(PatientFact(ca=P(lambda x: x >= 2)))
    def rule_blocked_vessels(self):
        self.risk_score += 3
        self.risk_flags.append("[R7] 2+ major vessels blocked (ca>=2) -> Serious disease (+3)")

    @Rule(PatientFact(thal=P(lambda x: x == 3)))
    def rule_reversible_thal(self):
        self.risk_score += 3
        self.risk_flags.append("[R8] Reversible thal defect (thal=3) -> High risk (+3)")

    @Rule(PatientFact(fbs=1, chol=P(lambda x: x > 200)))
    def rule_diabetes_chol(self):
        self.risk_score += 2
        self.risk_flags.append("[R9] High fasting sugar & Cholesterol >200 -> Diabetic risk (+2)")

    @Rule(PatientFact(restecg=P(lambda x: x == 2)))
    def rule_abnormal_ecg(self):
        self.risk_score += 2
        self.risk_flags.append("[R10] LV hypertrophy on ECG (restecg=2) -> Cardiac risk (+2)")

    @Rule(PatientFact(sex=1, age=P(lambda x: x > 55), trestbps=P(lambda x: x > 130)))
    def rule_male_senior_htn(self):
        self.risk_score += 3
        self.risk_flags.append("[R11] Male, Age>55 & BP>130 -> Triple risk combo (+3)")

    @Rule(PatientFact(thalach=P(lambda x: x > 150), chol=P(lambda x: x < 200), exang=0, oldpeak=P(lambda x: x < 1.0)))
    def rule_healthy(self):
        self.risk_score -= 2
        self.risk_flags.append("[R12] Good HR, Low chol, No angina, Low ST dep -> Protective (-2)")

    def compute_risk_level(self):
        s = self.risk_score
        if s <= 2:
            self.risk_level = "LOW RISK"
        elif s <= 6:
            self.risk_level = "MODERATE RISK"
        else:
            self.risk_level = "HIGH RISK"
        return self.risk_level