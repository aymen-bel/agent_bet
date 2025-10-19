# advanced_predictor.py
import pandas as pd
import numpy as np
from scipy.stats import poisson

class AdvancedMatchPredictor:
    def __init__(self, team_assessment_data):
        self.team_data = team_assessment_data
        
    def predict_match(self, home_team, away_team, venue="home"):
        """التنبؤ باستخدام توزيع بواسون المتقدم"""
        if home_team not in self.team_data or away_team not in self.team_data:
            return None
            
        home_metrics = self.team_data[home_team]
        away_metrics = self.team_data[away_team]
        
        # حساب القوة الهجومية والدفاعية
        home_attack, home_defense = self._calculate_team_strength(home_metrics, 'home')
        away_attack, away_defense = self._calculate_team_strength(away_metrics, 'away')
        
        # تطبيق تأثير الملعب
        if venue == "home":
            home_attack *= 1.2
            away_attack *= 0.8
        else:
            home_attack *= 0.9
            away_attack *= 1.1
        
        # حساب lambda لتوزيع بواسون
        home_lambda = home_attack * away_defense * 1.4  # متوسط أهداف الدوري
        away_lambda = away_attack * home_defense * 1.4
        
        # التنبؤ بالنتيجة باستخدام بواسون
        score_prediction = self._poisson_prediction(home_lambda, away_lambda)
        
        # حساب الاحتمالات
        probabilities = self._calculate_probabilities(home_lambda, away_lambda)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'score_prediction': score_prediction,
            'probabilities': probabilities,
            'expected_goals': {'home': home_lambda, 'away': away_lambda}
        }
    
    def _calculate_team_strength(self, metrics, venue):
        """حساب قوة الفريق الهجومية والدفاعية"""
        # قوة هجومية تعتمد على كفاءة التسجيل والضغط الهجومي
        attack_strength = (
            metrics.get('goals_per_match', 1.2) * 0.4 +
            metrics.get('shot_efficiency', 0.12) * 3 * 0.3 +
            metrics.get('conversion_rate', 0.15) * 2 * 0.3
        )
        
        # قوة دفاعية تعتمد على كفاءة الدفاع ونظافة الشباك
        defense_strength = (
            (2 - metrics.get('goals_conceded_per_match', 1.2)) * 0.4 +
            metrics.get('defensive_efficiency', 0.7) * 0.4 +
            metrics.get('clean_sheet_rate', 0.2) * 2 * 0.2
        )
        
        # تعديل حسب الأداء في المنزل/الخارج
        if venue == 'home':
            attack_strength *= metrics.get('home_strength', 1.0)
            defense_strength *= metrics.get('home_strength', 1.0)
        else:
            attack_strength *= metrics.get('away_strength', 0.8)
            defense_strength *= metrics.get('away_strength', 0.8)
        
        return max(0.3, min(2.0, attack_strength)), max(0.3, min(2.0, defense_strength))
    
    def _poisson_prediction(self, home_lambda, away_lambda):
        """التنبؤ بالنتيجة باستخدام توزيع بواسون"""
        # محاكاة 10000 مرة للحصول على تنبؤ دقيق
        simulations = 10000
        home_goals = np.random.poisson(home_lambda, simulations)
        away_goals = np.random.poisson(away_lambda, simulations)
        
        # أخذ المتوسط والمنوال
        avg_home = np.mean(home_goals)
        avg_away = np.mean(away_goals)
        
        # التقريب لنتيجة واقعية
        predicted_home = int(round(avg_home))
        predicted_away = int(round(avg_away))
        
        return {
            'home_goals': max(0, predicted_home),
            'away_goals': max(0, predicted_away),
            'home_goals_float': avg_home,
            'away_goals_float': avg_away
        }
    
    def _calculate_probabilities(self, home_lambda, away_lambda):
        """حساب احتمالات النتائج المختلفة"""
        max_goals = 6  # الحد الأقصى للأهداف في الحساب
        
        # حساب احتمالات جميع النتائج المحتملة
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_lambda) * poisson.pmf(j, away_lambda)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # احتمالات إضافية
        both_teams_score = 1 - (poisson.pmf(0, home_lambda) + poisson.pmf(0, away_lambda) - 
                               poisson.pmf(0, home_lambda) * poisson.pmf(0, away_lambda))
        
        over_2_5 = 1 - sum(poisson.pmf(i, home_lambda + away_lambda) for i in range(3))
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'both_teams_score': both_teams_score,
            'over_2_5': over_2_5
        }