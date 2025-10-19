# realistic_predictor.py
import pandas as pd
import numpy as np
from scipy.stats import poisson, skewnorm
import random
from datetime import datetime, timedelta

class RealisticMatchPredictor:
    def __init__(self, team_assessment_data):
        self.team_data = team_assessment_data
        self.prediction_history = {}
        
    def predict_match(self, home_team, away_team, venue="home", external_factors=None):
        """تنبؤ واقعي محسن مع خوارزمية ثقة محسنة وعوامل إضافية"""
        if home_team not in self.team_data or away_team not in self.team_data:
            return None
            
        home_metrics = self.team_data[home_team]
        away_metrics = self.team_data[away_team]
        
        # تطبيق العوامل الخارجية
        factors = self._process_external_factors(external_factors, home_team, away_team)
        
        # حساب القوة الهجومية والدفاعية المحسنة
        home_attack = self._calculate_enhanced_attack_strength(home_metrics, factors['home_attack_modifier'])
        home_defense = self._calculate_enhanced_defense_strength(home_metrics, factors['home_defense_modifier'])
        away_attack = self._calculate_enhanced_attack_strength(away_metrics, factors['away_attack_modifier'])
        away_defense = self._calculate_enhanced_defense_strength(away_metrics, factors['away_defense_modifier'])
        
        # تطبيق تأثير الملعب بشكل واقعي مع عوامل إضافية
        venue_impact = self._calculate_venue_impact(venue, factors)
        home_attack *= venue_impact['home_attack']
        away_attack *= venue_impact['away_attack']
        home_defense *= venue_impact['home_defense']
        away_defense *= venue_impact['away_defense']
        
        # حساب متوسط الأهداف المتوقع مع عوامل التوزيع
        league_avg_goals = 1.4
        
        home_expected = home_attack * away_defense * league_avg_goals
        away_expected = away_attack * home_defense * league_avg_goals
        
        # تطبيق عوامل التوزيع غير الطبيعي للأهداف
        home_expected = self._apply_goal_distribution(home_expected, home_metrics, 'attack')
        away_expected = self._apply_goal_distribution(away_expected, away_metrics, 'attack')
        
        # ضمان قيم واقعية
        home_expected = max(0.1, min(4.0, home_expected))
        away_expected = max(0.1, min(3.5, away_expected))
        
        # إنشاء تنبؤات متعددة
        multiple_predictions = self._generate_multiple_predictions(
            home_expected, away_expected, home_team, away_team
        )
        
        # حساب الاحتمالات المحسنة
        enhanced_probabilities = self._calculate_enhanced_probabilities(
            home_expected, away_expected, factors
        )
        
        # حساب الثقة المحسنة
        confidence_metrics = self._calculate_enhanced_confidence(
            home_metrics, away_metrics, factors, home_expected, away_expected
        )
        
        # تخزين التاريخ للتحقق من الدقة لاحقاً
        match_id = f"{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.prediction_history[match_id] = {
            'home_team': home_team,
            'away_team': away_team,
            'predictions': multiple_predictions,
            'expected_goals': {'home': home_expected, 'away': away_expected},
            'confidence': confidence_metrics,
            'factors': factors
        }
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'multiple_predictions': multiple_predictions,
            'probabilities': enhanced_probabilities,
            'expected_goals': {'home': home_expected, 'away': away_expected},
            'confidence_metrics': confidence_metrics,
            'external_factors': factors,
            'recommendations': self._generate_recommendations(
                enhanced_probabilities, multiple_predictions, confidence_metrics
            )
        }
    
    def _process_external_factors(self, external_factors, home_team, away_team):
        """معالجة العوامل الخارجية مثل الإصابات والمحفزات"""
        default_factors = {
            'home_injuries': 0,        # عدد الإصابات المهمة (0-5)
            'away_injuries': 0,
            'home_motivation': 1.0,    # عامل التحفيز (0.8-1.2)
            'away_motivation': 1.0,
            'home_fatigue': 1.0,       # التعب من المباريات السابقة (0.9-1.1)
            'away_fatigue': 1.0,
            'home_importance': 1.0,    # أهمية المباراة (0.9-1.1)
            'away_importance': 1.0,
            'weather_impact': 1.0,     # تأثير الطقس (0.95-1.05)
        }
        
        if external_factors:
            default_factors.update(external_factors)
        
        factors = default_factors
        
        # حساب معدلات التعديل بناءً على العوامل
        home_attack_modifier = (
            factors['home_motivation'] * 
            factors['home_importance'] * 
            (1.0 - factors['home_injuries'] * 0.08) *  # كل إصابة مهمة تخفض 8%
            factors['home_fatigue'] *
            factors['weather_impact']
        )
        
        away_attack_modifier = (
            factors['away_motivation'] * 
            factors['away_importance'] * 
            (1.0 - factors['away_injuries'] * 0.08) *
            factors['away_fatigue'] *
            factors['weather_impact']
        )
        
        home_defense_modifier = (
            factors['home_motivation'] * 
            (1.0 - factors['home_injuries'] * 0.06) *  # إصابات الدفاع أقل تأثيراً
            factors['home_fatigue']
        )
        
        away_defense_modifier = (
            factors['away_motivation'] * 
            (1.0 - factors['away_injuries'] * 0.06) *
            factors['away_fatigue']
        )
        
        return {
            'home_attack_modifier': max(0.5, min(1.5, home_attack_modifier)),
            'away_attack_modifier': max(0.5, min(1.5, away_attack_modifier)),
            'home_defense_modifier': max(0.5, min(1.5, home_defense_modifier)),
            'away_defense_modifier': max(0.5, min(1.5, away_defense_modifier)),
            'raw_factors': factors
        }
    
    def _calculate_enhanced_attack_strength(self, metrics, modifier=1.0):
        """حساب القوة الهجومية المحسنة"""
        base_attack = (
            metrics.get('goals_per_match', 1.2) * 0.5 +
            metrics.get('shot_efficiency', 0.1) * 5 * 0.2 +
            metrics.get('conversion_rate', 0.12) * 4 * 0.15 +
            metrics.get('attacking_consistency', 0.7) * 0.15
        )
        return max(0.3, min(2.0, base_attack * modifier))
    
    def _calculate_enhanced_defense_strength(self, metrics, modifier=1.0):
        """حساب القوة الدفاعية المحسنة"""
        base_defense = (
            (2 - metrics.get('goals_conceded_per_match', 1.2)) * 0.4 +
            metrics.get('defensive_efficiency', 0.65) * 0.25 +
            metrics.get('clean_sheet_rate', 0.2) * 2 * 0.2 +
            metrics.get('defensive_consistency', 0.7) * 0.15
        )
        return max(0.3, min(2.0, base_defense * modifier))
    
    def _calculate_venue_impact(self, venue, factors):
        """حساب تأثير الملعب مع العوامل الإضافية"""
        base_impact = {
            'home': {
                'home_attack': 1.15, 'home_defense': 1.05,
                'away_attack': 0.85, 'away_defense': 0.95
            },
            'away': {
                'home_attack': 0.9, 'home_defense': 0.95,
                'away_attack': 1.1, 'away_defense': 1.05
            },
            'neutral': {
                'home_attack': 1.0, 'home_defense': 1.0,
                'away_attack': 1.0, 'away_defense': 1.0
            }
        }
        
        impact = base_impact.get(venue, base_impact['home'])
        
        # تطبيق عوامل إضافية على تأثير الملعب
        impact['home_attack'] *= factors['home_attack_modifier']
        impact['away_attack'] *= factors['away_attack_modifier']
        impact['home_defense'] *= factors['home_defense_modifier']
        impact['away_defense'] *= factors['away_defense_modifier']
        
        return impact
    
    def _apply_goal_distribution(self, expected_goals, metrics, attack_type):
        """تطبيق توزيع غير طبيعي للأهداف (ليس بواسون بحت)"""
        # استخدام توزيع منحرف لتمثيل واقعية توزيع الأهداف
        consistency = metrics.get('attacking_consistency', 0.7) if attack_type == 'attack' else metrics.get('defensive_consistency', 0.7)
        
        # كلما زادت الثبات، قل الانحراف
        skewness = 2.0 * (1.0 - consistency)
        
        # إنشاء توزيع منحرف
        skewed_goals = skewnorm.rvs(skewness, loc=expected_goals, scale=expected_goals*0.3, size=1000)
        adjusted_goals = np.mean(skewed_goals)
        
        return max(0.1, adjusted_goals)
    
    def _generate_multiple_predictions(self, home_expected, away_expected, home_team, away_team):
        """إنشاء تنبؤات متعددة لكل مباراة"""
        simulations = 10000
        home_goals = np.random.poisson(home_expected, simulations)
        away_goals = np.random.poisson(away_expected, simulations)
        
        # حساب جميع النتائج الممكنة وتكراراتها
        score_counts = {}
        for h, a in zip(home_goals, away_goals):
            score = (h, a)
            score_counts[score] = score_counts.get(score, 0) + 1
        
        # الحصول على أفضل 5 تنبؤات
        top_scores = sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        predictions = []
        for score, count in top_scores:
            probability = count / simulations
            prediction_type = self._classify_prediction_type(score, probability, home_expected, away_expected)
            
            predictions.append({
                'home_goals': score[0],
                'away_goals': score[1],
                'probability': probability,
                'type': prediction_type,
                'confidence': self._calculate_score_confidence(score, probability, home_expected, away_expected)
            })
        
        return predictions
    
    def _classify_prediction_type(self, score, probability, home_expected, away_expected):
        """تصنيف نوع التنبؤ"""
        home_goals, away_goals = score
        
        if probability > 0.15:
            return "الأكثر ترجيحاً"
        elif probability > 0.08:
            if abs(home_goals - away_goals) <= 1:
                return "تنبؤ آمن"
            else:
                return "تنبوج مجازف"
        elif home_goals >= 3 or away_goals >= 3:
            return "نتيجة عالية"
        else:
            return "نتيجة منخفضة"
    
    def _calculate_score_confidence(self, score, probability, home_expected, away_expected):
        """حساب ثقة كل نتيجة على حدة"""
        home_goals, away_goals = score
        
        # ثقة تعتمد على الاحتمال والانحراف عن المتوقع
        expected_diff = abs((home_goals - away_goals) - (home_expected - away_expected))
        deviation_penalty = max(0, 1.0 - expected_diff * 0.2)
        
        return probability * deviation_penalty
    
    def _calculate_enhanced_probabilities(self, home_expected, away_expected, factors):
        """حساب احتمالات محسنة مع العوامل الخارجية"""
        max_goals = 8
        home_win = 0
        draw = 0
        away_win = 0
        
        for i in range(max_goals):
            for j in range(max_goals):
                prob = poisson.pmf(i, home_expected) * poisson.pmf(j, away_expected)
                if i > j:
                    home_win += prob
                elif i == j:
                    draw += prob
                else:
                    away_win += prob
        
        # تطبيق عوامل التعديل
        home_win *= factors['home_attack_modifier'] * (1.0 / factors['away_defense_modifier'])
        away_win *= factors['away_attack_modifier'] * (1.0 / factors['home_defense_modifier'])
        
        # تطبيع الاحتمالات
        total = home_win + draw + away_win
        home_win /= total
        draw /= total
        away_win /= total
        
        # احتمالات إضافية محسنة
        both_score = 1 - (poisson.pmf(0, home_expected) * poisson.pmf(0, away_expected))
        over_2_5 = 1 - sum(poisson.pmf(i, home_expected + away_expected) for i in range(3))
        over_1_5 = 1 - sum(poisson.pmf(i, home_expected + away_expected) for i in range(2))
        
        # تعديل بناءً على العوامل الخارجية
        both_score *= min(1.0, (factors['home_attack_modifier'] + factors['away_attack_modifier']) / 2)
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win,
            'both_teams_score': both_score,
            'over_2_5': over_2_5,
            'over_1_5': over_1_5,
            'under_2_5': 1 - over_2_5,
            'under_1_5': 1 - over_1_5
        }
    
    def _calculate_enhanced_confidence(self, home_metrics, away_metrics, factors, home_expected, away_expected):
        """خوارزمية ثقة محسنة"""
        confidence_factors = []
        
        # 1. ثبات الفريقين
        home_consistency = home_metrics.get('consistency_score', 0.7)
        away_consistency = away_metrics.get('consistency_score', 0.7)
        consistency_confidence = (home_consistency + away_consistency) / 2
        confidence_factors.append(('consistency', consistency_confidence))
        
        # 2. قوة الفريقين المطلقة
        home_strength = home_metrics.get('comprehensive_score', 50) / 100
        away_strength = away_metrics.get('comprehensive_score', 50) / 100
        strength_diff = abs(home_strength - away_strength)
        strength_confidence = min(1.0, strength_diff * 2)  # كلما زاد الفرق زادت الثقة
        confidence_factors.append(('strength_difference', strength_confidence))
        
        # 3. تأثير العوامل الخارجية
        external_impact = (
            factors['home_attack_modifier'] * 
            factors['away_attack_modifier'] * 
            factors['home_defense_modifier'] * 
            factors['away_defense_modifier']
        )
        external_confidence = 1.0 - abs(1.0 - external_impact) * 0.5
        confidence_factors.append(('external_factors', external_confidence))
        
        # 4. توقع الأهداف
        total_goals = home_expected + away_expected
        goals_confidence = min(1.0, total_goals / 3.0)  # مباريات عالية الأهداف أكثر قابلية للتنبؤ
        confidence_factors.append(('expected_goals', goals_confidence))
        
        # حساب الثقة النهائية (متوسط مرجح)
        weights = {'consistency': 0.3, 'strength_difference': 0.4, 'external_factors': 0.2, 'expected_goals': 0.1}
        total_confidence = sum(weight * confidence for (factor, confidence), weight in zip(confidence_factors, weights.values()))
        
        return {
            'overall_confidence': total_confidence,
            'factor_breakdown': dict(confidence_factors),
            'confidence_level': self._classify_confidence_level(total_confidence)
        }
    
    def _classify_confidence_level(self, confidence):
        """تصنيف مستوى الثقة"""
        if confidence >= 0.8:
            return "عالي جداً"
        elif confidence >= 0.7:
            return "عالي"
        elif confidence >= 0.6:
            return "متوسط-عالي"
        elif confidence >= 0.5:
            return "متوسط"
        elif confidence >= 0.4:
            return "منخفض-متوسط"
        else:
            return "منخفض"
    
    def _generate_recommendations(self, probabilities, predictions, confidence_metrics):
        """توليد توصيات مبنية على التحليل"""
        recommendations = []
        
        # التوصية الأساسية بناءً على أعلى احتمال
        max_prob_outcome = max(probabilities['home_win'], probabilities['draw'], probabilities['away_win'])
        
        if probabilities['home_win'] == max_prob_outcome:
            recommendations.append(f"فوز {predictions[0]['home_goals']}-{predictions[0]['away_goals']} (النتيجة الأكثر ترجيحاً)")
        elif probabilities['away_win'] == max_prob_outcome:
            recommendations.append(f"فوز {predictions[0]['away_goals']}-{predictions[0]['home_goals']} (النتيجة الأكثر ترجيحاً)")
        else:
            recommendations.append(f"تعادل {predictions[0]['home_goals']}-{predictions[0]['away_goals']} (النتيجة الأكثر ترجيحاً)")
        
        # توصيات إضافية بناءً على الاحتمالات
        if probabilities['both_teams_score'] > 0.65:
            recommendations.append("📶 تسجيل الفريقين للاهداف مرجح")
        
        if probabilities['over_2_5'] > 0.6:
            recommendations.append("⚡ مباراة عالية الأهداف متوقعة")
        elif probabilities['under_2_5'] > 0.6:
            recommendations.append("🛡️ مباراة منخفضة الأهداف متوقعة")
        
        # توصيات بناءً على مستوى الثقة
        if confidence_metrics['confidence_level'] in ["عالي جداً", "عالي"]:
            recommendations.append("✅ ثقة عالية في التنبؤ")
        elif confidence_metrics['confidence_level'] in ["منخفض-متوسط", "منخفض"]:
            recommendations.append("⚠️ تنبؤ ذو ثقة محدودة - يوصى بالحذر")
        
        return recommendations
    
    def get_prediction_accuracy(self, actual_results):
        """تحليل دقة التنبؤات السابقة"""
        correct_predictions = 0
        total_predictions = 0
        accuracy_details = []
        
        for match_id, prediction in self.prediction_history.items():
            if match_id in actual_results:
                actual = actual_results[match_id]
                predicted = prediction['predictions'][0]  # أفضل تنبؤ
                
                # التحقق من صحة التنبؤ
                is_correct = (
                    predicted['home_goals'] == actual['home_goals'] and 
                    predicted['away_goals'] == actual['away_goals']
                )
                
                correct_predictions += 1 if is_correct else 0
                total_predictions += 1
                
                accuracy_details.append({
                    'match_id': match_id,
                    'predicted': f"{predicted['home_goals']}-{predicted['away_goals']}",
                    'actual': f"{actual['home_goals']}-{actual['away_goals']}",
                    'correct': is_correct,
                    'confidence': prediction['confidence']['overall_confidence']
                })
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'overall_accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'details': accuracy_details
        }