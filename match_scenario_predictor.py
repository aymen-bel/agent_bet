# match_scenario_predictor.py
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

class MatchScenarioPredictor:
    def __init__(self, team_assessment_data):
        self.team_data = team_assessment_data
        self.scenario_probabilities = {}
        
    def predict_match_scenario(self, home_team, away_team, venue="home"):
        """التنبؤ بسيناريو المباراة بين فريقين"""
        if home_team not in self.team_data or away_team not in self.team_data:
            print(f"❌ أحد الفريقين غير موجود في البيانات: {home_team} أو {away_team}")
            return None
            
        home_metrics = self.team_data[home_team]
        away_metrics = self.team_data[away_team]
        
        # حساب القوة النسبية
        relative_strength = self._calculate_relative_strength(home_metrics, away_metrics, venue)
        
        # تحليل نقاط القوة والضعف
        strength_analysis = self._analyze_strengths_weaknesses(home_metrics, away_metrics)
        
        # التنبؤ بالنتيجة
        score_prediction = self._predict_score(home_metrics, away_metrics, venue)
        
        # احتمالات السيناريوهات
        scenarios = self._calculate_scenario_probabilities(home_metrics, away_metrics, venue)
        
        # توصيات تكتيكية
        tactical_recommendations = self._generate_tactical_recommendations(home_metrics, away_metrics)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'venue': venue,
            'relative_strength': relative_strength,
            'strength_analysis': strength_analysis,
            'score_prediction': score_prediction,
            'scenario_probabilities': scenarios,
            'tactical_recommendations': tactical_recommendations,
            'key_insights': self._generate_key_insights(home_metrics, away_metrics, relative_strength)
        }
    
    def _calculate_relative_strength(self, home_metrics, away_metrics, venue):
        """حساب القوة النسبية بين الفريقين"""
        # تصحيح القوة حسب الملعب
        if venue == "home":
            home_strength = home_metrics['comprehensive_score'] * (1 + home_metrics.get('home_advantage', 0) * 0.3)
            away_strength = away_metrics['comprehensive_score'] * (1 - away_metrics.get('away_resilience', 0) * 0.2)
        else:
            home_strength = home_metrics['comprehensive_score'] * (1 - home_metrics.get('home_advantage', 0) * 0.2)
            away_strength = away_metrics['comprehensive_score'] * (1 + away_metrics.get('away_resilience', 0) * 0.3)
        
        total_strength = home_strength + away_strength
        home_relative = home_strength / total_strength if total_strength > 0 else 0.5
        away_relative = away_strength / total_strength if total_strength > 0 else 0.5
        
        return {
            'home_strength': home_strength,
            'away_strength': away_strength,
            'home_relative': home_relative,
            'away_relative': away_relative,
            'strength_difference': home_strength - away_strength
        }
    
    def _analyze_strengths_weaknesses(self, home_metrics, away_metrics):
        """تحليل نقاط القوة والضعف"""
        # مقارنة المجالات الرئيسية
        comparisons = {
            'attack': self._compare_attack(home_metrics, away_metrics),
            'defense': self._compare_defense(home_metrics, away_metrics),
            'consistency': self._compare_consistency(home_metrics, away_metrics),
            'tactical': self._compare_tactical(home_metrics, away_metrics)
        }
        
        # تحديد المزايا التنافسية
        competitive_advantages = self._identify_competitive_advantages(comparisons)
        
        return {
            'comparisons': comparisons,
            'competitive_advantages': competitive_advantages,
            'key_matchups': self._identify_key_matchups(home_metrics, away_metrics)
        }
    
    def _compare_attack(self, home_metrics, away_metrics):
        """مقارنة القوة الهجومية"""
        home_attack = (
            home_metrics.get('shot_efficiency', 0) * 0.3 +
            home_metrics.get('conversion_rate', 0) * 0.3 +
            home_metrics.get('attacking_pressure', 0) * 0.2 +
            home_metrics.get('expected_goals_ratio', 1) * 0.2
        )
        
        away_attack = (
            away_metrics.get('shot_efficiency', 0) * 0.3 +
            away_metrics.get('conversion_rate', 0) * 0.3 +
            away_metrics.get('attacking_pressure', 0) * 0.2 +
            away_metrics.get('expected_goals_ratio', 1) * 0.2
        )
        
        return {
            'home_attack': home_attack,
            'away_attack': away_attack,
            'advantage': 'home' if home_attack > away_attack else 'away',
            'difference': abs(home_attack - away_attack)
        }
    
    def _compare_defense(self, home_metrics, away_metrics):
        """مقارنة القوة الدفاعية"""
        home_defense = (
            home_metrics.get('defensive_efficiency', 0) * 0.4 +
            home_metrics.get('clean_sheet_rate', 0) * 0.3 +
            home_metrics.get('defensive_stability', 0) * 0.3
        )
        
        away_defense = (
            away_metrics.get('defensive_efficiency', 0) * 0.4 +
            away_metrics.get('clean_sheet_rate', 0) * 0.3 +
            away_metrics.get('defensive_stability', 0) * 0.3
        )
        
        return {
            'home_defense': home_defense,
            'away_defense': away_defense,
            'advantage': 'home' if home_defense > away_defense else 'away',
            'difference': abs(home_defense - away_defense)
        }
    
    def _compare_consistency(self, home_metrics, away_metrics):
        """مقارنة الاتساق"""
        home_consistency = (
            home_metrics.get('results_consistency', 0.5) * 0.4 +
            home_metrics.get('performance_consistency', 0.5) * 0.3 +
            home_metrics.get('form_momentum', 0.5) * 0.3
        )
        
        away_consistency = (
            away_metrics.get('results_consistency', 0.5) * 0.4 +
            away_metrics.get('performance_consistency', 0.5) * 0.3 +
            away_metrics.get('form_momentum', 0.5) * 0.3
        )
        
        return {
            'home_consistency': home_consistency,
            'away_consistency': away_consistency,
            'advantage': 'home' if home_consistency > away_consistency else 'away',
            'difference': abs(home_consistency - away_consistency)
        }
    
    def _compare_tactical(self, home_metrics, away_metrics):
        """مقارنة الجوانب التكتيكية"""
        home_tactical = (
            home_metrics.get('possession_control', 0.5) * 0.4 +
            home_metrics.get('tactical_discipline', 0.5) * 0.3 +
            home_metrics.get('game_management', 0.5) * 0.3
        )
        
        away_tactical = (
            away_metrics.get('possession_control', 0.5) * 0.4 +
            away_metrics.get('tactical_discipline', 0.5) * 0.3 +
            away_metrics.get('game_management', 0.5) * 0.3
        )
        
        return {
            'home_tactical': home_tactical,
            'away_tactical': away_tactical,
            'advantage': 'home' if home_tactical > away_tactical else 'away',
            'difference': abs(home_tactical - away_tactical)
        }
    
    def _identify_competitive_advantages(self, comparisons):
        """تحديد المزايا التنافسية"""
        advantages = []
        threshold = 0.1  # حد الفارق الكبير
        
        for area, comparison in comparisons.items():
            if comparison['difference'] > threshold:
                advantages.append({
                    'area': area,
                    'advantage': comparison['advantage'],
                    'magnitude': comparison['difference']
                })
        
        return advantages
    
    def _identify_key_matchups(self, home_metrics, away_metrics):
        """تحديد المواجهات الحاسمة"""
        matchups = []
        
        # هجوم المنزل vs دفاع الخارج
        home_attack_vs_away_defense = home_metrics.get('shot_efficiency', 0) - away_metrics.get('defensive_efficiency', 0)
        if abs(home_attack_vs_away_defense) > 0.15:
            matchups.append({
                'type': 'home_attack_vs_away_defense',
                'advantage': 'home' if home_attack_vs_away_defense > 0 else 'away',
                'impact': abs(home_attack_vs_away_defense)
            })
        
        # هجوم الخارج vs دفاع المنزل
        away_attack_vs_home_defense = away_metrics.get('shot_efficiency', 0) - home_metrics.get('defensive_efficiency', 0)
        if abs(away_attack_vs_home_defense) > 0.15:
            matchups.append({
                'type': 'away_attack_vs_home_defense',
                'advantage': 'away' if away_attack_vs_home_defense > 0 else 'home',
                'impact': abs(away_attack_vs_home_defense)
            })
        
        return matchups
    
    def _predict_score(self, home_metrics, away_metrics, venue):
        """التنبؤ بالنتيجة"""
        # حساب متوسط الأهداف المتوقعة
        if venue == "home":
            home_expected = home_metrics.get('goals_per_match', 1.5) * (1 + home_metrics.get('home_advantage', 0) * 0.5)
            away_expected = away_metrics.get('goals_per_match', 1.2) * (1 - away_metrics.get('away_resilience', 0) * 0.3)
        else:
            home_expected = home_metrics.get('goals_per_match', 1.5) * (1 - home_metrics.get('home_advantage', 0) * 0.3)
            away_expected = away_metrics.get('goals_per_match', 1.2) * (1 + away_metrics.get('away_resilience', 0) * 0.5)
        
        # تطبيق توزيع بواسون للتنبؤ بالنتائج (مع محاكاة متعددة لمزيد من الدقة)
        simulations = 1000
        home_goals_sim = np.random.poisson(home_expected, simulations)
        away_goals_sim = np.random.poisson(away_expected, simulations)
        
        # تحسين التنبؤ بناءً على القوة النسبية
        strength_factor = self._calculate_relative_strength(home_metrics, away_metrics, venue)
        home_goals_sim = home_goals_sim * (0.8 + strength_factor['home_relative'] * 0.4)
        away_goals_sim = away_goals_sim * (0.8 + strength_factor['away_relative'] * 0.4)
        
        # أخذ المتوسط من المحاكاة
        home_goals = int(round(np.mean(home_goals_sim)))
        away_goals = int(round(np.mean(away_goals_sim)))
        
        return {
            'home_goals': max(0, home_goals),
            'away_goals': max(0, away_goals),
            'expected_home_goals': home_expected,
            'expected_away_goals': away_expected
        }
    
    def _calculate_scenario_probabilities(self, home_metrics, away_metrics, venue):
        """حساب احتمالات السيناريوهات المختلفة"""
        strength = self._calculate_relative_strength(home_metrics, away_metrics, venue)
        
        # احتمالات الفوز الأساسية
        base_home_win = 0.4 + strength['home_relative'] * 0.3
        base_away_win = 0.3 + strength['away_relative'] * 0.3
        base_draw = 0.3
        
        # تعديل بناءً على الاتساق
        consistency_factor = (home_metrics.get('results_consistency', 0.5) + away_metrics.get('results_consistency', 0.5)) / 2
        home_win_prob = base_home_win * (0.9 + consistency_factor * 0.2)
        away_win_prob = base_away_win * (0.9 + consistency_factor * 0.2)
        draw_prob = base_draw * (0.9 + (1 - consistency_factor) * 0.2)
        
        # تطبيع الاحتمالات
        total = home_win_prob + away_win_prob + draw_prob
        home_win_prob /= total
        away_win_prob /= total
        draw_prob /= total
        
        return {
            'home_win': home_win_prob,
            'away_win': away_win_prob,
            'draw': draw_prob,
            'both_teams_score': self._calculate_bts_probability(home_metrics, away_metrics),
            'over_2_5_goals': self._calculate_over_under_probability(home_metrics, away_metrics, 2.5),
            'clean_sheet_home': home_metrics.get('clean_sheet_rate', 0.2),
            'clean_sheet_away': away_metrics.get('clean_sheet_rate', 0.2)
        }
    
    def _calculate_bts_probability(self, home_metrics, away_metrics):
        """احتمال تسجيل الفريقين"""
        home_scoring_prob = min(0.9, home_metrics.get('goals_per_match', 1.5) / 2)
        away_scoring_prob = min(0.9, away_metrics.get('goals_per_match', 1.2) / 2)
        return home_scoring_prob * away_scoring_prob
    
    def _calculate_over_under_probability(self, home_metrics, away_metrics, threshold):
        """احتمال أكثر/أقل من عدد أهداف معين"""
        total_expected = home_metrics.get('goals_per_match', 1.5) + away_metrics.get('goals_per_match', 1.2)
        return min(0.95, total_expected / threshold)
    
    def _generate_tactical_recommendations(self, home_metrics, away_metrics):
        """توليد توصيات تكتيكية"""
        recommendations = []
        
        # توصيات للفريق المنزل
        if home_metrics.get('attacking_pressure', 0) > away_metrics.get('defensive_efficiency', 0):
            recommendations.append("الضغط الهجومي المستمر يمكن أن يكون فعالاً ضد دفاع الخصم")
        
        if away_metrics.get('disciplinary_index', 0) > 0.3:
            recommendations.append("استغلال ضعف انضباط الخصم عبر الهجمات السريعة")
        
        if home_metrics.get('possession_control', 0.5) < away_metrics.get('possession_control', 0.5):
            recommendations.append("الاعتماد على الهجمات المرتدة بدلاً من السيطرة الكروية")
        
        # توصيات للفريق الخارج
        if away_metrics.get('away_resilience', 0.5) < 0.4:
            recommendations.append("الخارج يحتاج لتحصين الدفاع في الدقائق الأولى")
        
        # توصيات إضافية
        if home_metrics.get('shot_efficiency', 0) > 0.4:
            recommendations.append("الاستفادة من الكفاءة العالية في التسجيل عبر خلق فرص واضحة")
        
        if away_metrics.get('defensive_consistency', 0.5) < 0.6:
            recommendations.append("استهداف نقاط الضعف الدفاعية المتكررة للخصم")
        
        return recommendations
    
    def _generate_key_insights(self, home_metrics, away_metrics, relative_strength):
        """توليد رؤى رئيسية"""
        insights = []
        
        # رؤى بناءً على القوة النسبية
        if relative_strength['strength_difference'] > 20:
            insights.append("تفوق واضح للفريق المنزل يتوقع سيطرته على مجريات المباراة")
        elif abs(relative_strength['strength_difference']) < 5:
            insights.append("مواجهة متكافئة يتوقع أن تحسمها التفاصيل الصغيرة")
        else:
            insights.append("مباراة متوازنة مع تقدم طفيف لأحد الفريقين")
        
        # رؤى هجومية
        if home_metrics.get('conversion_rate', 0) > 0.4:
            insights.append("كفاءة تسجيل عالية للفريق المنزل في الفرص المتاحة")
        
        if away_metrics.get('defensive_efficiency', 0) < 0.6:
            insights.append("ضعف في الكفاءة الدفاعية للفريق الخارج")
        
        # رؤى تكتيكية
        if home_metrics.get('form_momentum', 0.5) > 0.7:
            insights.append("زخم إيجابي قوي للفريق المنزل في المباريات الأخيرة")
        
        if away_metrics.get('form_momentum', 0.5) > 0.7:
            insights.append("الفريق الخارج في حالة ممتازة ويحقق نتائج إيجابية")
        
        # رؤى دفاعية
        if home_metrics.get('clean_sheet_rate', 0) > 0.4:
            insights.append("قوة دفاعية ملحوظة للفريق المنزل في الحفاظ على نظافة شباكه")
        
        return insights
    
    def generate_match_report(self, home_team, away_team, venue="home"):
        """توليد تقرير مفصل عن المباراة"""
        print(f"\n🎯 جاري تحليل المباراة: {home_team} vs {away_team}")
        
        prediction = self.predict_match_scenario(home_team, away_team, venue)
        
        if not prediction:
            print("❌ تعذر توليد التقرير. تأكد من وجود الفريقين في البيانات.")
            return None
        
        print(f"\n{'='*80}")
        print(f"📊 تقرير المباراة: {home_team} vs {away_team}")
        print(f"{'='*80}")
        
        print(f"\n🏆 القوة النسبية:")
        print(f"• قوة {home_team}: {prediction['relative_strength']['home_strength']:.1f}")
        print(f"• قوة {away_team}: {prediction['relative_strength']['away_strength']:.1f}")
        print(f"• الفارق: {prediction['relative_strength']['strength_difference']:+.1f}")
        
        print(f"\n🎯 التنبؤ بالنتيجة:")
        print(f"• {home_team} {prediction['score_prediction']['home_goals']} - {prediction['score_prediction']['away_goals']} {away_team}")
        print(f"• الأهداف المتوقعة: {prediction['score_prediction']['expected_home_goals']:.2f} - {prediction['score_prediction']['expected_away_goals']:.2f}")
        
        print(f"\n📈 احتمالات السيناريوهات:")
        scenarios = prediction['scenario_probabilities']
        print(f"• فوز {home_team}: {scenarios['home_win']:.1%}")
        print(f"• تعادل: {scenarios['draw']:.1%}")
        print(f"• فوز {away_team}: {scenarios['away_win']:.1%}")
        print(f"• تسجيل الفريقين: {scenarios['both_teams_score']:.1%}")
        print(f"• أكثر من 2.5 هدف: {scenarios['over_2_5_goals']:.1%}")
        print(f"• نظافة شباك {home_team}: {scenarios['clean_sheet_home']:.1%}")
        print(f"• نظافة شباك {away_team}: {scenarios['clean_sheet_away']:.1%}")
        
        print(f"\n⚔️ نقاط القوة والضعف:")
        analysis = prediction['strength_analysis']
        for area, comp in analysis['comparisons'].items():
            advantage_team = comp['advantage']
            team_name = home_team if advantage_team == 'home' else away_team
            print(f"• {area}: الأفضلية لـ {team_name} (فارق: {comp['difference']:.3f})")
        
        if analysis['competitive_advantages']:
            print(f"\n💎 المزايا التنافسية:")
            for advantage in analysis['competitive_advantages']:
                team_name = home_team if advantage['advantage'] == 'home' else away_team
                print(f"• {advantage['area']}: {team_name} (قوة: {advantage['magnitude']:.3f})")
        
        print(f"\n💡 التوصيات التكتيكية:")
        if prediction['tactical_recommendations']:
            for i, recommendation in enumerate(prediction['tactical_recommendations'], 1):
                print(f"  {i}. {recommendation}")
        else:
            print("  لا توجد توصيات تكتيكية محددة")
        
        print(f"\n🔍 الرؤى الرئيسية:")
        for i, insight in enumerate(prediction['key_insights'], 1):
            print(f"  {i}. {insight}")
        
        return prediction

    def analyze_multiple_matches(self, matches):
        """تحليل عدة مباريات في وقت واحد"""
        results = []
        for match in matches:
            home_team, away_team, venue = match
            result = self.predict_match_scenario(home_team, away_team, venue)
            if result:
                results.append(result)
                self.generate_match_report(home_team, away_team, venue)
        return results

# التشغيل الرئيسي المحسن
if __name__ == "__main__":
    print("🚀 بدء نظام التنبؤ بسيناريو المباريات...")
    
    try:
        # تحميل بيانات التقييم
        if not os.path.exists("comprehensive_team_ranking.csv"):
            print("❌ ملف التقييم غير موجود. يرجى تشغيل comprehensive_team_assessment.py أولاً")
            exit()
        
        team_data = pd.read_csv("comprehensive_team_ranking.csv")
        team_assessment = {}
        
        print(f"📁 جاري تحميل بيانات {len(team_data)} فريق...")
        
        for _, row in team_data.iterrows():
            team_assessment[row['Team']] = row.to_dict()
        
        # إنشاء مفسر السيناريو
        predictor = MatchScenarioPredictor(team_assessment)
        
        # عرض بعض الفرق المتاحة للمستخدم
        available_teams = list(team_assessment.keys())[:10]
        print(f"\n🔍 أمثلة على الفرق المتاحة: {', '.join(available_teams)}")
        
        # تحليل مباريات مثال مختلفة
        print(f"\n{'='*50}")
        print("🎯 تحليل مباريات مثال")
        print(f"{'='*50}")
        
        # استخدام أول فريقين في القائمة كمثال
        if len(available_teams) >= 2:
            team1, team2 = available_teams[0], available_teams[1]
            print(f"\n📊 المباراة 1: {team1} vs {team2}")
            report1 = predictor.generate_match_report(team1, team2, "home")
            
            if len(available_teams) >= 4:
                team3, team4 = available_teams[2], available_teams[3]
                print(f"\n{'='*50}")
                print(f"📊 المباراة 2: {team3} vs {team4}")
                report2 = predictor.generate_match_report(team3, team4, "home")
        
        print(f"\n✅ تم الانتهاء من تحليل المباريات!")
        
    except Exception as e:
        print(f"❌ حدث خطأ: {e}")
        print("🔧 تأكد من:")
        print("   - وجود ملف comprehensive_team_ranking.csv")
        print("   - تشغيل comprehensive_team_assessment.py أولاً")
        print("   - صحة格式 البيانات في الملف")