# fixed_main.py
import pandas as pd
import numpy as np
from team_assessor.advanced_team_assessor import EnhancedTeamAssessor
from predict.realistic_predictor import RealisticMatchPredictor

def main():
    print("🚀 النظام المُصلح للتنبؤ بكرة القدم")
    print("=" * 50)
    
    data_file = "data/football-data/combined_seasons_data.csv"
    
    # الخطوة 1: التقييم الواقعي
    print("\n1. 📊 التقييم الواقعي للفرق")
    assessor = EnhancedTeamAssessor(data_file)
    
    # تحميل وتنظيف البيانات
    assessor.data['HomeTeam'] = assessor.data['HomeTeam'].astype(str).str.strip()
    assessor.data['AwayTeam'] = assessor.data['AwayTeam'].astype(str).str.strip()
    
    all_teams = set(assessor.data['HomeTeam'].unique()) | set(assessor.data['AwayTeam'].unique())
    team_assessment = {}
    
    print("🔄 جاري تقييم الفرق...")
    for team in all_teams:
        metrics = assessor.calculate_realistic_metrics(team)
        if metrics:
            score = assessor.calculate_realistic_score(metrics)
            metrics['comprehensive_score'] = score
            team_assessment[team] = metrics
    
    print(f"✅ تم تقييم {len(team_assessment)} فريق")
    
    # عرض أفضل الفرق بشكل واقعي
    top_teams = sorted(team_assessment.items(), key=lambda x: x[1]['comprehensive_score'], reverse=True)[:5]
    print("\n🏆 أفضل 5 فرق (تقييم واقعي):")
    for team, metrics in top_teams:
        print(f"• {team}: {metrics['comprehensive_score']:.1f} نقطة | "
              f"نقاط: {metrics['points_per_match']:.2f} | فوز: {metrics['win_rate']:.1%}")
    
    # الخطوة 2: التنبؤ الواقعي
    print("\n2. 🔮 التنبؤ الواقعي بالمباريات")
    predictor = RealisticMatchPredictor(team_assessment)
    
    # مباريات اختبار واقعية
    test_matches = [
        ("Man City", "Liverpool", "home"),
        ("Arsenal", "Chelsea", "home"),
        ("Man United", "Tottenham", "home"),
        ("Newcastle", "Aston Villa", "home"),
        ("Brighton", "West Ham", "home")
    ]
    
    for home_team, away_team, venue in test_matches:
        if home_team in team_assessment and away_team in team_assessment:
            print(f"\n🎯 {home_team} vs {away_team}")
            prediction = predictor.predict_match(home_team, away_team, venue)
            
            if prediction:
                home_score = prediction['score_prediction']['home_goals']
                away_score = prediction['score_prediction']['away_goals']
                conf = prediction['score_prediction']['confidence']
                
                print(f"• النتيجة: {home_score}-{away_score} (ثقة: {conf:.1%})")
                print(f"• الاحتمالات: فوز {home_team} {prediction['probabilities']['home_win']:.1%} | "
                      f"تعادل {prediction['probabilities']['draw']:.1%} | "
                      f"فوز {away_team} {prediction['probabilities']['away_win']:.1%}")
                print(f"• الأهداف المتوقعة: {prediction['expected_goals']['home']:.2f} - {prediction['expected_goals']['away']:.2f}")
                
                # تحليل إضافي
                if prediction['probabilities']['both_teams_score'] > 0.6:
                    print("• 📶 متوقع تسجيل الفريقين")
                if prediction['probabilities']['over_2_5'] > 0.6:
                    print("• ⚡ متوقع أكثر من 2.5 هدف")
    
    # إحصائيات النظام
    print("\n" + "="*50)
    print("📈 إحصائيات النظام الواقعي:")
    
    avg_points = np.mean([m['points_per_match'] for m in team_assessment.values()])
    avg_goals = np.mean([m['goals_per_match'] for m in team_assessment.values()])
    avg_win_rate = np.mean([m['win_rate'] for m in team_assessment.values()])
    
    print(f"• متوسط النقاط: {avg_points:.2f} لكل مباراة")
    print(f"• متوسط الأهداف: {avg_goals:.2f} لكل مباراة") 
    print(f"• متوسط معدل الفوز: {avg_win_rate:.1%}")
    print(f"• نطاق التقييم: {min(m['comprehensive_score'] for m in team_assessment.values()):.1f} - {max(m['comprehensive_score'] for m in team_assessment.values()):.1f}")
    
    print("\n✅ اكتمل التشغيل بنجاح!")

if __name__ == "__main__":
    main()