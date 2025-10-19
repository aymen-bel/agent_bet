# daily_analysis.py
import pandas as pd
from datetime import datetime
from match_scenario_predictor import MatchScenarioPredictor
from prediction_tracker import PredictionTracker

def analyze_todays_matches():
    print(f"📅 تحليل مباريات اليوم: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 50)
    
    # تحميل أحدث تقييم للفرق
    try:
        team_data = pd.read_csv("comprehensive_team_ranking.csv")
        team_assessment = {}
        
        for _, row in team_data.iterrows():
            team_assessment[row['Team']] = row.to_dict()
        
        predictor = MatchScenarioPredictor(team_assessment)
        tracker = PredictionTracker()
        
        # قائمة مباريات اليوم (يمكن جلبها من مصدر خارجي)
        todays_matches = [
            ("Man City", "Liverpool", "home"),
            ("Arsenal", "Chelsea", "home"),
            # أضف المباريات الأخرى هنا
        ]
        
        print(f"🔍 عدد المباريات اليوم: {len(todays_matches)}")
        
        for home_team, away_team, venue in todays_matches:
            if home_team in team_assessment and away_team in team_assessment:
                print(f"\n🎯 {home_team} vs {away_team}")
                prediction = predictor.generate_match_report(home_team, away_team, venue)
                
                # تسجيل التوقع
                if prediction:
                    match_id = f"{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d')}"
                    tracker.record_prediction(match_id, prediction)
        
        # عرض ملخص
        print(f"\n📊 ملخص اليوم:")
        tracker.generate_performance_report()
        
    except FileNotFoundError:
        print("❌ لم يتم العثور على بيانات التقييم. يرجى تشغيل comprehensive_team_assessment.py أولاً")

if __name__ == "__main__":
    analyze_todays_matches()