# main_integration.py
import os
import pandas as pd
from comprehensive_team_assessment import ComprehensiveTeamAssessment
from match_scenario_predictor import MatchScenarioPredictor
from model_optimizer import ModelOptimizer
from prediction_tracker import PredictionTracker
from datetime import datetime

def main():
    print("🚀 النظام المتكامل للتحليل والتنبؤ بكرة القدم")
    print("=" * 60)
    
    # المسار إلى البيانات
    data_file = "data/football-data/combined_seasons_data.csv"
    
    if not os.path.exists(data_file):
        print(f"❌ ملف البيانات غير موجود: {data_file}")
        print("🔧 يرجى التأكد من وجود البيانات في المسار الصحيح")
        return
    
    # الخطوة 1: تقييم الفرق
    print("\n📍 المرحلة 1: تقييم الفرق")
    print("-" * 40)
    
    assessor = ComprehensiveTeamAssessment(data_file)
    assessor.preprocess_data()
    assessor.assess_all_teams()
    ranking = assessor.create_final_ranking()
    assessor.save_results()
    assessor.generate_detailed_report()
    
    # الخطوة 2: التنبؤ بالمباريات
    print("\n📍 المرحلة 2: التنبؤ بسيناريو المباريات")
    print("-" * 40)
    
    # تحويل التقييم إلى قاموس للتنبؤ
    team_assessment_dict = {}
    for _, row in ranking.iterrows():
        team_assessment_dict[row['Team']] = row.to_dict()
    
    predictor = MatchScenarioPredictor(team_assessment_dict)
    
    # تحليل مباريات مهمة
    important_matches = [
        ("Man City", "Liverpool", "home"),
        ("Arsenal", "Chelsea", "home"), 
        ("Man United", "Tottenham", "home"),
        ("Newcastle", "Aston Villa", "home"),
        ("Brighton", "West Ham", "home")
    ]
    
    tracker = PredictionTracker()
    
    for home_team, away_team, venue in important_matches:
        if home_team in team_assessment_dict and away_team in team_assessment_dict:
            print(f"\n🎯 جاري تحليل: {home_team} vs {away_team}")
            prediction = predictor.predict_match_scenario(home_team, away_team, venue)
            
            if prediction:
                # تسجيل التوقع للمتابعة
                match_id = f"{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                tracker.record_prediction(match_id, prediction)
                
                # عرض التقرير
                predictor.generate_match_report(home_team, away_team, venue)
    
    # الخطوة 3: تحسين النموذج
    print("\n📍 المرحلة 3: تحسين النموذج والتعلم")
    print("-" * 40)
    
    optimizer = ModelOptimizer(data_file)
    optimizer.preprocess_historical_data()
    features_df, results_df, goals_df = optimizer.prepare_training_data()
    
    if len(features_df) > 100:  # إذا كانت البيانات كافية
        print("📊 تدريب نماذج التعلم الآلي...")
        optimizer.train_result_predictor(features_df, results_df)
        optimizer.train_goals_predictor(features_df, goals_df)
        optimizer.generate_optimization_report()
        optimizer.save_optimized_model()
    else:
        print("⚠️  البيانات غير كافية لتدريب النماذج المتقدمة")
        print("💡 استمر في جمع المزيد من البيانات")
    
    # الخطوة 4: تقرير الأداء
    print("\n📍 المرحلة 4: تقرير الأداء الشامل")
    print("-" * 40)
    
    tracker.generate_performance_report()
    
    print("\n✅ اكتمل التشغيل بنجاح!")
    print("\n📁 الملفات الناتجة:")
    print("   • comprehensive_team_ranking.csv - ترتيب الفرق")
    print("   • complete_team_metrics.csv - جميع المؤشرات") 
    print("   • comprehensive_team_assessment.png - الرسوم البيانية")
    print("   • optimized_match_predictor.pkl - النموذج المحسن")
    print("   • prediction_tracking.json - تتبع التوقعات")

if __name__ == "__main__":
    main()