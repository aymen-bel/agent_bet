# weekly_update.py
from comprehensive_team_assessment import ComprehensiveTeamAssessment
from model_optimizer import ModelOptimizer
import os

def weekly_model_update():
    print("🔄 التحديث الأسبوعي للنموذج")
    print("=" * 40)
    
    data_file = "data/football-data/combined_seasons_data.csv"
    
    if not os.path.exists(data_file):
        print("❌ ملف البيانات غير موجود")
        return
    
    # 1. تحديث تقييم الفرق
    print("📊 تحديث تقييم الفرق...")
    assessor = ComprehensiveTeamAssessment(data_file)
    assessor.preprocess_data()
    assessor.assess_all_teams()
    assessor.create_final_ranking()
    assessor.save_results()
    
    # 2. تحسين النموذج
    print("🤖 تحسين النموذج بالبيانات الجديدة...")
    optimizer = ModelOptimizer(data_file)
    optimizer.preprocess_historical_data()
    features_df, results_df, goals_df = optimizer.prepare_training_data()
    
    if len(features_df) > 100:
        optimizer.train_result_predictor(features_df, results_df)
        optimizer.train_goals_predictor(features_df, goals_df)
        optimizer.generate_optimization_report()
        optimizer.save_optimized_model()
        print("✅ تم تحديث النموذج بنجاح")
    else:
        print("⚠️  البيانات غير كافية للتحديث")
    
    print("🎉 اكتمل التحديث الأسبوعي")

if __name__ == "__main__":
    weekly_model_update()