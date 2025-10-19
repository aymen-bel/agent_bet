# integrated_system.py
from comprehensive_team_assessment import ComprehensiveTeamAssessment
from match_scenario_predictor import MatchScenarioPredictor  
from model_optimizer import ModelOptimizer
from prediction_tracker import PredictionTracker
from datetime import datetime
class IntegratedFootballSystem:
    def __init__(self):
        self.team_assessor = None
        self.predictor = None
        self.optimizer = None
        self.tracker = PredictionTracker()
        
    def run_complete_analysis(self, historical_data_file, teams_to_analyze=None):
        """تشغيل التحليل الشامل"""
        print("🚀 بدء النظام المتكامل للتحليل والتنبؤ...")
        
        # 1. تقييم الفرق
        print("\n📊 المرحلة 1: تقييم الفرق...")
        self.team_assessor = ComprehensiveTeamAssessment(historical_data_file)
        self.team_assessor.preprocess_data()
        self.team_assessor.assess_all_teams()
        team_ranking = self.team_assessor.create_final_ranking()
        
        # 2. إنشاء predictor
        print("\n🔮 المرحلة 2: إعداد نظام التنبؤ...")
        team_assessment_dict = {}
        for _, row in team_ranking.iterrows():
            team_assessment_dict[row['Team']] = row.to_dict()
        
        self.predictor = MatchScenarioPredictor(team_assessment_dict)
        
        # 3. تحليل مباريات مثال
        if teams_to_analyze:
            print("\n🎯 المرحلة 3: تحليل المباريات...")
            for home_team, away_team in teams_to_analyze:
                prediction = self.predictor.generate_match_report(home_team, away_team, "home")
                
                # تسجيل التوقع للمتابعة
                match_id = f"{home_team}_vs_{away_team}_{datetime.now().strftime('%Y%m%d')}"
                self.tracker.record_prediction(match_id, prediction)
        
        # 4. تحسين النموذج
        print("\n🔄 المرحلة 4: تحسين النموذج...")
        self.optimizer = ModelOptimizer(historical_data_file)
        self.optimizer.preprocess_historical_data()
        
        # إذا كانت هناك توقعات مسجلة، تحليل الأخطاء
        if len(self.tracker.predictions) > 0:
            # يمكن إضافة تحليل الأخطاء هنا عندما تتوفر النتائج الفعلية
            pass
        
        print("\n✅ اكتمل التحليل الشامل!")
        return self

# استخدام النظام
if __name__ == "__main__":
    system = IntegratedFootballSystem()
    sample_matches = [('Man City', 'Liverpool'), ('Arsenal', 'Chelsea')]
    system.run_complete_analysis("data/football-data/combined_seasons_data.csv", sample_matches)