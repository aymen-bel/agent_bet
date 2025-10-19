# prediction_tracker.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class PredictionTracker:
    def __init__(self, tracking_file="prediction_tracking.json"):
        self.tracking_file = tracking_file
        self.predictions = self.load_predictions()
        
    def load_predictions(self):
        """تحميل التوقعات المسجلة"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def save_predictions(self):
        """حفظ التوقعات"""
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False, indent=2)
    
    def record_prediction(self, match_id, prediction_data, actual_result=None):
        """تسجيل توقع جديد"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction_data,
            'actual_result': actual_result,
            'evaluated': actual_result is not None
        }
        
        self.predictions[match_id] = prediction_record
        self.save_predictions()
        
        print(f"✅ تم تسجيل توقع للمباراة {match_id}")
    
    def update_actual_result(self, match_id, actual_result):
        """تحديث النتيجة الفعلية للمباراة"""
        if match_id in self.predictions:
            self.predictions[match_id]['actual_result'] = actual_result
            self.predictions[match_id]['evaluated'] = True
            self.save_predictions()
            print(f"✅ تم تحديث النتيجة الفعلية للمباراة {match_id}")
        else:
            print(f"❌ لم يتم العثور على توقع للمباراة {match_id}")
    
    def calculate_accuracy(self):
        """حساب دقة التوقعات"""
        evaluated_predictions = [p for p in self.predictions.values() if p['evaluated']]
        
        if not evaluated_predictions:
            return 0, 0, 0
        
        result_accuracy = 0
        goals_accuracy = 0
        total_matches = len(evaluated_predictions)
        
        for prediction in evaluated_predictions:
            pred = prediction['prediction']
            actual = prediction['actual_result']
            
            # دقة توقع النتيجة
            if (pred.get('score_prediction', {}).get('home_goals', 0) > 
                pred.get('score_prediction', {}).get('away_goals', 0) and
                actual.get('FTHG', 0) > actual.get('FTAG', 0)):
                result_accuracy += 1
            elif (pred.get('score_prediction', {}).get('home_goals', 0) < 
                  pred.get('score_prediction', {}).get('away_goals', 0) and
                  actual.get('FTHG', 0) < actual.get('FTAG', 0)):
                result_accuracy += 1
            elif (pred.get('score_prediction', {}).get('home_goals', 0) == 
                  pred.get('score_prediction', {}).get('away_goals', 0) and
                  actual.get('FTHG', 0) == actual.get('FTAG', 0)):
                result_accuracy += 1
            
            # دقة توقع الأهداف
            goals_error = (abs(pred.get('score_prediction', {}).get('home_goals', 0) - actual.get('FTHG', 0)) +
                          abs(pred.get('score_prediction', {}).get('away_goals', 0) - actual.get('FTAG', 0)))
            goals_accuracy += (4 - min(goals_error, 3)) / 4  # تسجيل من 0-1
        
        result_accuracy /= total_matches
        goals_accuracy /= total_matches
        overall_accuracy = (result_accuracy + goals_accuracy) / 2
        
        return result_accuracy, goals_accuracy, overall_accuracy
    
    def generate_performance_report(self):
        """توليد تقرير أداء التوقعات"""
        result_acc, goals_acc, overall_acc = self.calculate_accuracy()
        total_predictions = len(self.predictions)
        evaluated_predictions = len([p for p in self.predictions.values() if p['evaluated']])
        
        print("\n" + "="*80)
        print("📊 تقرير أداء التوقعات")
        print("="*80)
        
        print(f"\n📈 الإحصائيات العامة:")
        print(f"• إجمالي التوقعات: {total_predictions}")
        print(f"• التوقعات المُقيمة: {evaluated_predictions}")
        print(f"• دقة توقع النتيجة: {result_acc:.1%}")
        print(f"• دقة توقع الأهداف: {goals_acc:.1%}")
        print(f"• الدقة الشاملة: {overall_acc:.1%}")
        
        if evaluated_predictions > 0:
            print(f"\n🔍 تحليل التوقعات الحديثة:")
            recent_predictions = sorted(
                [p for p in self.predictions.values() if p['evaluated']],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:5]
            
            for i, pred in enumerate(recent_predictions, 1):
                match_pred = pred['prediction']
                actual = pred['actual_result']
                
                home_team = match_pred.get('home_team', 'Unknown')
                away_team = match_pred.get('away_team', 'Unknown')
                pred_score = f"{match_pred.get('score_prediction', {}).get('home_goals', 0)}-{match_pred.get('score_prediction', {}).get('away_goals', 0)}"
                actual_score = f"{actual.get('FTHG', 0)}-{actual.get('FTAG', 0)}"
                
                result_correct = "✅" if pred_score == actual_score else "❌"
                print(f"  {i}. {home_team} vs {away_team}: {pred_score} (فعلي: {actual_score}) {result_correct}")

# استخدام متكامل
if __name__ == "__main__":
    # إنشاء متعقب التوقعات
    tracker = PredictionTracker()
    
    # مثال على تسجيل توقع
    sample_prediction = {
        'home_team': 'Man City',
        'away_team': 'Liverpool', 
        'score_prediction': {'home_goals': 2, 'away_goals': 1},
        'scenario_probabilities': {'home_win': 0.6, 'draw': 0.2, 'away_win': 0.2}
    }
    
    sample_actual = {
        'FTHG': 2,
        'FTAG': 1,
        'FTR': 'H'
    }
    
    # تسجيل التوقع والنتيجة
    match_id = f"Man City_vs_Liverpool_{datetime.now().strftime('%Y%m%d')}"
    tracker.record_prediction(match_id, sample_prediction, sample_actual)
    
    # عرض تقرير الأداء
    tracker.generate_performance_report()